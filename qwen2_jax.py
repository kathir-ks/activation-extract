"""
Qwen 2.5 0.5B Model JAX Implementation for TPUs

Requirements:
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install transformers flax torch einops
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import struct
from typing import Optional, Tuple, Dict, Any
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
import math
from functools import partial
import time

# Import KV cache utilities
from kvcache_utils import (
    create_kv_cache_buffers,
    update_kv_cache_ar,
    write_prefill_cache,
    get_attention_kv,
    KVCacheConfig
)

@struct.dataclass
class QwenConfig:
    """Configuration for Qwen model."""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 32768
    use_fixed_cache: bool = False  # Use MaxText-style fixed-size cache
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors."""
    # q, k shape: (batch, num_heads, seq_len, head_dim)
    # cos, sin shape: (max_seq_len, head_dim)
    
    # Get cos and sin for the specific positions
    cos = jnp.take(cos, position_ids, axis=0)  # (seq_len, head_dim)
    sin = jnp.take(sin, position_ids, axis=0)  # (seq_len, head_dim)
    
    # Reshape to (1, 1, seq_len, head_dim) for broadcasting
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """RMSNorm implementation."""
    dim: int
    eps: float = 1e-6
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.dim,), self.dtype)
        variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.eps)
        return (weight * x).astype(self.dtype)


class QwenMLP(nn.Module):
    """Qwen MLP block."""
    config: QwenConfig
    
    @nn.compact
    def __call__(self, x):
        gate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='gate_proj'
        )(x)
        up_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='up_proj'
        )(x)
        
        hidden_states = nn.silu(gate_proj) * up_proj
        
        down_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='down_proj'
        )(hidden_states)
        
        return down_proj


class QwenAttention(nn.Module):
    """Multi-headed attention with grouped query attention support."""
    config: QwenConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        # Pre-compute and cache RoPE embeddings
        dim = self.head_dim
        max_seq_len = self.max_position_embeddings
        inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
        t = jnp.arange(max_seq_len).astype(jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.rope_cos = jnp.cos(emb)
        self.rope_sin = jnp.sin(emb)

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None, position_offset=0):
        batch_size, seq_len, _ = hidden_states.shape

        q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='q_proj'
        )(hidden_states)
        k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='k_proj'
        )(hidden_states)
        v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='v_proj'
        )(hidden_states)

        # Reshape for attention computation
        q = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply rotary embeddings (use cached values)
        # Use dynamic_slice instead of arange to avoid tracer issues
        # Create position_ids by slicing the pre-computed position array
        all_positions = jnp.arange(self.max_position_embeddings)
        position_ids = lax.dynamic_slice(all_positions, (position_offset,), (seq_len,))
        q, k = apply_rotary_pos_emb(q, k, self.rope_cos, self.rope_sin, position_ids)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = jnp.concatenate([k_cache, k], axis=2)
            v = jnp.concatenate([v_cache, v], axis=2)

        new_kv_cache = (k, v)

        # Repeat k/v heads if using GQA
        if self.num_key_value_groups > 1:
            k = jnp.repeat(k, self.num_key_value_groups, axis=1)
            v = jnp.repeat(v, self.num_key_value_groups, axis=1)

        # Compute attention scores
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.softmax(attn_weights, axis=-1).astype(self.config.dtype)
        attn_output = jnp.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='o_proj'
        )(attn_output)

        return attn_output, new_kv_cache


class QwenAttentionFixed(nn.Module):
    """
    Multi-headed attention with fixed-size KV cache (MaxText style)

    Uses pre-allocated buffers and dynamic updates instead of concatenation.
    JIT-friendly with static shapes throughout generation.
    """
    config: QwenConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        # Pre-compute and cache RoPE embeddings
        dim = self.head_dim
        max_seq_len = self.max_position_embeddings
        inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
        t = jnp.arange(max_seq_len).astype(jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.rope_cos = jnp.cos(emb)
        self.rope_sin = jnp.sin(emb)

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, cache_dict=None, layer_idx=0,
                 position_offset=0, is_prefill=False):
        """
        Forward pass with fixed-size KV cache

        Args:
            hidden_states: Input [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            cache_dict: Dictionary with fixed-size KV cache buffers
            layer_idx: Which layer this is (for indexing into cache)
            position_offset: Position offset for RoPE
            is_prefill: Whether this is prefill phase

        Returns:
            attn_output: [batch, seq_len, hidden_dim]
            cache_dict: Updated cache dictionary
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='q_proj'
        )(hidden_states)
        k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='k_proj'
        )(hidden_states)
        v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='v_proj'
        )(hidden_states)

        # Reshape for attention
        q = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        all_positions = jnp.arange(self.max_position_embeddings)
        position_ids = lax.dynamic_slice(all_positions, (position_offset,), (seq_len,))
        q, k = apply_rotary_pos_emb(q, k, self.rope_cos, self.rope_sin, position_ids)

        # Handle cache
        if cache_dict is not None:
            if is_prefill:
                # Prefill: write entire K,V to prefill cache
                # Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
                k_for_cache = k.transpose(0, 2, 1, 3)
                v_for_cache = v.transpose(0, 2, 1, 3)
                cache_dict = write_prefill_cache(cache_dict, layer_idx, k_for_cache, v_for_cache)

                # Get K,V for attention (just use current k,v for prefill)
                k_full = k
                v_full = v
            else:
                # Decode: update AR cache with single new token
                # k,v shape: [batch, heads, 1, dim]
                k_for_cache = k[:, :, 0, :]  # [batch, heads, dim]
                v_for_cache = v[:, :, 0, :]  # [batch, heads, dim]
                cache_dict = update_kv_cache_ar(cache_dict, layer_idx, k_for_cache, v_for_cache)

                # Get full K,V from cache (returns full buffers, we'll mask appropriately)
                k_cached, v_cached = get_attention_kv(cache_dict, layer_idx, use_prefill=True)
                # Transpose: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
                k_full = k_cached.transpose(0, 2, 1, 3)
                v_full = v_cached.transpose(0, 2, 1, 3)

                # Create attention mask for cache
                # Valid positions: [0:prefill_length) and [max_prefill:max_prefill+ar_index)
                prefill_length = cache_dict['prefill']['length']
                ar_index = cache_dict['ar']['index']
                max_prefill = cache_dict['prefill']['k'].shape[2]  # max_prefill_length
                total_cache_len = k_full.shape[2]  # max_prefill + max_decode

                # Create mask: -inf for invalid positions, 0 for valid positions
                # Shape: [1, 1, 1, total_cache_len] for broadcasting
                pos_indices = jnp.arange(total_cache_len)
                valid_prefill = pos_indices < prefill_length
                valid_ar = (pos_indices >= max_prefill) & (pos_indices < max_prefill + ar_index)
                cache_mask = jnp.where(valid_prefill | valid_ar, 0.0, -1e9)
                cache_mask = cache_mask.reshape(1, 1, 1, total_cache_len)

                # Combine with existing attention mask if present
                if attention_mask is not None:
                    attention_mask = attention_mask + cache_mask
                else:
                    attention_mask = cache_mask
        else:
            # No cache (shouldn't happen with fixed cache mode)
            k_full = k
            v_full = v

        # Repeat k/v heads if using GQA
        if self.num_key_value_groups > 1:
            k_full = jnp.repeat(k_full, self.num_key_value_groups, axis=1)
            v_full = jnp.repeat(v_full, self.num_key_value_groups, axis=1)

        # Compute attention
        attn_weights = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.softmax(attn_weights, axis=-1).astype(self.config.dtype)
        attn_output = jnp.matmul(attn_weights, v_full)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='o_proj'
        )(attn_output)

        return attn_output, cache_dict


class QwenDecoderLayer(nn.Module):
    """Qwen decoder layer."""
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None, position_offset=0):
        residual = hidden_states

        # Self Attention
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps, self.config.dtype, name='input_layernorm')(hidden_states)
        hidden_states, new_kv_cache = QwenAttention(self.config, name='self_attn')(
            hidden_states, attention_mask, kv_cache, position_offset
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps, self.config.dtype, name='post_attention_layernorm')(hidden_states)
        hidden_states = QwenMLP(self.config, name='mlp')(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class QwenModel(nn.Module):
    """Qwen model implementation in JAX."""
    config: QwenConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, kv_caches=None, position_offset=0):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.config.dtype,
            name='embed_tokens'
        )
        hidden_states = embed_tokens(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            # For generation with KV cache: when we have cached states, create appropriate mask
            if kv_caches is not None:
                # With cache: we're decoding, no masking needed (attend to all previous tokens)
                # We'll handle this in attention by not applying any mask
                attention_mask = None  # Will be handled in attention layer
            else:
                # No cache: prefill phase, use causal mask
                attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                attention_mask = jnp.where(attention_mask == 0, -1e9, 0.0)
                attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, 0), 0)

        # Apply decoder layers
        new_kv_caches = []
        for i in range(self.config.num_hidden_layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv_cache = QwenDecoderLayer(self.config, name=f'layers_{i}')(
                hidden_states, attention_mask, layer_kv_cache, position_offset
            )
            new_kv_caches.append(new_kv_cache)

        # Final norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='norm'
        )(hidden_states)

        # LM head
        if self.config.tie_word_embeddings:
            lm_logits = embed_tokens.attend(hidden_states)
        else:
            lm_head = nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                dtype=self.config.dtype,
                name='lm_head'
            )
            lm_logits = lm_head(hidden_states)

        return lm_logits, new_kv_caches


def convert_hf_to_jax_weights(hf_model, config):
    """Convert HuggingFace weights to JAX format.
    
    Raises:
        ValueError: If any expected weight is missing from the HF model state dict.
    """
    jax_params = {}
    state_dict = hf_model.state_dict()
    
    # Convert embeddings
    if 'model.embed_tokens.weight' not in state_dict:
        raise ValueError(
            "Missing required weight: 'model.embed_tokens.weight'. "
            "The HuggingFace model state dict may be corrupted or incompatible."
        )
    jax_params['embed_tokens'] = {'embedding': state_dict['model.embed_tokens.weight'].numpy()}
    
    # Convert each layer
    for layer_idx in range(config.num_hidden_layers):
        layer_params = {}
        hf_prefix = f'model.layers.{layer_idx}'
        jax_prefix = f'layers_{layer_idx}'
        
        # Attention weights — required for every layer
        attn_keys = [
            f'{hf_prefix}.self_attn.q_proj.weight',
            f'{hf_prefix}.self_attn.q_proj.bias',
            f'{hf_prefix}.self_attn.k_proj.weight',
            f'{hf_prefix}.self_attn.k_proj.bias',
            f'{hf_prefix}.self_attn.v_proj.weight',
            f'{hf_prefix}.self_attn.v_proj.bias',
            f'{hf_prefix}.self_attn.o_proj.weight',
        ]
        missing_attn = [k for k in attn_keys if k not in state_dict]
        if missing_attn:
            raise ValueError(
                f"Missing attention weights in layer {layer_idx}: {missing_attn}. "
                f"The model checkpoint may be corrupted or a different architecture."
            )
        
        layer_params['self_attn'] = {
            'q_proj': {
                'kernel': state_dict[f'{hf_prefix}.self_attn.q_proj.weight'].T.numpy(),
                'bias': state_dict[f'{hf_prefix}.self_attn.q_proj.bias'].numpy()
            },
            'k_proj': {
                'kernel': state_dict[f'{hf_prefix}.self_attn.k_proj.weight'].T.numpy(),
                'bias': state_dict[f'{hf_prefix}.self_attn.k_proj.bias'].numpy()
            },
            'v_proj': {
                'kernel': state_dict[f'{hf_prefix}.self_attn.v_proj.weight'].T.numpy(),
                'bias': state_dict[f'{hf_prefix}.self_attn.v_proj.bias'].numpy()
            },
            'o_proj': {
                'kernel': state_dict[f'{hf_prefix}.self_attn.o_proj.weight'].T.numpy()
            }
        }
        
        # MLP weights — required for every layer
        mlp_keys = [
            f'{hf_prefix}.mlp.gate_proj.weight',
            f'{hf_prefix}.mlp.up_proj.weight',
            f'{hf_prefix}.mlp.down_proj.weight',
        ]
        missing_mlp = [k for k in mlp_keys if k not in state_dict]
        if missing_mlp:
            raise ValueError(
                f"Missing MLP weights in layer {layer_idx}: {missing_mlp}. "
                f"The model checkpoint may be corrupted or a different architecture."
            )
        
        layer_params['mlp'] = {
            'gate_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.gate_proj.weight'].T.numpy()},
            'up_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.up_proj.weight'].T.numpy()},
            'down_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.down_proj.weight'].T.numpy()}
        }
        
        # LayerNorm weights — required for every layer
        for norm_name in ['input_layernorm', 'post_attention_layernorm']:
            norm_key = f'{hf_prefix}.{norm_name}.weight'
            if norm_key not in state_dict:
                raise ValueError(
                    f"Missing LayerNorm weight in layer {layer_idx}: '{norm_key}'. "
                    f"The model checkpoint may be corrupted or a different architecture."
                )
            layer_params[norm_name] = {
                'weight': state_dict[norm_key].numpy()
            }
        
        jax_params[jax_prefix] = layer_params
    
    # Final norm — required
    if 'model.norm.weight' not in state_dict:
        raise ValueError(
            "Missing required weight: 'model.norm.weight'. "
            "The HuggingFace model state dict may be corrupted or incompatible."
        )
    jax_params['norm'] = {'weight': state_dict['model.norm.weight'].numpy()}
    
    # LM head (if not tied)
    if not config.tie_word_embeddings:
        if 'lm_head.weight' not in state_dict:
            raise ValueError(
                "Missing required weight: 'lm_head.weight'. "
                "tie_word_embeddings is False but no separate lm_head found."
            )
        jax_params['lm_head'] = {'kernel': state_dict['lm_head.weight'].T.numpy()}
    
    return jax_params


def main():
    """Main function to load and run Qwen model on TPUs."""
    
    # Initialize TPU
    print("Initializing TPU...")
    try:
        jax.devices("tpu")
        print(f"Found {len(jax.devices('tpu'))} TPU cores")
    except:
        print("No TPU found, using default backend:", jax.default_backend())
        print(f"Available devices: {jax.devices()}")
    
    # Load model and tokenizer from HuggingFace
    print("\nLoading Qwen2.5-7B model from HuggingFace...")
    # model_name = "KathirKs/qwen-2.5-0.5b"
    model_name = "Qwen/Qwen2.5-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Get model configuration
    hf_config = hf_model.config
    config = QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=getattr(hf_config, 'rope_theta', 1000000.0),
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings', False)
    )
    
    print("\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    
    # Initialize JAX model
    print("\nInitializing JAX model...")
    jax_model = QwenModel(config)
    
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    
    # Initialize parameters
    params = jax_model.init(rng, dummy_input)
    
    # Convert weights
    print("\nConverting HuggingFace weights to JAX format...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    
    # Update params with converted weights
    params = {'params': converted_params}
    
    # Prepare input
    text = "The future of artificial intelligence"
    print(f"\nInput text: '{text}'")
    
    inputs = tokenizer(text, return_tensors="np")
    input_ids = jnp.array(inputs['input_ids'])
    
    # Run inference
    print("\nRunning inference...")

    # JIT compile functions for better performance on TPU
    # Prefill: process initial prompt
    @jax.jit
    def prefill(params, input_ids):
        return jax_model.apply(params, input_ids, kv_caches=None, position_offset=0)

    # Decode: generate one token at a time with KV cache
    @jax.jit
    def decode_step(params, input_id, kv_caches, position):
        return jax_model.apply(params, input_id, kv_caches=kv_caches, position_offset=position)

    # Generate tokens with KV caching
    max_new_tokens = 20

    print("\nGenerating text (with KV cache optimization)...")
    start_time = time.time()

    # Prefill phase: process the entire prompt
    logits, kv_caches = prefill(params, input_ids)
    next_token_logits = logits[:, -1, :]
    next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
    generated_ids = jnp.concatenate([input_ids, next_token_id], axis=1)

    prefill_time = time.time() - start_time
    decode_start = time.time()

    # Decode phase: generate tokens one at a time
    tokens_generated = 1
    for i in range(1, max_new_tokens):
        # Only process the new token
        logits, kv_caches = decode_step(params, next_token_id, kv_caches, input_ids.shape[1] + i - 1)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        tokens_generated += 1

        # Check for EOS token
        if tokenizer.eos_token_id and next_token_id[0, 0] == tokenizer.eos_token_id:
            break

    decode_time = time.time() - decode_start
    total_time = time.time() - start_time

    # Decode output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Prefill time: {prefill_time:.3f}s ({input_ids.shape[1]} tokens)")
    print(f"  Decode time: {decode_time:.3f}s ({tokens_generated} tokens)")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Tokens/sec (decode): {tokens_generated / decode_time:.2f}")
    
    
    # Performance metrics
    print("\n" + "="*50)
    print("Model successfully loaded and running on JAX!")
    print(f"Device: {jax.devices()[0]}")
    print(f"Total parameters: ~{sum(p.size for p in jax.tree_util.tree_leaves(params)) / 1e6:.1f}M")
    print("="*50)


if __name__ == "__main__":
    main()