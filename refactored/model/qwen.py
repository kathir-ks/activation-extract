"""
Qwen 2.5 Model JAX Implementation for TPUs

This module contains the core model implementation:
- RMSNorm: Root Mean Square Layer Normalization
- QwenMLP: Gated MLP block with SiLU activation
- QwenAttention: Multi-headed attention with Grouped Query Attention (GQA)
- QwenAttentionFixed: Attention with fixed-size KV cache
- QwenDecoderLayer: Transformer decoder layer
- QwenModel: Full language model

Requirements:
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install transformers flax torch
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np
import math

from .config import QwenConfig
from .kv_cache import (
    write_prefill_cache,
    update_kv_cache_ar,
    get_attention_kv,
)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine embeddings [max_seq_len, head_dim]
        sin: Sine embeddings [max_seq_len, head_dim]
        position_ids: Position indices [seq_len]
    
    Returns:
        q_embed, k_embed: Rotated query and key tensors
    """
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
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't subtract the mean.
    
    Attributes:
        dim: Hidden dimension size
        eps: Epsilon for numerical stability
        dtype: Data type for computations
    """
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
    """
    Qwen MLP block with gated activation.
    
    Uses SiLU (Swish) activation with a gate projection.
    
    Attributes:
        config: Model configuration
    """
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
    """
    Multi-headed attention with Grouped Query Attention (GQA) support.
    
    Uses standard KV caching with concatenation for generation.
    
    Attributes:
        config: Model configuration
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
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None, position_offset=0):
        """
        Forward pass for attention layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            kv_cache: Optional (k, v) tuple for cached keys/values
            position_offset: Position offset for RoPE
            
        Returns:
            attn_output: Attention output [batch, seq_len, hidden_dim]
            new_kv_cache: Updated (k, v) cache tuple
        """
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
    Multi-headed attention with fixed-size KV cache (MaxText style).
    
    Uses pre-allocated buffers and dynamic updates instead of concatenation.
    JIT-friendly with static shapes throughout generation.
    
    Attributes:
        config: Model configuration
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
        Forward pass with fixed-size KV cache.
        
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
                k_for_cache = k.transpose(0, 2, 1, 3)
                v_for_cache = v.transpose(0, 2, 1, 3)
                cache_dict = write_prefill_cache(cache_dict, layer_idx, k_for_cache, v_for_cache)
                k_full = k
                v_full = v
            else:
                # Decode: update AR cache with single new token
                k_for_cache = k[:, :, 0, :]
                v_for_cache = v[:, :, 0, :]
                cache_dict = update_kv_cache_ar(cache_dict, layer_idx, k_for_cache, v_for_cache)

                # Get full K,V from cache
                k_cached, v_cached = get_attention_kv(cache_dict, layer_idx, use_prefill=True)
                k_full = k_cached.transpose(0, 2, 1, 3)
                v_full = v_cached.transpose(0, 2, 1, 3)

                # Create attention mask for cache
                prefill_length = cache_dict['prefill']['length']
                ar_index = cache_dict['ar']['index']
                max_prefill = cache_dict['prefill']['k'].shape[2]
                total_cache_len = k_full.shape[2]

                pos_indices = jnp.arange(total_cache_len)
                valid_prefill = pos_indices < prefill_length
                valid_ar = (pos_indices >= max_prefill) & (pos_indices < max_prefill + ar_index)
                cache_mask = jnp.where(valid_prefill | valid_ar, 0.0, -1e9)
                cache_mask = cache_mask.reshape(1, 1, 1, total_cache_len)

                if attention_mask is not None:
                    attention_mask = attention_mask + cache_mask
                else:
                    attention_mask = cache_mask
        else:
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
    """
    Qwen decoder layer.
    
    Consists of:
    1. Pre-norm + Self-attention + Residual
    2. Pre-norm + MLP + Residual
    
    Attributes:
        config: Model configuration
    """
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None, position_offset=0):
        residual = hidden_states

        # Self Attention
        hidden_states = RMSNorm(
            self.config.hidden_size, 
            self.config.rms_norm_eps, 
            self.config.dtype, 
            name='input_layernorm'
        )(hidden_states)
        hidden_states, new_kv_cache = QwenAttention(self.config, name='self_attn')(
            hidden_states, attention_mask, kv_cache, position_offset
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = RMSNorm(
            self.config.hidden_size, 
            self.config.rms_norm_eps, 
            self.config.dtype, 
            name='post_attention_layernorm'
        )(hidden_states)
        hidden_states = QwenMLP(self.config, name='mlp')(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class QwenModel(nn.Module):
    """
    Qwen language model implementation in JAX.
    
    Full decoder-only transformer with:
    - Token embeddings
    - N decoder layers
    - Final RMSNorm
    - Language model head
    
    Attributes:
        config: Model configuration
    """
    config: QwenConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, kv_caches=None, position_offset=0):
        """
        Forward pass for the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            kv_caches: Optional list of (k, v) tuples for each layer
            position_offset: Position offset for generation
            
        Returns:
            lm_logits: Logits over vocabulary [batch, seq_len, vocab_size]
            new_kv_caches: Updated KV caches
        """
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
            if kv_caches is not None:
                # Decode phase: no masking needed
                attention_mask = None
            else:
                # Prefill phase: use causal mask
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


def convert_hf_to_jax_weights(hf_model, config: QwenConfig) -> Dict[str, Any]:
    """
    Convert HuggingFace weights to JAX format.
    
    Args:
        hf_model: HuggingFace model instance
        config: QwenConfig instance
        
    Returns:
        Dictionary of JAX-compatible parameters
    """
    jax_params = {}
    state_dict = hf_model.state_dict()
    
    # Convert embeddings
    if 'model.embed_tokens.weight' in state_dict:
        jax_params['embed_tokens'] = {'embedding': state_dict['model.embed_tokens.weight'].numpy()}
    
    # Convert each layer
    for layer_idx in range(config.num_hidden_layers):
        layer_params = {}
        hf_prefix = f'model.layers.{layer_idx}'
        jax_prefix = f'layers_{layer_idx}'
        
        # Attention weights
        if f'{hf_prefix}.self_attn.q_proj.weight' in state_dict:
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
        
        # MLP weights
        if f'{hf_prefix}.mlp.gate_proj.weight' in state_dict:
            layer_params['mlp'] = {
                'gate_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.gate_proj.weight'].T.numpy()},
                'up_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.up_proj.weight'].T.numpy()},
                'down_proj': {'kernel': state_dict[f'{hf_prefix}.mlp.down_proj.weight'].T.numpy()}
            }
        
        # LayerNorm weights
        if f'{hf_prefix}.input_layernorm.weight' in state_dict:
            layer_params['input_layernorm'] = {
                'weight': state_dict[f'{hf_prefix}.input_layernorm.weight'].numpy()
            }
        if f'{hf_prefix}.post_attention_layernorm.weight' in state_dict:
            layer_params['post_attention_layernorm'] = {
                'weight': state_dict[f'{hf_prefix}.post_attention_layernorm.weight'].numpy()
            }
        
        jax_params[jax_prefix] = layer_params
    
    # Final norm
    if 'model.norm.weight' in state_dict:
        jax_params['norm'] = {'weight': state_dict['model.norm.weight'].numpy()}
    
    # LM head (if not tied)
    if 'lm_head.weight' in state_dict and not config.tie_word_embeddings:
        jax_params['lm_head'] = {'kernel': state_dict['lm_head.weight'].T.numpy()}
    
    return jax_params
