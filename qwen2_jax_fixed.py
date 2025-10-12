"""
Qwen 2.5 Model with Fixed KV Cache (MaxText Style)

This module provides a complete model implementation using fixed-size
KV caches for maximum performance. It wraps QwenAttentionFixed and
provides a drop-in replacement for QwenModel.

Usage:
    from qwen2_jax_fixed import QwenModelFixed, generate_with_kv_cache

    # Create model (same as before)
    config = QwenConfig()
    model = QwenModelFixed(config)

    # Load weights (same as before)
    params = load_weights(...)

    # Generate with fixed cache
    generated_ids = generate_with_kv_cache(
        model, params, input_ids, max_tokens=100
    )
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import time

from qwen2_jax import (
    QwenConfig, RMSNorm, QwenMLP, QwenAttentionFixed,
    rotate_half, apply_rotary_pos_emb
)
from kvcache_utils import (
    create_kv_cache_buffers,
    KVCacheConfig
)


class QwenDecoderLayerFixed(nn.Module):
    """Qwen decoder layer with fixed KV cache"""
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, cache_dict=None,
                 layer_idx=0, position_offset=0, is_prefill=False):
        residual = hidden_states

        # Self Attention with pre-norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='input_layernorm'
        )(hidden_states)

        hidden_states, cache_dict = QwenAttentionFixed(
            self.config,
            name='self_attn'
        )(hidden_states, attention_mask, cache_dict, layer_idx, position_offset, is_prefill)

        hidden_states = residual + hidden_states

        # MLP with post-norm
        residual = hidden_states
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='post_attention_layernorm'
        )(hidden_states)

        hidden_states = QwenMLP(self.config, name='mlp')(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_dict


class QwenModelFixed(nn.Module):
    """
    Qwen model with fixed-size KV cache (MaxText style)

    Drop-in replacement for QwenModel with fixed cache support.
    """
    config: QwenConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, cache_dict=None,
                 position_offset=0, is_prefill=False):
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
            if cache_dict is not None and not is_prefill:
                # Decode phase: no masking needed (attend to all cached tokens)
                attention_mask = None
            else:
                # Prefill phase: use causal mask
                attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                attention_mask = jnp.where(attention_mask == 0, -1e9, 0.0)
                attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, 0), 0)

        # Apply decoder layers
        for i in range(self.config.num_hidden_layers):
            hidden_states, cache_dict = QwenDecoderLayerFixed(
                self.config,
                name=f'layers_{i}'
            )(hidden_states, attention_mask, cache_dict, i, position_offset, is_prefill)

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

        return lm_logits, cache_dict


def generate_with_kv_cache(
    model: QwenModelFixed,
    params: Dict,
    input_ids: jnp.ndarray,
    max_tokens: int = 100,
    temperature: float = 0.0,
    tokenizer=None
) -> jnp.ndarray:
    """
    Generate tokens with fixed KV cache

    High-level interface that handles cache creation and generation.
    Compatible with existing code - just replace generate function.

    Args:
        model: QwenModelFixed instance
        params: Model parameters
        input_ids: Input tokens [batch, seq_len]
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        tokenizer: Optional tokenizer for EOS detection

    Returns:
        generated_ids: [batch, seq_len + max_tokens]
    """
    batch_size, input_len = input_ids.shape

    # Create KV cache config
    kv_config = KVCacheConfig(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        max_prefill_length=input_len + 10,
        max_decode_length=max_tokens + 50
    )

    # Create cache buffers
    cache_dict = create_kv_cache_buffers(kv_config, batch_size)

    # JIT compile functions
    @jax.jit
    def prefill_fn(params, input_ids, cache_dict):
        """Prefill: process entire prompt"""
        return model.apply(
            params, input_ids,
            cache_dict=cache_dict,
            position_offset=0,
            is_prefill=True
        )

    @jax.jit
    def decode_fn(params, input_id, cache_dict, position):
        """Decode: process one token"""
        return model.apply(
            params, input_id,
            cache_dict=cache_dict,
            position_offset=position,
            is_prefill=False
        )

    # Prefill phase
    logits, cache_dict = prefill_fn(params, input_ids, cache_dict)

    # Sample first token
    next_token_logits = logits[:, -1, :]
    if temperature == 0.0:
        next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
    else:
        scaled_logits = next_token_logits / temperature
        next_token = jax.random.categorical(
            jax.random.PRNGKey(0), scaled_logits, axis=-1
        ).reshape(-1, 1)

    generated_ids = jnp.concatenate([input_ids, next_token], axis=1)

    # Decode loop
    for i in range(1, max_tokens):
        last_token = generated_ids[:, -1:]
        position = input_len + i - 1

        logits, cache_dict = decode_fn(params, last_token, cache_dict, position)

        next_token_logits = logits[:, -1, :]
        if temperature == 0.0:
            next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            scaled_logits = next_token_logits / temperature
            key = jax.random.PRNGKey(i)
            next_token = jax.random.categorical(
                key, scaled_logits, axis=-1
            ).reshape(-1, 1)

        generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

        # Check for EOS
        if tokenizer and tokenizer.eos_token_id:
            if jnp.any(next_token == tokenizer.eos_token_id):
                break

    return generated_ids


def generate_with_kv_cache_timed(
    model: QwenModelFixed,
    params: Dict,
    input_ids: jnp.ndarray,
    max_tokens: int = 100,
    temperature: float = 0.0,
    tokenizer=None
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    Generate with timing information

    Returns:
        generated_ids: [batch, seq_len + max_tokens]
        timing_info: Dict with prefill_time, decode_time, tokens_per_sec
    """
    batch_size, input_len = input_ids.shape

    # Create KV cache
    kv_config = KVCacheConfig(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        max_prefill_length=input_len + 10,
        max_decode_length=max_tokens + 50
    )
    cache_dict = create_kv_cache_buffers(kv_config, batch_size)

    # JIT compile
    @jax.jit
    def prefill_fn(params, input_ids, cache_dict):
        return model.apply(params, input_ids, cache_dict=cache_dict,
                          position_offset=0, is_prefill=True)

    @jax.jit
    def decode_fn(params, input_id, cache_dict, position):
        return model.apply(params, input_id, cache_dict=cache_dict,
                          position_offset=position, is_prefill=False)

    # Prefill
    start_time = time.time()
    logits, cache_dict = prefill_fn(params, input_ids, cache_dict)
    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
    generated_ids = jnp.concatenate([input_ids, next_token], axis=1)
    generated_ids.block_until_ready()  # Wait for completion
    prefill_time = time.time() - start_time

    # Decode
    decode_start = time.time()
    tokens_generated = 1

    for i in range(1, max_tokens):
        last_token = generated_ids[:, -1:]
        position = input_len + i - 1

        logits, cache_dict = decode_fn(params, last_token, cache_dict, position)
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            scaled_logits = next_token_logits / temperature
            key = jax.random.PRNGKey(i)
            next_token = jax.random.categorical(key, scaled_logits, axis=-1).reshape(-1, 1)

        generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)
        tokens_generated += 1

        # Check for EOS
        if tokenizer and tokenizer.eos_token_id:
            if jnp.any(next_token == tokenizer.eos_token_id):
                break

    generated_ids.block_until_ready()
    decode_time = time.time() - decode_start

    timing_info = {
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': prefill_time + decode_time,
        'tokens_generated': tokens_generated,
        'tokens_per_sec': tokens_generated / decode_time if decode_time > 0 else 0
    }

    return generated_ids, timing_info


if __name__ == "__main__":
    print("Testing QwenModelFixed...")

    # Create config and model
    config = QwenConfig()
    model = QwenModelFixed(config)

    # Initialize
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    print(f"Model created with {config.num_hidden_layers} layers")

    # Test generation
    print("\nTesting generation...")
    input_ids = jnp.ones((1, 5), dtype=jnp.int32)

    generated_ids, timing = generate_with_kv_cache_timed(
        model, params, input_ids, max_tokens=20
    )

    print(f"✓ Generated {generated_ids.shape[1] - input_ids.shape[1]} tokens")
    print(f"  Prefill: {timing['prefill_time']:.3f}s")
    print(f"  Decode: {timing['decode_time']:.3f}s")
    print(f"  Speed: {timing['tokens_per_sec']:.2f} tok/s")

    print("\n✓ QwenModelFixed works!")
