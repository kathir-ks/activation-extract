"""
KV Cache Utilities for Fixed-Size Buffers

Inspired by MaxText's approach to pre-allocated caches for JIT-friendly
generation with static shapes.

Key Features:
- Fixed-size buffers (no concatenation growth)
- JIT-compatible with static shapes
- Separate prefill and autoregressive caches
- Dynamic update at specific positions
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """
    Configuration for KV cache buffers.
    
    Attributes:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (GQA)
        head_dim: Dimension per head
        max_prefill_length: Maximum prompt length
        max_decode_length: Maximum generation length
        dtype: Data type for buffers
    """
    num_layers: int = 24
    num_kv_heads: int = 2
    head_dim: int = 64
    max_prefill_length: int = 512
    max_decode_length: int = 1200
    dtype: Any = jnp.float32


def create_kv_cache_buffers(
    config: KVCacheConfig,
    batch_size: int
) -> Dict[str, Any]:
    """
    Create pre-allocated KV cache buffers (MaxText style).
    
    Returns two separate caches:
    - prefill: Write-once during prompt processing
    - ar (autoregressive): Updated incrementally during generation
    
    Args:
        config: KV cache configuration
        batch_size: Batch size for generation
        
    Returns:
        Dictionary with structure:
        {
            'prefill': {
                'k': [layers, batch, max_prefill_len, heads, dim],
                'v': [layers, batch, max_prefill_len, heads, dim],
                'length': 0
            },
            'ar': {
                'k': [layers, batch, max_decode_len, heads, dim],
                'v': [layers, batch, max_decode_len, heads, dim],
                'index': 0
            }
        }
    """
    prefill_shape = (
        config.num_layers,
        batch_size,
        config.max_prefill_length,
        config.num_kv_heads,
        config.head_dim
    )

    ar_shape = (
        config.num_layers,
        batch_size,
        config.max_decode_length,
        config.num_kv_heads,
        config.head_dim
    )

    return {
        'prefill': {
            'k': jnp.zeros(prefill_shape, dtype=config.dtype),
            'v': jnp.zeros(prefill_shape, dtype=config.dtype),
            'length': 0
        },
        'ar': {
            'k': jnp.zeros(ar_shape, dtype=config.dtype),
            'v': jnp.zeros(ar_shape, dtype=config.dtype),
            'index': 0
        }
    }


def write_prefill_cache(
    cache_dict: Dict[str, Any],
    layer_idx: int,
    k_prefill: jnp.ndarray,
    v_prefill: jnp.ndarray
) -> Dict[str, Any]:
    """
    Write prefill keys/values to cache (write-once).
    
    Args:
        cache_dict: Cache dictionary
        layer_idx: Which layer to update
        k_prefill: Keys from prefill [batch, seq_len, num_kv_heads, head_dim]
        v_prefill: Values from prefill [batch, seq_len, num_kv_heads, head_dim]
        
    Returns:
        Updated cache dictionary
    """
    batch, seq_len, num_heads, head_dim = k_prefill.shape

    new_k_cache = jax.lax.dynamic_update_slice(
        cache_dict['prefill']['k'][layer_idx],
        k_prefill,
        (0, 0, 0, 0)
    )

    new_v_cache = jax.lax.dynamic_update_slice(
        cache_dict['prefill']['v'][layer_idx],
        v_prefill,
        (0, 0, 0, 0)
    )

    new_prefill_k = cache_dict['prefill']['k'].at[layer_idx].set(new_k_cache)
    new_prefill_v = cache_dict['prefill']['v'].at[layer_idx].set(new_v_cache)

    return {
        **cache_dict,
        'prefill': {
            'k': new_prefill_k,
            'v': new_prefill_v,
            'length': jnp.array(seq_len, dtype=jnp.int32)
        }
    }


def update_kv_cache_ar(
    cache_dict: Dict[str, Any],
    layer_idx: int,
    new_k: jnp.ndarray,
    new_v: jnp.ndarray
) -> Dict[str, Any]:
    """
    Update autoregressive cache at current index (MaxText style).
    
    Uses dynamic_update_slice_in_dim to write a single token's K/V.
    
    Args:
        cache_dict: Cache dictionary
        layer_idx: Which layer to update
        new_k: New key [batch, num_kv_heads, head_dim]
        new_v: New value [batch, num_kv_heads, head_dim]
        
    Returns:
        Updated cache with incremented index
    """
    ar_index = cache_dict['ar']['index']

    new_k_with_seq = jnp.expand_dims(new_k, axis=1)
    new_v_with_seq = jnp.expand_dims(new_v, axis=1)

    new_k_cache = jax.lax.dynamic_update_slice_in_dim(
        cache_dict['ar']['k'][layer_idx],
        new_k_with_seq,
        ar_index,
        axis=1
    )

    new_v_cache = jax.lax.dynamic_update_slice_in_dim(
        cache_dict['ar']['v'][layer_idx],
        new_v_with_seq,
        ar_index,
        axis=1
    )

    new_ar_k = cache_dict['ar']['k'].at[layer_idx].set(new_k_cache)
    new_ar_v = cache_dict['ar']['v'].at[layer_idx].set(new_v_cache)

    return {
        **cache_dict,
        'ar': {
            'k': new_ar_k,
            'v': new_ar_v,
            'index': ar_index + 1
        }
    }


def get_attention_kv(
    cache_dict: Dict[str, Any],
    layer_idx: int,
    use_prefill: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get K, V tensors for attention computation.
    
    During decode, returns full cache buffers - attention masking handles the rest.
    This avoids dynamic slicing issues with traced indices in JIT.
    
    Args:
        cache_dict: Cache dictionary
        layer_idx: Which layer
        use_prefill: Whether to include prefill cache
        
    Returns:
        k: [batch, total_seq_len, num_kv_heads, head_dim]
        v: [batch, total_seq_len, num_kv_heads, head_dim]
    """
    if use_prefill:
        k_prefill_full = cache_dict['prefill']['k'][layer_idx]
        v_prefill_full = cache_dict['prefill']['v'][layer_idx]
        k_ar_full = cache_dict['ar']['k'][layer_idx]
        v_ar_full = cache_dict['ar']['v'][layer_idx]

        k = jnp.concatenate([k_prefill_full, k_ar_full], axis=1)
        v = jnp.concatenate([v_prefill_full, v_ar_full], axis=1)
    else:
        k = cache_dict['ar']['k'][layer_idx]
        v = cache_dict['ar']['v'][layer_idx]

    return k, v


def create_activation_buffer(
    num_extract_layers: int,
    max_tokens: int,
    batch_size: int,
    hidden_dim: int,
    dtype: Any = jnp.float32
) -> jnp.ndarray:
    """
    Create pre-allocated activation buffer.
    
    Args:
        num_extract_layers: Number of layers to extract
        max_tokens: Maximum tokens to generate
        batch_size: Batch size
        hidden_dim: Hidden dimension size
        dtype: Data type for buffer
        
    Returns:
        Buffer of shape [num_extract_layers, max_tokens, batch_size, hidden_dim]
    """
    return jnp.zeros(
        (num_extract_layers, max_tokens, batch_size, hidden_dim),
        dtype=dtype
    )


def update_activation_buffer(
    buffer: jnp.ndarray,
    step: int,
    layer_activations: jnp.ndarray
) -> jnp.ndarray:
    """
    Write activations at current step position.
    
    Args:
        buffer: Activation buffer [num_layers, max_tokens, batch, hidden]
        step: Current generation step
        layer_activations: Activations to write [num_layers, batch, hidden]
        
    Returns:
        Updated buffer
    """
    return buffer.at[:, step, :, :].set(layer_activations)


def update_activation_buffer_dynamic(
    buffer: jnp.ndarray,
    step: int,
    layer_activations: jnp.ndarray
) -> jnp.ndarray:
    """
    Write activations using dynamic_update_slice_in_dim.
    
    Same as update_activation_buffer but uses JAX's dynamic update primitive.
    """
    acts_with_step = jnp.expand_dims(layer_activations, axis=1)

    return jax.lax.dynamic_update_slice_in_dim(
        buffer,
        acts_with_step,
        step,
        axis=1
    )


def get_cache_info(cache_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about current cache state.
    
    Useful for debugging and validation.
    """
    return {
        'prefill_length': cache_dict['prefill']['length'],
        'ar_index': cache_dict['ar']['index'],
        'prefill_k_shape': cache_dict['prefill']['k'].shape,
        'ar_k_shape': cache_dict['ar']['k'].shape,
        'total_cached_tokens': cache_dict['prefill']['length'] + cache_dict['ar']['index']
    }


def validate_cache_shapes(
    cache_dict: Dict[str, Any], 
    config: KVCacheConfig, 
    batch_size: int
) -> None:
    """
    Validate that cache has expected shapes.
    
    Raises:
        AssertionError: If shapes don't match
    """
    expected_prefill_shape = (
        config.num_layers, batch_size, config.max_prefill_length,
        config.num_kv_heads, config.head_dim
    )
    expected_ar_shape = (
        config.num_layers, batch_size, config.max_decode_length,
        config.num_kv_heads, config.head_dim
    )

    assert cache_dict['prefill']['k'].shape == expected_prefill_shape, \
        f"Prefill K shape mismatch: {cache_dict['prefill']['k'].shape} vs {expected_prefill_shape}"
    assert cache_dict['ar']['k'].shape == expected_ar_shape, \
        f"AR K shape mismatch: {cache_dict['ar']['k'].shape} vs {expected_ar_shape}"
