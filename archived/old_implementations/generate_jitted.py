"""
Fully JIT-Compiled Generation with Fixed KV Cache

This module implements generation using lax.fori_loop with pre-allocated
buffers for maximum performance. All operations are JIT-compiled with
zero Python overhead during generation.

Key Features:
- Fixed-size KV cache and activation buffers
- Fully JIT-compiled decode loop
- Separate prefill and decode phases
- Optional activation extraction
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple, Optional, Any
from functools import partial
import time

from kvcache_utils import (
    create_kv_cache_buffers,
    create_activation_buffer,
    update_activation_buffer,
    KVCacheConfig
)


def prefill_with_fixed_cache(
    model,
    params: Dict,
    input_ids: jnp.ndarray,
    cache_dict: Dict[str, Any]
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Prefill phase: Process input prompt and populate prefill cache

    Args:
        model: Model instance (with use_fixed_cache=True)
        params: Model parameters
        input_ids: Input tokens [batch, seq_len]
        cache_dict: Pre-allocated cache buffers

    Returns:
        logits: [batch, seq_len, vocab_size]
        cache_dict: Updated cache with prefill written
    """
    # Forward pass with is_prefill=True
    # The model will write to prefill cache
    logits, cache_dict = model.apply(
        params,
        input_ids,
        cache_dict=cache_dict,
        is_prefill=True
    )

    return logits, cache_dict


def decode_step_fixed_cache(
    model,
    params: Dict,
    input_id: jnp.ndarray,
    cache_dict: Dict[str, Any],
    position: int
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Decode step: Generate one token using KV cache

    Args:
        model: Model instance
        params: Model parameters
        input_id: Single token [batch, 1]
        cache_dict: Cache dictionary
        position: Current position (for RoPE)

    Returns:
        logits: [batch, 1, vocab_size]
        cache_dict: Updated cache
    """
    logits, cache_dict = model.apply(
        params,
        input_id,
        cache_dict=cache_dict,
        position_offset=position,
        is_prefill=False
    )

    return logits, cache_dict


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def generate_with_fixed_cache_jitted(
    model,
    params: Dict,
    input_ids: jnp.ndarray,
    cache_dict: Dict[str, Any],
    max_tokens: int,
    extract_activations: bool = False,
    activation_buffer: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict[str, Any]]:
    """
    Fully JIT-compiled generation with fixed KV cache

    This function uses lax.fori_loop for maximum performance.
    All shapes are fixed, making it fully JIT-compatible.

    Args:
        model: Model instance (static arg)
        params: Model parameters
        input_ids: Input tokens [batch, seq_len]
        cache_dict: Pre-allocated KV cache
        max_tokens: Maximum tokens to generate (static arg)
        extract_activations: Whether to extract activations (static arg)
        activation_buffer: Pre-allocated activation buffer (static arg)

    Returns:
        generated_ids: [batch, seq_len + max_tokens]
        activations: [num_layers, max_tokens, batch, hidden_dim] or None
        final_cache: Updated cache dictionary
    """
    batch_size, input_length = input_ids.shape

    # ===================================================================
    # PREFILL PHASE: Process entire prompt
    # ===================================================================

    logits, cache_dict = prefill_with_fixed_cache(
        model, params, input_ids, cache_dict
    )

    # Sample first generated token
    next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
    next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)  # [batch, 1]

    # Initialize generated sequence
    generated_ids = jnp.concatenate([input_ids, next_token], axis=1)

    # ===================================================================
    # DECODE LOOP: Generate tokens one by one
    # ===================================================================

    def decode_step(step, carry):
        """
        Single decode step - called by lax.fori_loop

        Args:
            step: Current step index
            carry: (cache_dict, act_buffer, gen_ids)

        Returns:
            Updated carry tuple
        """
        cache_dict, act_buffer, gen_ids = carry

        # Get last token
        last_token = gen_ids[:, -1:]  # [batch, 1]

        # Position for RoPE
        position = input_length + step

        # Forward pass
        if extract_activations:
            # Extract activations (slower path)
            logits, new_cache, acts = model.apply(
                params,
                last_token,
                cache_dict=cache_dict,
                position_offset=position,
                is_prefill=False,
                return_activations=True
            )

            # Update activation buffer
            # acts shape: [num_layers, batch, hidden_dim]
            new_act_buffer = update_activation_buffer(act_buffer, step, acts)
        else:
            # Fast path without activations
            logits, new_cache = decode_step_fixed_cache(
                model, params, last_token, cache_dict, position
            )
            new_act_buffer = act_buffer

        # Sample next token (greedy for now)
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)

        # Append to generated sequence
        new_gen_ids = jnp.concatenate([gen_ids, next_token], axis=1)

        return (new_cache, new_act_buffer, new_gen_ids)

    # Initial carry
    initial_carry = (cache_dict, activation_buffer, generated_ids)

    # Run decode loop (FULLY JIT-COMPILED!)
    final_cache, final_acts, final_ids = lax.fori_loop(
        0,  # Start
        max_tokens - 1,  # End (we already generated 1 token)
        decode_step,
        initial_carry
    )

    return final_ids, final_acts, final_cache


def generate_single_task(
    model,
    params: Dict,
    input_ids: jnp.ndarray,
    kv_config: KVCacheConfig,
    max_tokens: int = 100,
    extract_activations: bool = False,
    hidden_dim: int = 896,
    num_extract_layers: int = 14
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict[str, Any]]:
    """
    High-level interface for single task generation

    This function handles buffer creation and calls the JIT-compiled
    generation function.

    Args:
        model: Model instance
        params: Model parameters
        input_ids: Input tokens [batch, seq_len]
        kv_config: KV cache configuration
        max_tokens: Maximum tokens to generate
        extract_activations: Whether to extract activations
        hidden_dim: Hidden dimension for activation buffer
        num_extract_layers: Number of layers to extract

    Returns:
        generated_ids: [batch, seq_len + max_tokens]
        activations: [num_layers, max_tokens, batch, hidden_dim] or None
        timing_info: Dict with timing information
    """
    batch_size = input_ids.shape[0]

    # Create buffers
    start_time = time.time()

    cache_dict = create_kv_cache_buffers(kv_config, batch_size)

    if extract_activations:
        act_buffer = create_activation_buffer(
            num_extract_layers, max_tokens, batch_size, hidden_dim
        )
    else:
        act_buffer = None

    buffer_creation_time = time.time() - start_time

    # Generate
    start_time = time.time()

    generated_ids, activations, final_cache = generate_with_fixed_cache_jitted(
        model, params, input_ids, cache_dict, max_tokens,
        extract_activations, act_buffer
    )

    # Wait for completion
    generated_ids.block_until_ready()

    generation_time = time.time() - start_time

    timing_info = {
        'buffer_creation_time': buffer_creation_time,
        'generation_time': generation_time,
        'tokens_generated': max_tokens,
        'tokens_per_sec': max_tokens / generation_time if generation_time > 0 else 0
    }

    return generated_ids, activations, timing_info


def generate_batch_parallel(
    model,
    params: Dict,
    input_ids_batch: jnp.ndarray,
    kv_config: KVCacheConfig,
    max_tokens: int = 100,
    extract_activations: bool = False,
    hidden_dim: int = 896,
    num_extract_layers: int = 14
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict[str, Any]]:
    """
    Generate for multiple tasks in parallel using vmap

    Args:
        model: Model instance
        params: Model parameters
        input_ids_batch: Input tokens [num_tasks, seq_len]
        kv_config: KV cache configuration
        max_tokens: Maximum tokens to generate
        extract_activations: Whether to extract activations
        hidden_dim: Hidden dimension
        num_extract_layers: Number of layers to extract

    Returns:
        generated_ids: [num_tasks, seq_len + max_tokens]
        activations: [num_tasks, num_layers, max_tokens, hidden_dim] or None
        timing_info: Dict with timing information
    """
    num_tasks = input_ids_batch.shape[0]

    # Create buffers for all tasks
    cache_dicts = []
    act_buffers = []

    for _ in range(num_tasks):
        cache_dict = create_kv_cache_buffers(kv_config, batch_size=1)
        cache_dicts.append(cache_dict)

        if extract_activations:
            act_buffer = create_activation_buffer(
                num_extract_layers, max_tokens, 1, hidden_dim
            )
            act_buffers.append(act_buffer)

    # Use vmap to parallelize across tasks
    # TODO: Implement vmap version for multi-task parallel generation
    # For now, loop over tasks (can be optimized later)

    all_generated = []
    all_activations = []

    start_time = time.time()

    for i in range(num_tasks):
        input_ids = input_ids_batch[i:i+1]
        cache_dict = cache_dicts[i]
        act_buffer = act_buffers[i] if extract_activations else None

        gen_ids, acts, _ = generate_with_fixed_cache_jitted(
            model, params, input_ids, cache_dict, max_tokens,
            extract_activations, act_buffer
        )

        all_generated.append(gen_ids)
        if extract_activations:
            all_activations.append(acts)

    # Stack results
    generated_ids = jnp.concatenate(all_generated, axis=0)

    if extract_activations:
        activations = jnp.stack(all_activations, axis=0)
    else:
        activations = None

    generation_time = time.time() - start_time

    timing_info = {
        'num_tasks': num_tasks,
        'generation_time': generation_time,
        'tokens_per_task': max_tokens,
        'total_tokens': num_tasks * max_tokens,
        'tokens_per_sec': (num_tasks * max_tokens) / generation_time
    }

    return generated_ids, activations, timing_info


if __name__ == "__main__":
    print("Testing JIT-compiled generation...")

    # This is a placeholder test - needs actual model
    # Real tests will be in test_generation_jitted.py

    print("✓ Module loaded successfully")
    print("  - generate_with_fixed_cache_jitted: Fully JIT-compiled")
    print("  - generate_single_task: High-level interface")
    print("  - generate_batch_parallel: Multi-task support")
    print("\nRun test_generation_jitted.py for full tests with real model")
