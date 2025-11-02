"""
Data Parallel Generation using JAX pmap

Distributes generation across multiple TPU/GPU devices using pmap.
Each device processes different prompts in parallel.
"""

import jax
import jax.numpy as jnp
from jax import pmap
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import partial

from qwen2_jax_fixed import QwenModelFixed
from kvcache_utils import create_kv_cache_buffers, KVCacheConfig


def generate_simple(model, params, input_ids, max_tokens=100, temperature=0.0):
    """
    Simple generation without timing or EOS checks (for use inside pmap)

    NO JIT inside this function - pmap already handles compilation

    Args:
        model: Model instance
        params: Model parameters
        input_ids: Input [batch, seq_len]
        max_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        generated_ids: [batch, seq_len + max_tokens]
    """
    batch_size, input_len = input_ids.shape

    # Create KV cache
    config = model.config
    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=input_len + 10,
        max_decode_length=max_tokens
    )
    cache_dict = create_kv_cache_buffers(kv_config, batch_size)

    # Prefill - NO JIT, pmap handles it
    logits, cache_dict = model.apply(params, input_ids, cache_dict=cache_dict, position_offset=0, is_prefill=True)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    generated_ids = jnp.concatenate([input_ids, next_token], axis=1)

    # Decode loop
    for i in range(1, max_tokens):
        last_token = generated_ids[:, -1:]
        position = input_len + i - 1
        logits, cache_dict = model.apply(params, last_token, cache_dict=cache_dict, position_offset=position, is_prefill=False)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

    return generated_ids


def replicate_params(params: Dict, num_devices: int) -> Dict:
    """
    Replicate model parameters across devices
    
    Args:
        params: Model parameters
        num_devices: Number of devices
        
    Returns:
        Replicated parameters with leading device dimension
    """
    from jax.tree_util import tree_map
    return tree_map(lambda x: jnp.array([x] * num_devices), params)


def pad_batch_to_devices(input_ids_list: List[jnp.ndarray], num_devices: int) -> Tuple[jnp.ndarray, int]:
    """
    Pad batch to be divisible by number of devices
    
    Args:
        input_ids_list: List of input_ids arrays
        num_devices: Number of devices
        
    Returns:
        Stacked and padded input_ids, original batch size
    """
    original_size = len(input_ids_list)
    
    # Pad to multiple of num_devices
    remainder = original_size % num_devices
    if remainder != 0:
        padding_needed = num_devices - remainder
        # Duplicate last element for padding
        for _ in range(padding_needed):
            input_ids_list.append(input_ids_list[-1])
    
    # Find max length for padding
    max_len = max(ids.shape[1] for ids in input_ids_list)
    
    # Pad all sequences to same length
    padded_list = []
    for ids in input_ids_list:
        if ids.shape[1] < max_len:
            pad_width = max_len - ids.shape[1]
            ids = jnp.pad(ids, ((0, 0), (0, pad_width)), constant_values=0)
        padded_list.append(ids)
    
    # Stack: [batch, seq_len]
    stacked = jnp.concatenate(padded_list, axis=0)
    
    # Reshape to [num_devices, batch_per_device, seq_len]
    batch_per_device = len(padded_list) // num_devices
    stacked = stacked.reshape(num_devices, batch_per_device, -1)
    
    return stacked, original_size


@partial(pmap, static_broadcasted_argnums=(0, 3, 4))
def generate_one_device(model, params, input_ids, max_tokens, temperature):
    """
    Generate on a single device (called via pmap)

    This function is pmapped, so it runs on each device with different data.

    Args:
        model: Model instance (static, broadcasted)
        params: Model parameters (replicated across devices)
        input_ids: Input tokens [batch_per_device, seq_len]
        max_tokens: Maximum tokens to generate (static)
        temperature: Sampling temperature (static)

    Returns:
        generated_ids: [batch_per_device, seq_len + max_tokens]
    """
    # Generate for each example in this device's batch
    # Note: This is still sequential within device, but parallel across devices
    batch_per_device = input_ids.shape[0]

    results = []
    for i in range(batch_per_device):
        single_input = input_ids[i:i+1]  # [1, seq_len]

        # Use simple generation (no timing/EOS checks for pmap compatibility)
        generated = generate_simple(
            model, params, single_input,
            max_tokens=max_tokens,
            temperature=temperature
        )
        results.append(generated)

    # Stack results: [batch_per_device, seq_len + max_tokens]
    return jnp.concatenate(results, axis=0)


def generate_parallel(
    model: QwenModelFixed,
    params: Dict,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.0,
    verbose: bool = False
) -> List[str]:
    """
    Generate outputs in parallel across all available devices
    
    Args:
        model: Model instance
        params: Model parameters
        tokenizer: Tokenizer
        prompts: List of prompts
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        verbose: Print progress
        
    Returns:
        List of generated texts
    """
    # Get available devices
    devices = jax.devices()
    num_devices = len(devices)
    
    if verbose:
        print(f"Using {num_devices} devices for parallel generation: {[d.device_kind for d in devices]}")
    
    # Tokenize all prompts
    if verbose:
        print(f"Tokenizing {len(prompts)} prompts...")
    
    input_ids_list = []
    input_lengths = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="np", truncation=True, max_length=2048)
        input_ids = jnp.array(inputs['input_ids'])
        input_ids_list.append(input_ids)
        input_lengths.append(input_ids.shape[1])
    
    # Pad batch to be divisible by num_devices
    padded_input_ids, original_size = pad_batch_to_devices(input_ids_list, num_devices)
    
    if verbose:
        print(f"Batch shape after padding: {padded_input_ids.shape}")
        print(f"  Devices: {padded_input_ids.shape[0]}")
        print(f"  Batch per device: {padded_input_ids.shape[1]}")
        print(f"  Sequence length: {padded_input_ids.shape[2]}")
    
    # Replicate parameters across devices
    if verbose:
        print("Replicating parameters across devices...")
    
    replicated_params = replicate_params(params, num_devices)
    
    # Generate in parallel across devices
    if verbose:
        print(f"Generating {max_tokens} tokens per prompt in parallel...")

    # Ensure static arguments are Python scalars (not JAX arrays)
    max_tokens_scalar = int(max_tokens)
    temperature_scalar = float(temperature)

    # Call pmapped function
    generated_ids = generate_one_device(
        model,
        replicated_params,
        padded_input_ids,
        max_tokens_scalar,
        temperature_scalar
    )
    
    # generated_ids shape: [num_devices, batch_per_device, seq_len + max_tokens]
    
    # Reshape back to [total_batch, seq_len + max_tokens]
    generated_ids = generated_ids.reshape(-1, generated_ids.shape[-1])
    
    # Trim to original batch size
    generated_ids = generated_ids[:original_size]
    
    # Decode outputs
    if verbose:
        print("Decoding outputs...")
    
    outputs = []
    for i, gen_ids in enumerate(generated_ids):
        # Extract generated part (after input)
        input_len = input_lengths[i]
        generated_part = gen_ids[input_len:]
        
        # Decode
        text = tokenizer.decode(np.array(generated_part), skip_special_tokens=True)
        outputs.append(text)
    
    if verbose:
        print(f"✓ Generated {len(outputs)} outputs")
    
    return outputs


if __name__ == "__main__":
    print("Data Parallel Generation Module")
    print("=" * 60)
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {len(jax.devices())}")
    print(f"Device types: {[d.device_kind for d in jax.devices()]}")
