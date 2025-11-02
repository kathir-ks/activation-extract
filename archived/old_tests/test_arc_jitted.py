"""
Test ARC inference with JIT-compiled generation (generate_jitted.py)
"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_fixed import QwenModelFixed
from generate_jitted import generate_single_task
from kvcache_utils import KVCacheConfig


def test_jitted_generation():
    """Test JIT-compiled generation with generate_jitted.py"""
    print("="*60)
    print("Testing JIT-Compiled Generation for ARC")
    print("="*60)

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\n1. Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    config = QwenConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        intermediate_size=hf_model.config.intermediate_size,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        use_fixed_cache=True
    )

    print("2. Initializing JAX model...")
    model = QwenModelFixed(config)

    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input)

    print("3. Converting weights...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    # Test with JIT-compiled generation
    print("\n" + "="*60)
    print("Test 1: JIT-Compiled Generation (generate_single_task)")
    print("="*60)

    test_prompt = "Question: What is 2+2? Answer:"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = jnp.array(inputs['input_ids'])

    print(f"Prompt: '{test_prompt}'")
    print(f"Input shape: {input_ids.shape}")

    # Create KV cache config
    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=input_ids.shape[1] + 10,
        max_decode_length=20
    )

    start_time = time.time()
    generated_ids, activations, timing = generate_single_task(
        model, params, input_ids,
        kv_config=kv_config,
        max_tokens=20,
        extract_activations=False,
        hidden_dim=config.hidden_size,
        num_extract_layers=0
    )
    total_time = time.time() - start_time

    output_text = tokenizer.decode(generated_ids[0])

    print(f"\n✓ JIT-compiled generation successful!")
    print(f"Output: '{output_text}'")
    print(f"Total time: {total_time:.2f}s")
    print(f"  Buffer creation: {timing['buffer_creation_time']:.3f}s")
    print(f"  Generation: {timing['generation_time']:.3f}s")
    print(f"  Speed: {timing['tokens_per_sec']:.2f} tok/s")

    # Test 2: Multiple calls (JIT warmup)
    print("\n" + "="*60)
    print("Test 2: JIT Warmup and Performance")
    print("="*60)

    print("Running 3 iterations to warm up JIT...")
    times = []
    
    for i in range(3):
        start = time.time()
        generated_ids, _, timing = generate_single_task(
            model, params, input_ids,
            kv_config=kv_config,
            max_tokens=10,
            extract_activations=False,
            hidden_dim=config.hidden_size,
            num_extract_layers=0
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}s ({timing['tokens_per_sec']:.2f} tok/s)")

    print(f"\n✓ JIT warmup complete!")
    print(f"First call (with compilation): {times[0]:.2f}s")
    print(f"Subsequent calls (average): {sum(times[1:])/len(times[1:]):.2f}s")
    print(f"Speedup after JIT: {times[0]/times[-1]:.2f}x")

    print("\n" + "="*60)
    print("✓ All JIT-compiled generation tests passed!")
    print("="*60)

    return True


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    test_jitted_generation()
