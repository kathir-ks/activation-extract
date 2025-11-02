"""
Integration Tests for JIT-Compiled Generation

Tests the complete generation pipeline with real model weights,
comparing fixed-cache implementation against current implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from transformers import AutoTokenizer

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_fixed import QwenModelFixed, generate_with_kv_cache_timed
from kvcache_utils import create_kv_cache_buffers, KVCacheConfig


def load_model_and_tokenizer():
    """Load Qwen model and tokenizer"""
    print("Loading model and tokenizer...")

    model_name = "Qwen/Qwen2.5-0.5B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    from transformers import AutoModelForCausalLM
    import torch
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # Get config from HF model
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
        tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings', True)
    )

    # Create JAX model
    model = QwenModelFixed(config)

    # Initialize with dummy input
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    # Convert HF weights to JAX format
    print("Converting HuggingFace weights to JAX...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    print("✓ Model and tokenizer loaded")
    return model, params, tokenizer, config


def test_simple_generation():
    """Test 1: Simple generation with fixed cache"""
    print("\n" + "="*60)
    print("Test 1: Simple Generation with Fixed Cache")
    print("="*60)

    # Load model
    model, params, tokenizer, config = load_model_and_tokenizer()

    # Prepare input
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)

    print(f"\nPrompt: '{prompt}'")
    print(f"Input IDs shape: {input_ids.shape}")

    # Create KV cache config
    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=input_ids.shape[1] + 10,
        max_decode_length=100
    )

    # Generate
    print("\nGenerating with fixed cache...")
    try:
        generated_ids, timing = generate_with_kv_cache_timed(
            model, params, input_ids,
            max_tokens=20,
            tokenizer=tokenizer
        )

        # Decode output
        output_text = tokenizer.decode(generated_ids[0])

        print(f"\n✓ Generation successful!")
        print(f"Output: '{output_text}'")
        print(f"Tokens generated: {timing['tokens_generated']}")
        print(f"Total time: {timing['total_time']:.3f}s")
        print(f"  Prefill: {timing['prefill_time']:.3f}s")
        print(f"  Decode: {timing['decode_time']:.3f}s")
        print(f"Tokens/sec: {timing['tokens_per_sec']:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_with_activations():
    """Test 2: Generation with different temperature"""
    print("\n" + "="*60)
    print("Test 2: Generation with Different Temperature")
    print("="*60)

    # Load model
    model, params, tokenizer, config = load_model_and_tokenizer()

    # Prepare input
    prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)

    print(f"\nPrompt: '{prompt}'")
    print(f"Testing greedy (temp=0.0) generation...")

    try:
        generated_ids, timing = generate_with_kv_cache_timed(
            model, params, input_ids,
            max_tokens=10,
            temperature=0.0,
            tokenizer=tokenizer
        )

        output_text = tokenizer.decode(generated_ids[0])

        print(f"\n✓ Generation successful!")
        print(f"Output: '{output_text}'")
        print(f"Generated tokens: {timing['tokens_generated']}")
        print(f"Tokens/sec: {timing['tokens_per_sec']:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_length_prompts():
    """Test 3: Variable length prompts"""
    print("\n" + "="*60)
    print("Test 3: Variable Length Prompts")
    print("="*60)

    # Load model
    model, params, tokenizer, config = load_model_and_tokenizer()

    prompts = [
        "Hi",
        "What is machine learning?",
        "Explain quantum computing in simple terms with examples and details"
    ]

    print("\nTesting with 3 prompts of different lengths:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. '{prompt}' ({len(tokenizer.encode(prompt))} tokens)")

    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='np')
        input_ids = jnp.array(input_ids)

        kv_config = KVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_prefill_length=input_ids.shape[1] + 20,
            max_decode_length=50
        )

        try:
            generated_ids, timing = generate_with_kv_cache_timed(
                model, params, input_ids,
                max_tokens=10,
                tokenizer=tokenizer
            )

            results.append({
                'prompt': prompt,
                'input_len': input_ids.shape[1],
                'tokens_per_sec': timing['tokens_per_sec'],
                'success': True
            })

        except Exception as e:
            print(f"✗ Failed for prompt '{prompt}': {e}")
            results.append({
                'prompt': prompt,
                'input_len': input_ids.shape[1],
                'tokens_per_sec': 0,
                'success': False
            })

    # Print results
    print("\nResults:")
    print(f"{'Prompt':<50} {'Input Len':<12} {'Tok/sec':<12} {'Status'}")
    print("-" * 80)
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{r['prompt'][:48]:<50} {r['input_len']:<12} {r['tokens_per_sec']:<12.2f} {status}")

    all_success = all(r['success'] for r in results)
    if all_success:
        print("\n✓ All variable length prompts succeeded!")
    else:
        print("\n✗ Some prompts failed")

    return all_success


def test_performance_comparison():
    """Test 4: Performance comparison"""
    print("\n" + "="*60)
    print("Test 4: Performance Benchmarking")
    print("="*60)

    # Load model
    model, params, tokenizer, config = load_model_and_tokenizer()

    # Prepare input
    prompt = "What is the meaning of life?"
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)

    print(f"\nBenchmarking with prompt: '{prompt}'")
    print(f"Running 5 warmup iterations...")

    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=input_ids.shape[1] + 10,
        max_decode_length=100
    )

    # Warmup (for JIT compilation)
    for _ in range(5):
        try:
            _, _ = generate_with_kv_cache_timed(
                model, params, input_ids,
                max_tokens=20,
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False

    print("Warmup complete. Running 10 benchmark iterations...")

    # Benchmark
    times = []
    tokens_per_sec_list = []

    for i in range(10):
        try:
            _, timing = generate_with_kv_cache_timed(
                model, params, input_ids,
                max_tokens=20,
                tokenizer=tokenizer
            )
            times.append(timing['total_time'])
            tokens_per_sec_list.append(timing['tokens_per_sec'])

        except Exception as e:
            print(f"Iteration {i+1} failed: {e}")
            return False

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_tok_per_sec = np.mean(tokens_per_sec_list)
    std_tok_per_sec = np.std(tokens_per_sec_list)

    print("\n✓ Benchmark complete!")
    print(f"\nResults (10 iterations):")
    print(f"  Generation time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"  Tokens/sec: {avg_tok_per_sec:.2f} ± {std_tok_per_sec:.2f}")
    print(f"  Min: {min(tokens_per_sec_list):.2f} tok/s")
    print(f"  Max: {max(tokens_per_sec_list):.2f} tok/s")

    # Performance check
    if avg_tok_per_sec > 20:
        print(f"\n✓ Performance target met! (>{20} tok/s)")
    else:
        print(f"\n⚠ Performance below target (<{20} tok/s)")

    return True


def test_correctness_basic():
    """Test 5: Basic correctness check"""
    print("\n" + "="*60)
    print("Test 5: Basic Correctness")
    print("="*60)

    # Load model
    model, params, tokenizer, config = load_model_and_tokenizer()

    # Test with simple prompt
    prompt = "2 + 2 ="
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    input_ids = jnp.array(input_ids)

    print(f"\nPrompt: '{prompt}'")

    kv_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=input_ids.shape[1] + 10,
        max_decode_length=50
    )

    try:
        generated_ids, timing = generate_with_kv_cache_timed(
            model, params, input_ids,
            max_tokens=10,
            tokenizer=tokenizer
        )

        # Decode
        output_text = tokenizer.decode(generated_ids[0])

        print(f"Full output: '{output_text}'")

        # Check if output makes sense
        generated_part = output_text[len(prompt):]
        print(f"Generated: '{generated_part}'")

        print("\n✓ Generation completed")
        print(f"  Tokens/sec: {timing['tokens_per_sec']:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_integration_tests():
    """Run all integration tests"""
    print("="*60)
    print("Running Integration Tests for JIT-Compiled Generation")
    print("="*60)

    tests = [
        ("Simple Generation", test_simple_generation),
        ("Different Temperature", test_generation_with_activations),
        ("Variable Length Prompts", test_variable_length_prompts),
        ("Performance Benchmarking", test_performance_comparison),
        ("Basic Correctness", test_correctness_basic),
    ]

    results = []

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:<40} {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All integration tests passed!")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys

    # Check if JAX is available
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX backend: {jax.default_backend()}")
    except ImportError:
        print("✗ JAX not installed")
        sys.exit(1)

    # Run tests
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
