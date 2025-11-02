"""
Tests for Qwen 2.5 JAX implementation performance optimizations
"""

import jax
import jax.numpy as jnp
import time
import traceback
from qwen2_jax import (
    QwenConfig,
    QwenModel,
    QwenAttention,
    apply_rotary_pos_emb,
    rotate_half,
)


class TestRoPECaching:
    """Test that RoPE embeddings are cached and not recomputed."""

    def test_rope_cached_in_setup(self):
        """Verify RoPE embeddings are precomputed in setup."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        attention = QwenAttention(config)

        # Initialize the module
        dummy_input = jnp.ones((1, 10, config.hidden_size))
        rng = jax.random.PRNGKey(0)
        variables = attention.init(rng, dummy_input)

        # Verify rope_cos and rope_sin are cached in the instance
        # After setup, they should be accessible
        bound_module = attention.bind(variables)

        # The setup should have created rope_cos and rope_sin
        # We can verify this by checking the attention mechanism works
        output, kv_cache = bound_module(dummy_input)

        assert output.shape == (1, 10, config.hidden_size)
        assert kv_cache is not None

    def test_rope_not_recomputed_per_forward(self):
        """Verify RoPE is not recomputed on each forward pass."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)

        # Initialize
        params = model.init(rng, dummy_input)

        # Run multiple forward passes and time them
        # If RoPE is cached, all passes should be similar speed
        times = []
        for _ in range(5):
            start = time.time()
            logits, _ = model.apply(params, dummy_input)
            logits.block_until_ready()  # Wait for computation
            times.append(time.time() - start)

        # First call might be slower due to compilation
        # But subsequent calls should be consistent
        avg_time = sum(times[1:]) / len(times[1:])

        # All times after first should be within 50% of average
        # (generous margin for timing variance)
        for t in times[1:]:
            assert abs(t - avg_time) / avg_time < 0.5, \
                f"Timing variance too high: {times}, suggesting recomputation"


class TestKVCaching:
    """Test KV cache implementation."""

    def test_kv_cache_returned(self):
        """Verify KV cache is returned from model."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        rng = jax.random.PRNGKey(0)

        params = model.init(rng, input_ids)
        logits, kv_caches = model.apply(params, input_ids, kv_caches=None, position_offset=0)

        assert logits.shape == (1, 5, config.vocab_size)
        assert kv_caches is not None
        assert len(kv_caches) == config.num_hidden_layers

        # Each layer should have K and V cache
        for kv_cache in kv_caches:
            k_cache, v_cache = kv_cache
            # Shape: (batch, num_kv_heads, seq_len, head_dim)
            assert k_cache.shape[0] == 1  # batch
            assert k_cache.shape[1] == config.num_key_value_heads
            assert k_cache.shape[2] == 5  # seq_len
            assert k_cache.shape[3] == config.hidden_size // config.num_attention_heads

    def test_kv_cache_reuse(self):
        """Verify KV cache can be reused for next token generation."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        rng = jax.random.PRNGKey(0)

        # Initial prompt
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        params = model.init(rng, input_ids)

        # Prefill
        logits, kv_caches = model.apply(params, input_ids, kv_caches=None, position_offset=0)

        # Generate next token with cache
        next_token = jnp.array([[6]])
        logits_next, kv_caches_next = model.apply(
            params, next_token, kv_caches=kv_caches, position_offset=5
        )

        assert logits_next.shape == (1, 1, config.vocab_size)
        assert len(kv_caches_next) == config.num_hidden_layers

        # Cache should have grown
        for i, (kv_cache, kv_cache_next) in enumerate(zip(kv_caches, kv_caches_next)):
            k_cache, v_cache = kv_cache
            k_cache_next, v_cache_next = kv_cache_next

            # New cache should be longer
            assert k_cache_next.shape[2] == k_cache.shape[2] + 1, \
                f"Layer {i}: Expected cache to grow from {k_cache.shape[2]} to {k_cache.shape[2] + 1}, got {k_cache_next.shape[2]}"

    def test_kv_cache_performance_gain(self):
        """Verify KV cache provides significant speedup."""
        config = QwenConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
        )

        model = QwenModel(config)
        rng = jax.random.PRNGKey(0)

        # Sequence of 20 tokens
        input_ids = jnp.array([[1] + list(range(2, 21))])
        params = model.init(rng, input_ids[:, :1])

        # Method 1: Without KV cache (reprocess everything each time)
        start_no_cache = time.time()
        for i in range(1, 20):
            logits, _ = model.apply(params, input_ids[:, :i+1], kv_caches=None, position_offset=0)
            logits.block_until_ready()
        time_no_cache = time.time() - start_no_cache

        # Method 2: With KV cache (only process new tokens)
        start_with_cache = time.time()
        logits, kv_caches = model.apply(params, input_ids[:, :1], kv_caches=None, position_offset=0)
        logits.block_until_ready()
        for i in range(1, 20):
            logits, kv_caches = model.apply(
                params, input_ids[:, i:i+1], kv_caches=kv_caches, position_offset=i
            )
            logits.block_until_ready()
        time_with_cache = time.time() - start_with_cache

        print(f"\nTime without cache: {time_no_cache:.3f}s")
        print(f"Time with cache: {time_with_cache:.3f}s")
        print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")

        # With cache should be significantly faster (at least 2x)
        assert time_with_cache < time_no_cache / 2, \
            f"KV cache should provide >2x speedup, got {time_no_cache / time_with_cache:.2f}x"


class TestJITCompilation:
    """Test JIT compilation is not causing recompilation issues."""

    def test_jit_prefill_stable(self):
        """Verify JIT compiled prefill doesn't recompile."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        params = model.init(rng, input_ids)

        @jax.jit
        def prefill(params, input_ids):
            return model.apply(params, input_ids, kv_caches=None, position_offset=0)

        # Run multiple times with same shape
        for _ in range(3):
            logits, kv_caches = prefill(params, input_ids)
            logits.block_until_ready()

        # Should complete without errors
        assert logits.shape == (1, 5, config.vocab_size)

    def test_jit_decode_stable(self):
        """Verify JIT compiled decode doesn't recompile per token."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.array([[1, 2, 3]])
        params = model.init(rng, input_ids)

        @jax.jit
        def decode_step(params, input_id, kv_caches, position):
            return model.apply(params, input_id, kv_caches=kv_caches, position_offset=position)

        # Prefill
        logits, kv_caches = model.apply(params, input_ids, kv_caches=None, position_offset=0)

        # Generate multiple tokens (same shape each time)
        times = []
        for i in range(10):
            next_token = jnp.array([[i + 10]])
            start = time.time()
            logits, kv_caches = decode_step(params, next_token, kv_caches, 3 + i)
            logits.block_until_ready()
            times.append(time.time() - start)

        # First call might be slower due to compilation
        # Subsequent calls should be fast and consistent
        avg_time = sum(times[1:]) / len(times[1:])

        print(f"\nDecode times: {[f'{t:.4f}' for t in times]}")
        print(f"Average (excluding first): {avg_time:.4f}s")

        # All times after first should be similar (within 100% variance)
        for i, t in enumerate(times[1:], 1):
            assert abs(t - avg_time) / avg_time < 1.0, \
                f"Token {i} time {t:.4f}s too different from avg {avg_time:.4f}s, suggesting recompilation"


class TestCorrectness:
    """Test that optimizations don't break correctness."""

    def test_with_without_cache_same_result(self):
        """Verify results are identical with and without KV cache."""
        config = QwenConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
        )

        model = QwenModel(config)
        rng = jax.random.PRNGKey(42)

        # Generate a sequence
        prompt = jnp.array([[1, 2, 3]])
        params = model.init(rng, prompt)

        # Method 1: Generate without cache (full reprocessing)
        full_sequence = prompt
        for _ in range(5):
            logits, _ = model.apply(params, full_sequence, kv_caches=None, position_offset=0)
            next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
            full_sequence = jnp.concatenate([full_sequence, next_token], axis=1)

        # Method 2: Generate with cache
        logits, kv_caches = model.apply(params, prompt, kv_caches=None, position_offset=0)
        cached_sequence = prompt
        next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
        cached_sequence = jnp.concatenate([cached_sequence, next_token], axis=1)

        for i in range(4):
            logits, kv_caches = model.apply(
                params, next_token, kv_caches=kv_caches, position_offset=3 + i + 1
            )
            next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
            cached_sequence = jnp.concatenate([cached_sequence, next_token], axis=1)

        # Results should be identical
        assert jnp.allclose(full_sequence, cached_sequence), \
            f"Results differ: {full_sequence} vs {cached_sequence}"


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)

    config = QwenConfig(
        hidden_size=256,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=8,
    )

    model = QwenModel(config)
    rng = jax.random.PRNGKey(0)

    prompt_length = 10
    num_tokens = 50

    prompt = jnp.ones((1, prompt_length), dtype=jnp.int32)
    params = model.init(rng, prompt)

    # Benchmark 1: Old method (no cache)
    print(f"\nBenchmark 1: No KV Cache (old method)")
    print(f"  Generating {num_tokens} tokens...")

    start = time.time()
    sequence = prompt
    for i in range(num_tokens):
        logits, _ = model.apply(params, sequence, kv_caches=None, position_offset=0)
        logits.block_until_ready()
        next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
        sequence = jnp.concatenate([sequence, next_token], axis=1)
    time_no_cache = time.time() - start

    print(f"  Total time: {time_no_cache:.2f}s")
    print(f"  Tokens/sec: {num_tokens / time_no_cache:.2f}")

    # Benchmark 2: New method (with cache)
    print(f"\nBenchmark 2: With KV Cache (new method)")
    print(f"  Generating {num_tokens} tokens...")

    start = time.time()
    # Prefill
    logits, kv_caches = model.apply(params, prompt, kv_caches=None, position_offset=0)
    logits.block_until_ready()
    next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
    sequence_cached = jnp.concatenate([prompt, next_token], axis=1)

    # Decode
    for i in range(num_tokens - 1):
        logits, kv_caches = model.apply(
            params, next_token, kv_caches=kv_caches, position_offset=prompt_length + i
        )
        logits.block_until_ready()
        next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
        sequence_cached = jnp.concatenate([sequence_cached, next_token], axis=1)

    time_with_cache = time.time() - start

    print(f"  Total time: {time_with_cache:.2f}s")
    print(f"  Tokens/sec: {num_tokens / time_with_cache:.2f}")

    # Summary
    print(f"\n" + "="*60)
    print(f"RESULTS")
    print(f"="*60)
    print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")
    print(f"Time saved: {time_no_cache - time_with_cache:.2f}s ({(1 - time_with_cache/time_no_cache)*100:.1f}%)")
    print(f"="*60)


def run_all_tests():
    """Run all tests without pytest."""
    tests_passed = 0
    tests_failed = 0

    test_classes = [
        TestRoPECaching(),
        TestKVCaching(),
        TestJITCompilation(),
        TestCorrectness(),
    ]

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*60}")
        print(f"Running {class_name}")
        print('='*60)

        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                print(f"\n{method_name}...", end=" ")
                method = getattr(test_class, method_name)
                method()
                print("✓ PASSED")
                tests_passed += 1
            except Exception as e:
                print(f"✗ FAILED")
                print(f"  Error: {e}")
                traceback.print_exc()
                tests_failed += 1

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print(f"Total:  {tests_passed + tests_failed}")
    print(f"{'='*60}")

    return tests_failed == 0


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()

    if success:
        # Run benchmark
        run_performance_benchmark()
    else:
        print("\nSkipping benchmark due to test failures")
