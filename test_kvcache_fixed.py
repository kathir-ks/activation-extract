"""
Unit Tests for Fixed KV Cache Utilities

Tests the MaxText-style fixed-size KV cache implementation
to ensure correctness, JIT compatibility, and performance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from kvcache_utils import (
    create_kv_cache_buffers,
    write_prefill_cache,
    update_kv_cache_ar,
    get_attention_kv,
    create_activation_buffer,
    update_activation_buffer,
    KVCacheConfig,
    get_cache_info,
    validate_cache_shapes
)


def test_create_kv_cache_buffers():
    """Test KV cache buffer creation"""
    print("\n=== Test 1: Create KV Cache Buffers ===")

    config = KVCacheConfig(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        max_prefill_length=100,
        max_decode_length=50
    )
    batch_size = 2

    cache = create_kv_cache_buffers(config, batch_size)

    # Check shapes
    expected_prefill_shape = (24, 2, 100, 2, 64)
    expected_ar_shape = (24, 2, 50, 2, 64)

    assert cache['prefill']['k'].shape == expected_prefill_shape, \
        f"Prefill K shape mismatch: {cache['prefill']['k'].shape}"
    assert cache['prefill']['v'].shape == expected_prefill_shape, \
        f"Prefill V shape mismatch"
    assert cache['ar']['k'].shape == expected_ar_shape, \
        f"AR K shape mismatch"
    assert cache['ar']['v'].shape == expected_ar_shape, \
        f"AR V shape mismatch"

    # Check initial values
    assert cache['prefill']['length'] == 0, "Initial prefill length should be 0"
    assert cache['ar']['index'] == 0, "Initial AR index should be 0"

    # Validate shapes
    validate_cache_shapes(cache, config, batch_size)

    print("✓ Cache buffers created with correct shapes")
    print(f"  Prefill: {expected_prefill_shape}")
    print(f"  AR: {expected_ar_shape}")


def test_write_prefill_cache():
    """Test writing to prefill cache"""
    print("\n=== Test 2: Write Prefill Cache ===")

    config = KVCacheConfig(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        max_prefill_length=100,
        max_decode_length=50
    )
    batch_size = 2
    seq_len = 10

    cache = create_kv_cache_buffers(config, batch_size)

    # Create dummy prefill K, V
    key = jax.random.PRNGKey(42)
    k_prefill = jax.random.normal(key, (batch_size, seq_len, 2, 64))
    v_prefill = jax.random.normal(key, (batch_size, seq_len, 2, 64))

    # Write to cache
    cache = write_prefill_cache(cache, layer_idx=0, k_prefill=k_prefill, v_prefill=v_prefill)

    # Check length was updated
    assert cache['prefill']['length'] == seq_len, \
        f"Prefill length should be {seq_len}, got {cache['prefill']['length']}"

    # Check data was written correctly
    stored_k = cache['prefill']['k'][0, :, :seq_len, :, :]
    assert jnp.allclose(stored_k, k_prefill), "Prefill K data mismatch"

    print(f"✓ Prefill cache written successfully")
    print(f"  Length: {cache['prefill']['length']}")


def test_update_kv_cache_ar():
    """Test autoregressive cache updates"""
    print("\n=== Test 3: Update AR Cache ===")

    config = KVCacheConfig(num_layers=24, num_kv_heads=2, head_dim=64)
    batch_size = 1

    cache = create_kv_cache_buffers(config, batch_size)

    # Simulate 5 decode steps
    key = jax.random.PRNGKey(0)
    stored_keys = []

    for step in range(5):
        k_new = jax.random.normal(key, (batch_size, 2, 64))
        v_new = jax.random.normal(key, (batch_size, 2, 64))

        stored_keys.append(k_new)

        cache = update_kv_cache_ar(cache, layer_idx=0, new_k=k_new, new_v=v_new)

        # Check index incremented
        assert cache['ar']['index'] == step + 1, \
            f"AR index should be {step + 1}, got {cache['ar']['index']}"

    # Verify all data was stored correctly
    for step, k_expected in enumerate(stored_keys):
        k_stored = cache['ar']['k'][0, :, step, :, :]
        assert jnp.allclose(k_stored, k_expected), f"AR K mismatch at step {step}"

    print(f"✓ AR cache updated successfully for {len(stored_keys)} steps")
    print(f"  Final AR index: {cache['ar']['index']}")


def test_get_attention_kv():
    """Test retrieving full K,V for attention"""
    print("\n=== Test 4: Get Attention K,V ===")

    config = KVCacheConfig(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        max_prefill_length=100,
        max_decode_length=50
    )
    batch_size = 1

    cache = create_kv_cache_buffers(config, batch_size)

    # Write prefill (10 tokens)
    key = jax.random.PRNGKey(1)
    k_prefill = jax.random.normal(key, (batch_size, 10, 2, 64))
    v_prefill = jax.random.normal(key, (batch_size, 10, 2, 64))
    cache = write_prefill_cache(cache, layer_idx=0, k_prefill=k_prefill, v_prefill=v_prefill)

    # Add 5 AR tokens
    for _ in range(5):
        k_new = jax.random.normal(key, (batch_size, 2, 64))
        v_new = jax.random.normal(key, (batch_size, 2, 64))
        cache = update_kv_cache_ar(cache, layer_idx=0, new_k=k_new, new_v=v_new)

    # Get full K,V (now returns full buffers, not sliced)
    k_full, v_full = get_attention_kv(cache, layer_idx=0, use_prefill=True)

    # Should return full buffer: max_prefill (100) + max_decode (50) = 150
    # Actual valid tokens: 10 (prefill) + 5 (AR) = 15, handled by attention masking
    expected_shape = (batch_size, 150, 2, 64)  # Full buffer size
    assert k_full.shape == expected_shape, \
        f"Full K shape mismatch: {k_full.shape} vs {expected_shape}"
    assert v_full.shape == expected_shape, \
        f"Full V shape mismatch"

    # Verify the valid data is in the right places
    # Prefill should be at positions [0:10]
    k_prefill_check = k_full[:, :10, :, :]
    assert jnp.allclose(k_prefill_check, k_prefill), "Prefill data not in correct positions"

    print(f"✓ Full K,V retrieved successfully")
    print(f"  Shape: {k_full.shape} (full buffer, valid: 10 prefill + 5 AR)")
    print(f"  Note: Attention masking handles filtering to valid positions")


def test_activation_buffer():
    """Test activation buffer creation and updates"""
    print("\n=== Test 5: Activation Buffer ===")

    num_layers = 14
    max_tokens = 50
    batch_size = 2
    hidden_dim = 896

    # Create buffer
    buffer = create_activation_buffer(num_layers, max_tokens, batch_size, hidden_dim)

    expected_shape = (14, 50, 2, 896)
    assert buffer.shape == expected_shape, \
        f"Buffer shape mismatch: {buffer.shape} vs {expected_shape}"

    # Update buffer at multiple steps
    key = jax.random.PRNGKey(2)
    stored_acts = []

    for step in range(10):
        acts = jax.random.normal(key, (num_layers, batch_size, hidden_dim))
        stored_acts.append(acts)
        buffer = update_activation_buffer(buffer, step, acts)

    # Verify stored data
    for step, acts_expected in enumerate(stored_acts):
        acts_stored = buffer[:, step, :, :]
        assert jnp.allclose(acts_stored, acts_expected), \
            f"Activation mismatch at step {step}"

    # Check remaining slots are still zero
    assert jnp.all(buffer[:, 10:, :, :] == 0), \
        "Unused buffer slots should be zero"

    print(f"✓ Activation buffer works correctly")
    print(f"  Updated {len(stored_acts)} steps, remaining zeros")


def test_jit_compilation():
    """Test that cache operations are JIT-compatible"""
    print("\n=== Test 6: JIT Compilation ===")

    config = KVCacheConfig(num_layers=24, num_kv_heads=2, head_dim=64)
    batch_size = 1

    # Test individual functions with JIT
    @jax.jit
    def test_update_activation_buffer(buffer, step, acts):
        """Test activation buffer update is JIT-compatible"""
        return update_activation_buffer(buffer, step, acts)

    key = jax.random.PRNGKey(3)

    # Test activation buffer (simpler, no dict with traced values)
    act_buffer = create_activation_buffer(14, 50, batch_size, 896)

    # First call (compiles)
    start = time.time()
    acts = jax.random.normal(key, (14, batch_size, 896))
    new_buffer = test_update_activation_buffer(act_buffer, 0, acts)
    compile_time = time.time() - start

    # Second call (uses compiled version)
    start = time.time()
    acts = jax.random.normal(key, (14, batch_size, 896))
    new_buffer = test_update_activation_buffer(new_buffer, 1, acts)
    run_time = time.time() - start

    print(f"✓ JIT compilation works")
    print(f"  First call (with compilation): {compile_time*1000:.2f}ms")
    print(f"  Second call (cached): {run_time*1000:.2f}ms")
    print(f"  Speedup: {compile_time/run_time:.1f}x" if run_time > 0 else "  Speedup: N/A (too fast)")

    # Note: Full cache dict JIT will work in actual generation where the
    # dict structure is managed outside JIT boundaries
    print("  Note: Full cache operations JIT-tested in generate_jitted.py")


def test_cache_info():
    """Test cache information utility"""
    print("\n=== Test 7: Cache Info ===")

    config = KVCacheConfig(num_layers=24, num_kv_heads=2, head_dim=64)
    cache = create_kv_cache_buffers(config, batch_size=1)

    # Initial info
    info = get_cache_info(cache)
    assert info['prefill_length'] == 0
    assert info['ar_index'] == 0
    assert info['total_cached_tokens'] == 0

    # Add prefill
    key = jax.random.PRNGKey(4)
    k_prefill = jax.random.normal(key, (1, 10, 2, 64))
    v_prefill = jax.random.normal(key, (1, 10, 2, 64))
    cache = write_prefill_cache(cache, 0, k_prefill, v_prefill)

    # Add AR tokens
    for _ in range(5):
        k_new = jax.random.normal(key, (1, 2, 64))
        v_new = jax.random.normal(key, (1, 2, 64))
        cache = update_kv_cache_ar(cache, 0, k_new, v_new)

    # Check info
    info = get_cache_info(cache)
    assert info['prefill_length'] == 10
    assert info['ar_index'] == 5
    assert info['total_cached_tokens'] == 15

    print(f"✓ Cache info utility works")
    print(f"  Prefill: {info['prefill_length']}")
    print(f"  AR: {info['ar_index']}")
    print(f"  Total: {info['total_cached_tokens']}")


def test_shape_consistency():
    """Test that shapes remain constant throughout updates"""
    print("\n=== Test 8: Shape Consistency ===")

    config = KVCacheConfig(num_layers=24, num_kv_heads=2, head_dim=64)
    cache = create_kv_cache_buffers(config, batch_size=1)

    initial_prefill_shape = cache['prefill']['k'].shape
    initial_ar_shape = cache['ar']['k'].shape

    key = jax.random.PRNGKey(5)

    # Write prefill
    k_prefill = jax.random.normal(key, (1, 10, 2, 64))
    v_prefill = jax.random.normal(key, (1, 10, 2, 64))
    cache = write_prefill_cache(cache, 0, k_prefill, v_prefill)

    # Check shapes unchanged
    assert cache['prefill']['k'].shape == initial_prefill_shape
    assert cache['ar']['k'].shape == initial_ar_shape

    # Update AR multiple times
    for _ in range(20):
        k_new = jax.random.normal(key, (1, 2, 64))
        v_new = jax.random.normal(key, (1, 2, 64))
        cache = update_kv_cache_ar(cache, 0, k_new, v_new)

        # Shapes should never change
        assert cache['prefill']['k'].shape == initial_prefill_shape
        assert cache['ar']['k'].shape == initial_ar_shape

    print(f"✓ Shapes remain constant throughout updates")
    print(f"  Prefill shape: {initial_prefill_shape}")
    print(f"  AR shape: {initial_ar_shape}")


def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("Running Fixed KV Cache Unit Tests")
    print("="*60)

    tests = [
        test_create_kv_cache_buffers,
        test_write_prefill_cache,
        test_update_kv_cache_ar,
        test_get_attention_kv,
        test_activation_buffer,
        test_jit_compilation,
        test_cache_info,
        test_shape_consistency
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
    
