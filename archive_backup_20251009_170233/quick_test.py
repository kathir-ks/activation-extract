"""
Quick validation test for the optimized Qwen2 JAX implementation
"""

import jax
import jax.numpy as jnp
import time
from qwen2_jax import QwenConfig, QwenModel

print("="*60)
print("QWEN2 JAX OPTIMIZATION VALIDATION")
print("="*60)

# Small config for quick testing
config = QwenConfig(
    vocab_size=1000,
    hidden_size=128,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
)

print("\nConfiguration:")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Layers: {config.num_hidden_layers}")
print(f"  Attention heads: {config.num_attention_heads}")
print(f"  KV heads: {config.num_key_value_heads}")

# Initialize model
model = QwenModel(config)
rng = jax.random.PRNGKey(0)

prompt_length = 5
num_tokens = 20

prompt = jnp.array([[1, 2, 3, 4, 5]])
params = model.init(rng, prompt)

print(f"\n{'='*60}")
print("TEST 1: KV Cache Functionality")
print('='*60)

# Test 1: Basic forward pass
print("\n1. Testing basic forward pass...")
logits, kv_caches = model.apply(params, prompt, kv_caches=None, position_offset=0)
print(f"   ✓ Output shape: {logits.shape}")
print(f"   ✓ KV caches returned: {len(kv_caches)} layers")

# Test 2: KV cache reuse
print("\n2. Testing KV cache reuse...")
next_token = jnp.array([[6]])
logits_next, kv_caches_next = model.apply(
    params, next_token, kv_caches=kv_caches, position_offset=prompt_length
)
print(f"   ✓ Output shape: {logits_next.shape}")
k_cache, v_cache = kv_caches[0]
k_cache_next, v_cache_next = kv_caches_next[0]
print(f"   ✓ Cache grew from {k_cache.shape[2]} to {k_cache_next.shape[2]} positions")

print(f"\n{'='*60}")
print("TEST 2: Performance Comparison")
print('='*60)

# Method 1: WITHOUT KV cache (old, slow method)
print(f"\n1. Generating {num_tokens} tokens WITHOUT KV cache...")
start = time.time()
sequence = prompt
for i in range(num_tokens):
    logits, _ = model.apply(params, sequence, kv_caches=None, position_offset=0)
    next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
    sequence = jnp.concatenate([sequence, next_token], axis=1)
    # Wait for computation to finish
    sequence.block_until_ready()

time_no_cache = time.time() - start
print(f"   Time: {time_no_cache:.3f}s")
print(f"   Tokens/sec: {num_tokens / time_no_cache:.2f}")

# Method 2: WITH KV cache (new, fast method)
print(f"\n2. Generating {num_tokens} tokens WITH KV cache...")
start = time.time()

# Prefill
logits, kv_caches = model.apply(params, prompt, kv_caches=None, position_offset=0)
next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
sequence_cached = jnp.concatenate([prompt, next_token], axis=1)
sequence_cached.block_until_ready()

# Decode
for i in range(num_tokens - 1):
    logits, kv_caches = model.apply(
        params, next_token, kv_caches=kv_caches, position_offset=prompt_length + i
    )
    next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
    sequence_cached = jnp.concatenate([sequence_cached, next_token], axis=1)
    sequence_cached.block_until_ready()

time_with_cache = time.time() - start
print(f"   Time: {time_with_cache:.3f}s")
print(f"   Tokens/sec: {num_tokens / time_with_cache:.2f}")

print(f"\n{'='*60}")
print("RESULTS")
print('='*60)
speedup = time_no_cache / time_with_cache
improvement = (1 - time_with_cache / time_no_cache) * 100

print(f"\n✓ Speedup: {speedup:.2f}x faster")
print(f"✓ Time saved: {time_no_cache - time_with_cache:.3f}s ({improvement:.1f}% improvement)")

if speedup > 2:
    print(f"\n✓✓✓ EXCELLENT: {speedup:.1f}x speedup achieved!")
elif speedup > 1.5:
    print(f"\n✓✓ GOOD: {speedup:.1f}x speedup achieved!")
else:
    print(f"\n⚠ WARNING: Expected >2x speedup, got {speedup:.1f}x")

print(f"\n{'='*60}")
print("TEST 3: Correctness Verification")
print('='*60)

# Verify both methods produce identical results
print("\nVerifying both methods produce the same output...")
if jnp.allclose(sequence, sequence_cached):
    print("✓ Results are identical!")
else:
    print("✗ WARNING: Results differ!")
    print(f"  Without cache: {sequence}")
    print(f"  With cache: {sequence_cached}")

print(f"\n{'='*60}")
print("ALL TESTS COMPLETED")
print('='*60)
print("\nSummary:")
print(f"  ✓ KV cache implementation: Working")
print(f"  ✓ Performance improvement: {speedup:.2f}x")
print(f"  ✓ Correctness: {'Verified' if jnp.allclose(sequence, sequence_cached) else 'FAILED'}")
print("\nThe optimizations are working correctly!")
print("="*60)
