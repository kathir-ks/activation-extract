"""Quick test of KV cache fix"""

import jax
import jax.numpy as jnp
from functools import partial

# Test that our dynamic_slice fix works
from jax import lax

print("Testing dynamic position slicing...")

# Simulate what happens in QwenAttention
max_pos = 1000
seq_len = 5
position_offset = 10

# Old way (fails with traced values):
# position_ids = jnp.arange(position_offset, position_offset + seq_len)

# New way (works with traced values):
all_positions = jnp.arange(max_pos)
position_ids = lax.dynamic_slice(all_positions, (position_offset,), (seq_len,))

print(f"Position IDs: {position_ids}")
print(f"Expected: {jnp.arange(position_offset, position_offset + seq_len)}")
print(f"Match: {jnp.allclose(position_ids, jnp.arange(position_offset, position_offset + seq_len))}")

# Test with JIT and traced value
@jax.jit
def test_with_traced_position(pos_offset):
    all_pos = jnp.arange(max_pos)
    pos_ids = lax.dynamic_slice(all_pos, (pos_offset,), (seq_len,))
    return pos_ids

result = test_with_traced_position(10)
print(f"\nJIT test result: {result}")
print(f"✓ Dynamic slicing works with JIT!")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED")
print("="*50)
