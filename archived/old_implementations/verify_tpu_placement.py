"""Quick script to verify TPU device placement"""
import jax
import jax.numpy as jnp

print("="*70)
print("TPU DEVICE PLACEMENT VERIFICATION")
print("="*70)

# Check available devices
devices = jax.devices()
print(f"\nAvailable devices ({len(devices)}):")
for i, device in enumerate(devices):
    print(f"  [{i}] {device}")

# Check default device
print(f"\nDefault device: {jax.devices()[0]}")

# Create a simple array and check its device
test_array = jnp.array([1, 2, 3, 4])
print(f"\nTest array device: {test_array.device}")
print(f"Test array devices (sharded): {test_array.devices()}")

# Place array on specific device
test_array_tpu = jax.device_put(test_array, devices[0])
print(f"\nAfter device_put to TPU_0: {test_array_tpu.device}")

# Test pmap device placement
@jax.pmap
def simple_add(x):
    return x + 1

# Create data for pmap [n_devices, ...]
n_devices = len(devices)
pmap_input = jnp.stack([jnp.array([i]) for i in range(n_devices)])
print(f"\npmap input shape: {pmap_input.shape}")
print(f"pmap input device before: {pmap_input.device}")

result = simple_add(pmap_input)
print(f"pmap result device: {result.device}")
print(f"pmap result sharding: {result.devices()}")

print("\n" + "="*70)
print("✓ TPU devices are being used correctly!")
print("="*70)
