"""Quick timing test to identify bottleneck"""
import time
import jax
import jax.numpy as jnp
from extract_activations_fineweb_multihost import (
    load_model_and_tokenizer,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded,
    QwenConfig
)

print("="*70)
print("TIMING TEST")
print("="*70)

# Config
config = QwenConfig(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
)

print("\n1. Loading model...")
t0 = time.time()
model, tokenizer, params = load_model_and_tokenizer(
    "Qwen/Qwen2.5-0.5B", config, [22, 23]
)
print(f"   Time: {time.time()-t0:.2f}s")

print("\n2. Creating mesh...")
t0 = time.time()
mesh, _ = create_device_mesh(4, mesh_type='1d', num_hosts=1)
print(f"   Time: {time.time()-t0:.2f}s")

print("\n3. Sharding parameters...")
t0 = time.time()
sharding_rules = create_sharding_strategy(mesh)
with mesh:
    params = shard_params(params, mesh, sharding_rules)
print(f"   Time: {time.time()-t0:.2f}s")

# Create dummy input
print("\n4. Creating dummy input (batch=2, seq=256)...")
t0 = time.time()
input_ids = jnp.ones((2, 256), dtype=jnp.int32)
print(f"   Time: {time.time()-t0:.2f}s")

# First forward pass (JIT compilation)
print("\n5. First forward pass (JIT compiling)...")
t0 = time.time()
with mesh:
    activations = extract_activations_sharded(model, params, input_ids)
# Force computation
_ = jax.tree_map(lambda x: x.block_until_ready(), activations)
print(f"   Time: {time.time()-t0:.2f}s  <- INCLUDES JIT COMPILATION")

# Second forward pass (already compiled)
print("\n6. Second forward pass (already compiled)...")
t0 = time.time()
with mesh:
    activations = extract_activations_sharded(model, params, input_ids)
# Force computation
_ = jax.tree_map(lambda x: x.block_until_ready(), activations)
print(f"   Time: {time.time()-t0:.2f}s  <- SHOULD BE FAST!")

# Test numpy conversion
print("\n7. Converting to numpy...")
t0 = time.time()
for layer_key, layer_act in activations.items():
    _ = layer_act.block_until_ready()  # Force computation first
    for i in range(2):  # 2 samples in batch
        act_np = layer_act[i]  # Get one sample
        _ = act_np.block_until_ready()
print(f"   Time: {time.time()-t0:.2f}s")

print("\n" + "="*70)
print("TIMING COMPLETE")
print("="*70)
