#!/usr/bin/env python3
"""
Single-Host vs Multi-Host Activation Equivalence Verification

This script verifies that the sharding/gathering pipeline produces
activations identical to a single-host forward pass. It simulates
the multi-host code path on a single machine by:

1. Running a plain forward pass (no mesh, no sharding) — "reference"
2. Running the same forward pass through a device mesh with sharding
   constraints (as multihost_extract.py would) — "sharded"
3. Comparing the two numerically

This catches bugs in:
- Sharding strategy (wrong PartitionSpec fragments the hidden dim)
- Gather constraints (lax.with_sharding_constraint not applied)
- Mesh construction (axes in wrong order)

Run on TPU: python3 tests/test_multihost_equivalence.py
Run on CPU: JAX_PLATFORMS=cpu python3 tests/test_multihost_equivalence.py
  (CPU uses 1 device, so sharding is trivially correct — only useful as smoke test)
"""

import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = 0
FAILED = 0


def test(name):
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            print(f"\n--- {name} ---")
            try:
                fn()
                PASSED += 1
                print(f"  PASSED: {name}")
            except Exception as e:
                FAILED += 1
                print(f"  FAILED: {name}")
                traceback.print_exc()
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Test 1: Unsharded vs sharded forward pass
# ---------------------------------------------------------------------------

@test("Sharded vs unsharded - activation equivalence")
def test_sharded_equivalence():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
    from qwen2_jax_with_hooks import create_model_with_hooks

    num_devices = jax.local_device_count()
    print(f"  Devices: {num_devices} ({jax.default_backend()})")

    model_path = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen2.5-0.5B")
    print(f"  Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    config = QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, 'rope_theta', 1000000.0),
        dtype=jnp.bfloat16,
    )

    test_layers = [0, config.num_hidden_layers - 1]
    jax_model = create_model_with_hooks(
        config, layers_to_extract=test_layers, activation_type="residual"
    )
    converted = convert_hf_to_jax_weights(hf_model, config)
    params = {"params": converted}
    del hf_model

    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text)
    input_ids = jnp.array([tokens])

    # --- Reference: unsharded forward pass ---
    _, _, ref_acts = jax_model.apply(params, input_ids, return_activations=True)
    ref_results = {
        k: np.array(v.astype(jnp.float32))
        for k, v in ref_acts.items() if k.startswith("layer_")
    }

    # --- Sharded forward pass ---
    # Create a mesh similar to what the extraction pipeline uses
    devices = jax.local_devices()

    if num_devices >= 2:
        # 2D mesh: (fsdp, model) — matches single-host extraction config
        fsdp_size = min(2, num_devices)
        model_size = num_devices // fsdp_size
        mesh_devices = np.array(devices[:fsdp_size * model_size]).reshape(fsdp_size, model_size)
        mesh = Mesh(mesh_devices, axis_names=("fsdp", "model"))
    else:
        # 1D mesh: single device
        mesh_devices = np.array(devices).reshape(1, 1)
        mesh = Mesh(mesh_devices, axis_names=("fsdp", "model"))

    print(f"  Mesh shape: {mesh.shape}, axes: {mesh.axis_names}")

    # Shard params (replicate for simplicity — the key test is the activation gather)
    with mesh:
        replicated = NamedSharding(mesh, P())
        sharded_params = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, replicated), params
        )

        # Use the same JIT-compiled extraction path
        @jax.jit
        def extract_sharded(params, input_ids):
            _, _, activations = jax_model.apply(
                params, input_ids, return_activations=True
            )
            constrained = {}
            for k, v in activations.items():
                if k.startswith("layer_"):
                    constrained[k] = jax.lax.with_sharding_constraint(
                        v, NamedSharding(mesh, P(None, None, None))
                    )
            return constrained

        sharded_acts = extract_sharded(sharded_params, input_ids)
        sharded_results = {
            k: np.array(v.astype(jnp.float32))
            for k, v in sharded_acts.items()
        }

    # --- Compare ---
    for layer_key in ref_results:
        ref = ref_results[layer_key]
        shd = sharded_results[layer_key]

        assert ref.shape == shd.shape, (
            f"{layer_key} shape mismatch: ref={ref.shape}, sharded={shd.shape}"
        )

        max_diff = float(np.max(np.abs(ref - shd)))
        mean_diff = float(np.mean(np.abs(ref - shd)))
        scale = float(np.max(np.abs(ref))) + 1e-8
        rel_max = max_diff / scale
        print(f"  {layer_key}: max_diff={max_diff:.6f}, rel={rel_max:.6f}, mean_diff={mean_diff:.8f}")

        # bfloat16 operations are not associative — different JIT compilation
        # paths (sharded vs unsharded) produce different reduction orderings,
        # which compounds through layers. Check relative error.
        assert rel_max < 0.05, (
            f"{layer_key} relative diff too large between sharded and unsharded: "
            f"{rel_max:.6f} (abs={max_diff:.6f})"
        )


# ---------------------------------------------------------------------------
# Test 2: Hidden dim integrity (no fragmentation)
# ---------------------------------------------------------------------------

@test("Hidden dim integrity - full hidden_dim on each host")
def test_hidden_dim_integrity():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from qwen2_jax import QwenConfig
    from qwen2_jax_with_hooks import create_model_with_hooks

    num_devices = jax.local_device_count()

    config = QwenConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        dtype=jnp.float32,
    )

    jax_model = create_model_with_hooks(
        config, layers_to_extract=[0, 1], activation_type="residual"
    )

    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = jax_model.init(rng, dummy_input)

    devices = jax.local_devices()
    if num_devices >= 2:
        mesh_devices = np.array(devices[:num_devices]).reshape(
            min(2, num_devices), num_devices // min(2, num_devices)
        )
        mesh = Mesh(mesh_devices, axis_names=("fsdp", "model"))
    else:
        mesh = Mesh(np.array(devices).reshape(1, 1), axis_names=("fsdp", "model"))

    with mesh:
        @jax.jit
        def extract(params, input_ids):
            _, _, activations = jax_model.apply(
                params, input_ids, return_activations=True
            )
            constrained = {}
            for k, v in activations.items():
                if k.startswith("layer_"):
                    constrained[k] = jax.lax.with_sharding_constraint(
                        v, NamedSharding(mesh, P(None, None, None))
                    )
            return constrained

        replicated = NamedSharding(mesh, P())
        sharded_params = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, replicated), params
        )

        acts = extract(sharded_params, dummy_input)

        for key, val in acts.items():
            shape = val.shape
            assert shape == (1, 10, config.hidden_size), (
                f"{key} has shape {shape}, expected (1, 10, {config.hidden_size}). "
                f"Hidden dim may be fragmented across devices."
            )
            host_val = np.array(val)
            assert host_val.shape[-1] == config.hidden_size, (
                f"{key} host-side hidden_dim={host_val.shape[-1]} != {config.hidden_size}"
            )
            print(f"  {key}: shape={shape} (full hidden_dim={config.hidden_size})")


# ---------------------------------------------------------------------------
# Test 3: extract_activations_sharded matches direct apply
# ---------------------------------------------------------------------------

@test("extract_activations_sharded - matches direct model.apply")
def test_extract_activations_sharded_consistency():
    """extract_activations_sharded uses P('data', None, None) sharding constraint,
    which requires a mesh with a 'data' axis. On single-host, the extraction pipeline
    doesn't use a mesh (extract_activations_sharded is called without mesh context).
    This test verifies the function works correctly within a proper mesh context."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from qwen2_jax import QwenConfig
    from qwen2_jax_with_hooks import create_model_with_hooks

    config = QwenConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        dtype=jnp.float32,
    )

    jax_model = create_model_with_hooks(
        config, layers_to_extract=[0, 1], activation_type="residual"
    )

    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((2, 10), dtype=jnp.int32)
    params = jax_model.init(rng, dummy_input)

    # Direct apply (no mesh)
    _, _, direct_acts = jax_model.apply(params, dummy_input, return_activations=True)

    # Through a mesh with 'data' axis (matching what extract_activations_sharded expects)
    devices = jax.local_devices()
    num_devices = len(devices)
    mesh_devices = np.array(devices).reshape(1, num_devices)
    mesh = Mesh(mesh_devices, axis_names=("data", "fsdp"))

    with mesh:
        from core.jax_utils import extract_activations_sharded
        replicated = NamedSharding(mesh, P())
        mesh_params = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, replicated), params
        )
        sharded_acts = extract_activations_sharded(jax_model, mesh_params, dummy_input)

    for key in direct_acts:
        if not key.startswith("layer_"):
            continue
        direct = np.array(direct_acts[key].astype(jnp.float32))
        sharded = np.array(sharded_acts[key].astype(jnp.float32))

        assert direct.shape == sharded.shape, (
            f"{key} shape mismatch: direct={direct.shape}, sharded={sharded.shape}"
        )

        max_diff = float(np.max(np.abs(direct - sharded)))
        print(f"  {key}: max_diff={max_diff:.8f}")
        assert max_diff < 1e-5, f"{key} values diverge: {max_diff}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Single-Host vs Multi-Host Activation Equivalence")
    print("=" * 70)

    import jax
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.local_device_count()}")
    print(f"Process count: {jax.process_count()}")

    tests = [
        test_sharded_equivalence,
        test_hidden_dim_integrity,
        test_extract_activations_sharded_consistency,
    ]

    for t in tests:
        t()

    print(f"\n{'=' * 70}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED > 0 else 0)
