#!/usr/bin/env python3
"""
Unit tests for code review fixes:
  1. Weight validation (ValueError on missing weights)
  2. RoPE deduplication (compute_rope_embeddings helper)
  3. GQA broadcast (_expand_kv_heads using broadcast_to)
  4. ActivationStorage counter (total_activations vs unique samples)
  5. Model forward pass (QwenModel + hooks model with shared RoPE)
"""

import sys
import os
import tempfile
import traceback
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = 0
FAILED = 0

def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            print(f"\n--- {name} ---")
            try:
                fn()
                PASSED += 1
                print(f"✅ PASSED: {name}")
            except Exception as e:
                FAILED += 1
                print(f"❌ FAILED: {name}")
                traceback.print_exc()
        return wrapper
    return decorator


@test("Import verification")
def test_imports():
    from qwen2_jax import (
        compute_rope_embeddings, _expand_kv_heads,
        convert_hf_to_jax_weights, QwenConfig, QwenModel
    )
    from qwen2_jax_with_hooks import (
        QwenModelWithActivations, QwenDecoderLayerWithHooks,
        compute_rope_embeddings as cre_hook, create_model_with_hooks
    )
    from core.activation_storage import ActivationStorage
    assert compute_rope_embeddings is cre_hook, "Should be same function via import"


@test("RoPE compute_rope_embeddings - shapes and values")
def test_rope():
    import jax.numpy as jnp
    from qwen2_jax import compute_rope_embeddings

    head_dim, max_pos, theta = 64, 32768, 1000000.0
    cos, sin = compute_rope_embeddings(head_dim, max_pos, theta)

    assert cos.shape == (max_pos, head_dim), f"cos shape: {cos.shape}"
    assert sin.shape == (max_pos, head_dim), f"sin shape: {sin.shape}"

    # Verify values match direct computation
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(max_pos).astype(jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    assert jnp.allclose(cos, jnp.cos(emb)), "cos values mismatch"
    assert jnp.allclose(sin, jnp.sin(emb)), "sin values mismatch"


@test("GQA _expand_kv_heads - broadcast correctness")
def test_gqa():
    import jax.numpy as jnp
    from qwen2_jax import _expand_kv_heads

    # 2 KV heads, expand with 7 groups -> 14 query heads
    x = jnp.zeros((2, 2, 10, 64))
    x = x.at[:, 0].set(1.0)
    x = x.at[:, 1].set(2.0)
    expanded = _expand_kv_heads(x, 7)

    assert expanded.shape == (2, 14, 10, 64), f"shape: {expanded.shape}"
    for g in range(7):
        assert jnp.allclose(expanded[:, g], 1.0), f"head 0, group {g}"
        assert jnp.allclose(expanded[:, 7 + g], 2.0), f"head 1, group {g}"


@test("Weight validation - raises ValueError on missing weights")
def test_weight_validation():
    from qwen2_jax import convert_hf_to_jax_weights, QwenConfig

    class FakeModel:
        def state_dict(self):
            return {}

    try:
        convert_hf_to_jax_weights(FakeModel(), QwenConfig())
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "embed_tokens" in str(e), f"Error should mention embed_tokens: {e}"


@test("Weight validation - raises ValueError on missing layer weights")
def test_weight_validation_layer():
    import torch
    from qwen2_jax import convert_hf_to_jax_weights, QwenConfig

    # Provide embed_tokens but no layer weights
    class PartialModel:
        def state_dict(self):
            return {
                'model.embed_tokens.weight': torch.randn(151936, 896),
            }

    try:
        convert_hf_to_jax_weights(PartialModel(), QwenConfig())
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "layer 0" in str(e), f"Error should mention layer 0: {e}"
        assert "attention" in str(e).lower() or "self_attn" in str(e).lower(), \
            f"Error should mention attention weights: {e}"


@test("ActivationStorage - total_activations vs unique samples")
def test_storage_counter():
    from core.activation_storage import ActivationStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ActivationStorage(output_dir=tmpdir, verbose=False)
        # 3 unique samples, each across 2 layers = 6 activations
        for sample_idx in range(3):
            for layer_idx in range(2):
                storage.add_activation(
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                    activation=np.random.randn(10, 64).astype(np.float32),
                    token_ids=[1, 2, 3]
                )
        assert storage.total_activations == 6, \
            f"Expected 6 activations, got {storage.total_activations}"
        assert len(storage.seen_sample_indices) == 3, \
            f"Expected 3 unique samples, got {len(storage.seen_sample_indices)}"


@test("QwenModel forward pass with shared RoPE")
def test_model_forward():
    import jax
    import jax.numpy as jnp
    from qwen2_jax import QwenModel, QwenConfig

    config = QwenConfig()
    model = QwenModel(config)
    dummy = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy)
    logits, kv_caches = model.apply(params, dummy)

    assert logits.shape == (1, 10, config.vocab_size), f"logits: {logits.shape}"
    assert len(kv_caches) == config.num_hidden_layers, f"kv: {len(kv_caches)}"


@test("Hooks model forward pass with shared RoPE + activation extraction")
def test_hooks_model():
    import jax
    import jax.numpy as jnp
    from qwen2_jax import QwenConfig
    from qwen2_jax_with_hooks import create_model_with_hooks

    config = QwenConfig()
    model = create_model_with_hooks(config, layers_to_extract=[0, 12, 23])
    dummy = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy)

    # Test with activations
    logits, kv_caches, activations = model.apply(
        params, dummy, return_activations=True
    )
    assert logits.shape == (1, 10, config.vocab_size), f"logits: {logits.shape}"
    assert 'layer_0' in activations, f"keys: {activations.keys()}"
    assert 'layer_12' in activations, f"keys: {activations.keys()}"
    assert 'layer_23' in activations, f"keys: {activations.keys()}"
    assert activations['layer_0'].shape == (1, 10, config.hidden_size)

    # Test without activations (fast path)
    logits2, kv2 = model.apply(params, dummy, return_activations=False)
    assert logits2.shape == logits.shape, "Fast path logits shape mismatch"


if __name__ == "__main__":
    print("=" * 60)
    print("  CODE REVIEW FIXES - UNIT TESTS")
    print("=" * 60)

    tests = [
        test_imports,
        test_rope,
        test_gqa,
        test_weight_validation,
        test_weight_validation_layer,
        test_storage_counter,
        test_model_forward,
        test_hooks_model,
    ]

    for t in tests:
        t()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print("=" * 60)

    sys.exit(1 if FAILED > 0 else 0)
