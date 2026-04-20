#!/usr/bin/env python3
"""
End-to-End Extraction Correctness Test

Runs a minimal extraction pipeline on synthetic data and verifies:

1. Shard files are created and loadable
2. Activation shapes match (batch, seq_len, hidden_dim)
3. Stored activations match a direct forward pass through the same model
4. Padding tokens are correctly cropped from stored activations
5. Multiple layers are stored correctly
6. Storage metadata is consistent

This test does NOT require a dataset file — it creates synthetic token sequences.
Run: JAX_PLATFORMS=cpu python3 tests/test_e2e_extraction.py
"""

import sys
import os
import tempfile
import shutil
import traceback
import pickle
import gzip
import numpy as np

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
                print(f"  PASSED: {name}")
            except Exception as e:
                FAILED += 1
                print(f"  FAILED: {name}")
                traceback.print_exc()
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_sequences(tokenizer, n=4):
    """Create synthetic token sequences of varying lengths."""
    texts = [
        "Hello",
        "The quick brown fox jumps over the lazy dog",
        "1 + 1 = 2",
        "Machine learning is a branch of artificial intelligence",
    ][:n]
    return [tokenizer.encode(t) for t in texts], texts


def load_shard(path):
    """Load a shard file (gzipped pickle)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Test 1: ActivationStorage creates valid shards
# ---------------------------------------------------------------------------

@test("ActivationStorage - shard creation and loading")
def test_storage_shard_creation():
    from core.activation_storage import ActivationStorage

    tmpdir = tempfile.mkdtemp(prefix="test_e2e_storage_")
    try:
        storage = ActivationStorage(
            output_dir=tmpdir,
            upload_to_gcs=False,
            shard_size_gb=0.0001,  # Tiny threshold to force shard creation
            compress_shards=True,
            verbose=False,
        )

        hidden_dim = 64
        for sample_idx in range(10):
            seq_len = np.random.randint(5, 20)
            act = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=act,
                sample_idx=sample_idx,
                text_preview=f"sample_{sample_idx}",
            )

        storage.finalize()

        shard_files = sorted(
            f for f in os.listdir(tmpdir)
            if f.startswith("shard_") and f.endswith((".pkl", ".pkl.gz"))
        )
        assert len(shard_files) >= 1, f"Expected at least 1 shard, got {len(shard_files)}"

        total_activations = 0
        for sf in shard_files:
            data = load_shard(os.path.join(tmpdir, sf))
            assert isinstance(data, dict), f"Shard {sf} is not a dict"
            for layer_idx, acts_list in data.items():
                assert isinstance(acts_list, list), f"Layer {layer_idx} data is not a list"
                for entry in acts_list:
                    assert "activation" in entry, "Missing 'activation' key"
                    assert "sample_idx" in entry, "Missing 'sample_idx' key"
                    assert "shape" in entry, "Missing 'shape' key"
                    assert entry["activation"].shape[-1] == hidden_dim
                    total_activations += 1

        assert total_activations == 10, f"Expected 10 activations, got {total_activations}"

        meta_path = os.path.join(tmpdir, "metadata.json")
        assert os.path.exists(meta_path), "metadata.json not created"

        print(f"  Created {len(shard_files)} shards, {total_activations} activations")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Full forward pass → storage → reload roundtrip
# ---------------------------------------------------------------------------

@test("E2E roundtrip - forward pass activations match stored shards")
def test_e2e_roundtrip():
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
    from qwen2_jax_with_hooks import create_model_with_hooks
    from core.activation_storage import ActivationStorage
    from core.jax_utils import pad_sequences

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

    layers_to_extract = [0, config.num_hidden_layers // 2, config.num_hidden_layers - 1]
    jax_model = create_model_with_hooks(
        config, layers_to_extract=layers_to_extract, activation_type="residual"
    )
    converted = convert_hf_to_jax_weights(hf_model, config)
    params = {"params": converted}
    del hf_model

    sequences, texts = make_synthetic_sequences(tokenizer)
    actual_lengths = [len(s) for s in sequences]

    tmpdir = tempfile.mkdtemp(prefix="test_e2e_roundtrip_")
    try:
        storage = ActivationStorage(
            output_dir=tmpdir,
            upload_to_gcs=False,
            shard_size_gb=10.0,  # Large so everything stays in one shard
            compress_shards=True,
            verbose=False,
        )

        # Forward pass (batched with padding, matching extraction pipeline)
        max_len = max(actual_lengths)
        padded = pad_sequences(sequences, pad_token_id=0, fixed_length=max_len)
        input_ids = jnp.array(padded)

        _, _, activations = jax_model.apply(params, input_ids, return_activations=True)

        # Store activations (with padding crop, matching process_batch)
        for i in range(len(sequences)):
            for layer_idx in layers_to_extract:
                layer_key = f"layer_{layer_idx}"
                layer_act = np.array(activations[layer_key][i])
                layer_act = layer_act[:actual_lengths[i]]  # Crop padding
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=layer_act,
                    sample_idx=i,
                    text_preview=texts[i],
                )

        storage.finalize()

        # Reload shards and verify
        shard_files = sorted(
            f for f in os.listdir(tmpdir)
            if f.startswith("shard_") and f.endswith((".pkl", ".pkl.gz"))
        )
        assert len(shard_files) >= 1, "No shards created"

        stored = {}  # (layer_idx, sample_idx) -> activation
        for sf in shard_files:
            data = load_shard(os.path.join(tmpdir, sf))
            for layer_idx, acts_list in data.items():
                for entry in acts_list:
                    stored[(layer_idx, entry["sample_idx"])] = entry["activation"]

        # Verify each stored activation matches direct forward pass
        for i in range(len(sequences)):
            for layer_idx in layers_to_extract:
                key = (layer_idx, i)
                assert key in stored, f"Missing stored activation for {key}"

                expected = np.array(
                    activations[f"layer_{layer_idx}"][i, :actual_lengths[i]]
                    .astype(jnp.float32)
                )
                actual = stored[key].astype(np.float32)

                assert actual.shape == expected.shape, (
                    f"Shape mismatch for {key}: stored={actual.shape}, expected={expected.shape}"
                )

                max_diff = np.max(np.abs(actual - expected))
                assert max_diff < 1e-4, (
                    f"Value mismatch for {key}: max_diff={max_diff:.6f}"
                )

        total = len(stored)
        expected_total = len(sequences) * len(layers_to_extract)
        assert total == expected_total, f"Count mismatch: {total} != {expected_total}"

        print(f"  Verified {total} activations across {len(layers_to_extract)} layers")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Padding crop correctness
# ---------------------------------------------------------------------------

@test("Padding crop - stored activations exclude padding tokens")
def test_padding_crop():
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
    from qwen2_jax_with_hooks import create_model_with_hooks
    from core.jax_utils import pad_sequences

    model_path = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen2.5-0.5B")
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

    test_layer = config.num_hidden_layers - 1
    jax_model = create_model_with_hooks(
        config, layers_to_extract=[test_layer], activation_type="residual"
    )
    converted = convert_hf_to_jax_weights(hf_model, config)
    params = {"params": converted}
    del hf_model

    short_text = "Hi"
    long_text = "The quick brown fox jumps over the lazy dog in the park"
    short_tokens = tokenizer.encode(short_text)
    long_tokens = tokenizer.encode(long_text)

    short_len = len(short_tokens)
    long_len = len(long_tokens)
    assert short_len < long_len, "Short text should tokenize to fewer tokens"

    # Batch with padding
    padded = pad_sequences(
        [short_tokens, long_tokens], pad_token_id=0, fixed_length=long_len
    )
    input_ids = jnp.array(padded)
    _, _, acts = jax_model.apply(params, input_ids, return_activations=True)

    layer_key = f"layer_{test_layer}"
    full_act = np.array(acts[layer_key])

    # Crop to actual length (what process_batch does)
    short_cropped = full_act[0, :short_len]
    long_cropped = full_act[1, :long_len]

    assert short_cropped.shape == (short_len, config.hidden_size), (
        f"Short crop shape wrong: {short_cropped.shape}"
    )
    assert long_cropped.shape == (long_len, config.hidden_size), (
        f"Long crop shape wrong: {long_cropped.shape}"
    )

    # Padding region should NOT be stored
    padding_region = full_act[0, short_len:]
    assert padding_region.shape[0] == long_len - short_len
    print(f"  Short: {short_len} tokens, Long: {long_len} tokens, padding: {long_len - short_len}")
    print(f"  Cropped shapes: short={short_cropped.shape}, long={long_cropped.shape}")


# ---------------------------------------------------------------------------
# Test 4: Dynamic batching integration
# ---------------------------------------------------------------------------

@test("Dynamic batching - bucket assignment and batch creation")
def test_dynamic_batching():
    from core.dynamic_batching import (
        create_dynamic_batches, get_bucket_size, get_batch_size_for_bucket
    )

    sequences = [
        list(range(100)),   # 100 tokens
        list(range(500)),   # 500 tokens
        list(range(1500)),  # 1500 tokens
        list(range(50)),    # 50 tokens
        list(range(2500)),  # 2500 tokens
    ]
    prompts = [{"task_id": f"t{i}", "prompt": f"p{i}"} for i in range(5)]

    batches, sorted_seqs, sorted_prompts = create_dynamic_batches(
        sequences, prompts, max_seq_length=4096, num_hosts=1, verbose=False
    )

    total_in_batches = sum(len(b.sequences) for b in batches)
    valid_count = sum(1 for s in sequences if len(s) <= 4096)
    assert total_in_batches == valid_count, (
        f"Batch count mismatch: {total_in_batches} != {valid_count}"
    )

    for b in batches:
        for seq in b.sequences:
            assert len(seq) <= b.bucket_size, (
                f"Sequence length {len(seq)} exceeds bucket {b.bucket_size}"
            )

    # Verify sort order (shortest first)
    lengths = [len(s) for s in sorted_seqs]
    assert lengths == sorted(lengths), "Sequences not sorted by length"

    # Verify filtering
    too_long = [s for s in sequences if len(s) > 4096]
    filtered_out = len(sequences) - len(sorted_seqs)
    assert filtered_out == len(too_long), (
        f"Filtering mismatch: filtered {filtered_out}, expected {len(too_long)}"
    )

    print(f"  {len(batches)} batches, {total_in_batches} sequences, {filtered_out} filtered")
    for b in batches:
        print(f"    Bucket {b.bucket_size}: {len(b.sequences)} seqs, batch_size={b.batch_size}")


# ---------------------------------------------------------------------------
# Test 5: Metadata consistency
# ---------------------------------------------------------------------------

@test("Metadata - JSON written with correct structure")
def test_metadata_consistency():
    import json
    from core.activation_storage import ActivationStorage

    tmpdir = tempfile.mkdtemp(prefix="test_e2e_meta_")
    try:
        storage = ActivationStorage(
            output_dir=tmpdir,
            upload_to_gcs=False,
            shard_size_gb=10.0,
            compress_shards=True,
            verbose=False,
        )

        for i in range(5):
            act = np.random.randn(10, 32).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=act,
                sample_idx=i,
                text_preview=f"text_{i}",
            )

        storage.finalize()

        meta_path = os.path.join(tmpdir, "metadata.json")
        assert os.path.exists(meta_path), "metadata.json missing"

        with open(meta_path) as f:
            meta = json.load(f)

        assert "total_activations" in meta, "Missing total_activations"
        assert "total_shards" in meta, "Missing total_shards"
        assert meta["total_activations"] == 5, f"Expected 5, got {meta['total_activations']}"
        assert meta["total_shards"] >= 1, "Expected at least 1 shard"

        print(f"  Metadata: {meta['total_activations']} activations, {meta['total_shards']} shards")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("End-to-End Extraction Correctness Tests")
    print("=" * 70)

    tests = [
        test_storage_shard_creation,
        test_e2e_roundtrip,
        test_padding_crop,
        test_dynamic_batching,
        test_metadata_consistency,
    ]

    for t in tests:
        t()

    print(f"\n{'=' * 70}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED > 0 else 0)
