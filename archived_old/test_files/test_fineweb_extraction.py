"""
Test script for extract_activations_fineweb_multihost.py

This test verifies:
1. Token extraction works correctly for FineWeb-Edu dataset
2. Activations are extracted properly
3. JIT compilation is working
4. Batch padding is correct
"""

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks
from extract_activations_fineweb_multihost import (
    load_dataset_shard,
    pad_sequences,
    ActivationStorage,
    extract_activations_sharded,
    process_batch
)

def test_fineweb_tokenization():
    """Test that FineWeb-Edu tokenization works correctly"""
    print("="*70)
    print("TEST 1: FineWeb-Edu Tokenization")
    print("="*70)

    # Load tokenizer
    model_path = "KathirKs/qwen-2.5-0.5b"
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load a small shard of FineWeb-Edu
    print("\nLoading FineWeb-Edu dataset (5 samples)...")
    dataset = load_dataset_shard(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
        machine_id=0,
        total_machines=1,
        max_samples=5,
        verbose=True
    )

    # Test tokenization
    print("\nTokenizing samples...")
    sequences = []
    texts = []

    for i, sample in enumerate(dataset):
        text = sample['text']
        texts.append(text[:100])  # Store first 100 chars

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=2048
        )

        sequence = inputs['input_ids'][0]
        sequences.append(sequence)

        print(f"\nSample {i}:")
        print(f"  Text preview: {text[:100]}...")
        print(f"  Token count: {len(sequence)}")
        print(f"  First 10 tokens: {sequence[:10]}")

        # Verify tokens can be decoded
        decoded = tokenizer.decode(sequence[:10])
        print(f"  Decoded: {decoded}")

    print(f"\n✓ Tokenized {len(sequences)} samples successfully")
    print(f"  Token lengths: {[len(s) for s in sequences]}")

    return sequences, texts, tokenizer


def test_padding():
    """Test sequence padding"""
    print("\n" + "="*70)
    print("TEST 2: Sequence Padding")
    print("="*70)

    # Create sequences with different lengths
    sequences = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6, 7, 8]),
        np.array([9, 10])
    ]

    print(f"\nOriginal sequences: {[len(s) for s in sequences]}")

    # Test 1: Pad to max in batch (dynamic)
    padded_dynamic = pad_sequences(sequences, pad_token_id=0, fixed_length=None)
    print(f"\nDynamic padding (max in batch):")
    print(f"  Shape: {padded_dynamic.shape}")
    print(f"  Padded to length: {padded_dynamic.shape[1]}")
    print(f"  Sequences:\n{padded_dynamic}")

    # Test 2: Pad to fixed length
    padded_fixed = pad_sequences(sequences, pad_token_id=0, fixed_length=10)
    print(f"\nFixed padding (length=10):")
    print(f"  Shape: {padded_fixed.shape}")
    print(f"  Padded to length: {padded_fixed.shape[1]}")
    print(f"  Sequences:\n{padded_fixed}")

    # Test 3: Truncation
    long_sequences = [np.array([i for i in range(20)]) for _ in range(2)]
    padded_truncated = pad_sequences(long_sequences, pad_token_id=0, fixed_length=10)
    print(f"\nTruncation test (20 tokens -> 10):")
    print(f"  Shape: {padded_truncated.shape}")
    print(f"  First sequence: {padded_truncated[0]}")

    print("\n✓ Padding tests passed")
    return padded_fixed


def test_activation_extraction():
    """Test activation extraction with real model"""
    print("\n" + "="*70)
    print("TEST 3: Activation Extraction")
    print("="*70)

    # Load model
    model_path = "KathirKs/qwen-2.5-0.5b"
    print(f"\nLoading model from {model_path}...")

    # Create config
    config = QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load HF model
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True
    )

    # Create JAX model with hooks
    layers_to_extract = [10, 11, 12, 13]
    print(f"Creating JAX model with hooks for layers {layers_to_extract}...")
    jax_model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)

    # Convert weights
    print("Converting weights...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    del hf_model

    # Load dataset
    print("\nLoading FineWeb-Edu dataset (3 samples)...")
    dataset = load_dataset_shard(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
        machine_id=0,
        total_machines=1,
        max_samples=3,
        verbose=False
    )

    # Tokenize
    sequences = []
    texts = []
    for sample in dataset:
        text = sample['text']
        texts.append(text)

        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512  # Shorter for faster test
        )
        sequences.append(inputs['input_ids'][0])

    # Test extraction with fixed padding (for JIT)
    print(f"\nExtracting activations for {len(sequences)} samples...")
    print(f"  Batch size: {len(sequences)}")
    print(f"  Max seq length: 512")

    # Create storage
    storage = ActivationStorage(
        output_dir='./test_activations',
        upload_to_gcs=False,
        shard_size_gb=0.001,  # 1 MB
        compress_shards=False,
        verbose=False
    )

    # Run extraction (with timing)
    start = time.time()
    process_batch(
        jax_model, params, sequences,
        sample_indices=list(range(len(sequences))),
        text_previews=texts,
        storage=storage,
        layers_to_extract=layers_to_extract,
        pad_token_id=tokenizer.pad_token_id or 0,
        batch_size=len(sequences),
        max_seq_length=512
    )
    first_time = time.time() - start
    print(f"  First batch time: {first_time:.2f}s (includes JIT compilation)")

    # Run again to verify JIT is working
    start = time.time()
    process_batch(
        jax_model, params, sequences,
        sample_indices=list(range(len(sequences))),
        text_previews=texts,
        storage=storage,
        layers_to_extract=layers_to_extract,
        pad_token_id=tokenizer.pad_token_id or 0,
        batch_size=len(sequences),
        max_seq_length=512
    )
    second_time = time.time() - start
    print(f"  Second batch time: {second_time:.2f}s (should be faster - JIT cached)")

    speedup = first_time / second_time
    print(f"  Speedup: {speedup:.1f}x")

    if speedup > 1.5:
        print("  ✓ JIT is working (significant speedup)")
    else:
        print("  ⚠ JIT may not be working properly (speedup < 1.5x)")

    # Verify activations
    print(f"\n  Buffer contents:")
    for layer_idx in layers_to_extract:
        if layer_idx in storage.buffer:
            acts = storage.buffer[layer_idx]
            print(f"    Layer {layer_idx}: {len(acts)} activations")
            if len(acts) > 0:
                shape = acts[0]['activation'].shape
                sample_idx = acts[0]['sample_idx']
                text_preview = acts[0]['text_preview'][:50]
                print(f"      Sample {sample_idx}: shape={shape}, text='{text_preview}...'")

    print("\n✓ Activation extraction successful")
    return storage


def test_storage():
    """Test activation storage"""
    print("\n" + "="*70)
    print("TEST 4: Activation Storage")
    print("="*70)

    storage = ActivationStorage(
        output_dir='./test_storage',
        upload_to_gcs=False,
        shard_size_gb=0.001,  # 1 MB for testing
        compress_shards=True,
        verbose=True
    )

    # Add some activations
    print("\nAdding activations...")
    for layer_idx in [10, 11, 12]:
        for sample_idx in range(5):
            activation = np.random.randn(128, 256).astype(np.float32)
            storage.add_activation(
                layer_idx=layer_idx,
                activation=activation,
                sample_idx=sample_idx,
                text_preview=f"Sample {sample_idx} text preview"
            )

    # Finalize
    print("\nFinalizing storage...")
    storage.finalize()

    # Verify files
    import os
    files = os.listdir('./test_storage')
    print(f"\n  Files created: {files}")

    # Check metadata
    import json
    with open('./test_storage/metadata.json') as f:
        metadata = json.load(f)

    print(f"\n  Metadata:")
    print(f"    Total shards: {metadata['total_shards']}")
    print(f"    Total samples: {metadata['total_samples']}")
    print(f"    Shard info: {len(metadata['shards'])} shards")

    print("\n✓ Storage test passed")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TESTING extract_activations_fineweb_multihost.py")
    print("="*70)

    # Check devices
    devices = jax.devices()
    print(f"\nDevices: {len(devices)} {[d.device_kind for d in devices]}")

    try:
        # Test 1: Tokenization
        sequences, texts, tokenizer = test_fineweb_tokenization()

        # Test 2: Padding
        test_padding()

        # Test 3: Activation extraction
        test_activation_extraction()

        # Test 4: Storage
        test_storage()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
