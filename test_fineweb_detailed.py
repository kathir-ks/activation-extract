"""
Detailed test to verify activations are extracted correctly
"""

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks
from extract_activations_fineweb_multihost import (
    load_dataset_shard,
    ActivationStorage,
    extract_activations_sharded,
    pad_sequences
)

def main():
    print("="*70)
    print("DETAILED ACTIVATION EXTRACTION TEST")
    print("="*70)

    # Setup
    model_path = "KathirKs/qwen-2.5-0.5b"
    layers_to_extract = [10, 11]

    # Config
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
    print(f"\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded")

    # Load model
    print(f"\n2. Loading model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, trust_remote_code=True
    )
    jax_model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    del hf_model
    print(f"   ✓ Model loaded")

    # Load dataset
    print(f"\n3. Loading FineWeb-Edu dataset (2 samples)...")
    dataset = load_dataset_shard(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
        machine_id=0,
        total_machines=1,
        max_samples=2,
        verbose=False
    )

    # Tokenize
    sequences = []
    texts = []
    for i, sample in enumerate(dataset):
        text = sample['text']
        texts.append(text)

        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128  # Short for fast test
        )
        sequence = inputs['input_ids'][0]
        sequences.append(sequence)
        print(f"   Sample {i}: {len(sequence)} tokens, text: {text[:50]}...")

    # Pad sequences
    print(f"\n4. Padding sequences to fixed length 128...")
    padded = pad_sequences(sequences, pad_token_id=tokenizer.pad_token_id or 0, fixed_length=128)
    input_ids = jnp.array(padded)
    print(f"   ✓ Padded shape: {input_ids.shape}")

    # Extract activations
    print(f"\n5. Extracting activations...")
    activations = extract_activations_sharded(jax_model, params, input_ids)

    print(f"   ✓ Activations extracted:")
    for layer_key, layer_act in activations.items():
        print(f"     {layer_key}: shape={layer_act.shape}")

    # Store activations
    print(f"\n6. Storing activations...")
    storage = ActivationStorage(
        output_dir='./test_detailed',
        upload_to_gcs=False,
        shard_size_gb=0.001,
        compress_shards=False,
        verbose=False
    )

    # Process each sample
    for i in range(len(sequences)):
        print(f"\n   Sample {i}:")
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                # Get activation for this sample
                layer_act = activations[layer_key][i]  # [seq_len, hidden_dim]
                layer_act_np = np.array(layer_act)

                print(f"     Layer {layer_idx}: shape={layer_act_np.shape}, "
                      f"mean={layer_act_np.mean():.4f}, std={layer_act_np.std():.4f}")

                # Store
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=layer_act_np,
                    sample_idx=i,
                    text_preview=texts[i]
                )

    # Check buffer
    print(f"\n7. Checking storage buffer...")
    print(f"   Buffer size: {storage.buffer_size_bytes / (1024*1024):.2f} MB")
    print(f"   Total samples: {storage.total_samples}")
    for layer_idx, acts in storage.buffer.items():
        print(f"   Layer {layer_idx}: {len(acts)} activations")
        for i, act_data in enumerate(acts):
            print(f"     Sample {act_data['sample_idx']}: "
                  f"shape={act_data['shape']}, "
                  f"text='{act_data['text_preview'][:50]}...'")

    # Finalize
    print(f"\n8. Finalizing storage...")
    storage.finalize()

    # Verify saved files
    import os
    import pickle
    files = [f for f in os.listdir('./test_detailed') if f.endswith('.pkl')]
    print(f"   ✓ Saved files: {files}")

    if files:
        # Load and verify
        with open(f'./test_detailed/{files[0]}', 'rb') as f:
            shard_data = pickle.load(f)

        print(f"\n9. Verifying saved shard...")
        for layer_idx, acts in shard_data.items():
            print(f"   Layer {layer_idx}: {len(acts)} activations")
            for act_data in acts:
                print(f"     Sample {act_data['sample_idx']}: "
                      f"shape={act_data['shape']}, "
                      f"activation mean={act_data['activation'].mean():.4f}")

    print("\n" + "="*70)
    print("TEST COMPLETE ✓")
    print("="*70)


if __name__ == '__main__':
    main()
