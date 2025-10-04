"""
Example: How to use Layer Activation Extraction with Qwen Model

This demonstrates:
1. Creating a model with activation hooks
2. Running inference with extraction
3. Saving activations for SAE training
"""

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
import pickle
import os

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import (
    create_model_with_hooks,
    generate_with_layer_activations,
    extract_specific_layer_positions,
    DEFAULT_SAE_LAYERS
)
from transformers import AutoModelForCausalLM


def example_basic_extraction():
    """Basic example: Extract activations from specific layers"""

    print("=" * 70)
    print("EXAMPLE 1: Basic Layer Activation Extraction")
    print("=" * 70)

    # 1. Create configuration
    config = QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=24,
        max_position_embeddings=2048
    )

    # 2. Create model with hooks for layers 10-23
    print("\nCreating model with activation hooks for layers:", DEFAULT_SAE_LAYERS)
    print(f"  (Extracting {len(DEFAULT_SAE_LAYERS)} layers)")
    model = create_model_with_hooks(config, layers_to_extract=DEFAULT_SAE_LAYERS)

    # 3. Initialize model parameters
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")

    # 4. Create sample input
    input_text = "The answer to 2 + 2 is"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    input_ids = jnp.array([tokenizer.encode(input_text)])

    print(f"\nInput: '{input_text}'")
    print(f"Input shape: {input_ids.shape}")

    # 5. Forward pass with activation extraction
    print("\nRunning forward pass...")
    logits, activations = model.apply(params, input_ids, return_activations=True)

    print(f"\nLogits shape: {logits.shape}")
    print(f"\nExtracted {len(activations)} layer activations:")
    for layer_name, acts in activations.items():
        print(f"  {layer_name}: {acts.shape}")

    # 6. Extract last token activations for SAE training
    print("\nExtracting last token activations for SAE training:")
    for layer_idx in DEFAULT_SAE_LAYERS:
        layer_name = f'layer_{layer_idx}'
        if layer_name in activations:
            last_token_acts = extract_specific_layer_positions(
                activations, layer_name, token_positions=None  # None = last token
            )
            print(f"  Layer {layer_idx} last token: {last_token_acts.shape}")

    return activations


def example_generation_with_extraction():
    """Example: Generate text while extracting activations"""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Generation with Activation Extraction")
    print("=" * 70)

    # Setup
    config = QwenConfig(num_hidden_layers=24)
    model = create_model_with_hooks(config, layers_to_extract=[12, 23])

    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    # Input
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    prompt = "1 + 1 ="
    input_ids = jnp.array([tokenizer.encode(prompt)])

    print(f"Generating from: '{prompt}'")
    print("Extracting activations at each generation step...")

    # Generate with activation extraction
    generated_ids, final_activations, activations_per_step = generate_with_layer_activations(
        model,
        params,
        input_ids,
        max_tokens=10,
        layers_to_extract=[12, 23],
        return_all_step_activations=True
    )

    # Decode output
    generated_text = tokenizer.decode(generated_ids[0])
    print(f"\nGenerated: '{generated_text}'")

    # Show activations per step
    print(f"\nCaptured activations for {len(activations_per_step)} generation steps:")
    for step_data in activations_per_step[:3]:  # Show first 3 steps
        print(f"  Step {step_data['step']}, seq_len={step_data['seq_len']}:")
        for layer_name in step_data['activations']:
            print(f"    {layer_name}: {step_data['activations'][layer_name].shape}")

    return activations_per_step


def example_save_activations_for_sae():
    """Example: Save activations in format for SAE training"""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Saving Activations for SAE Training")
    print("=" * 70)

    # Setup
    config = QwenConfig(num_hidden_layers=24, hidden_size=896)
    layers_to_extract = [6, 12, 18, 23]
    model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)

    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
    params = model.init(key, dummy_input)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    # Simulate processing multiple samples
    sample_texts = [
        "The capital of France is",
        "2 + 2 equals",
        "Python is a programming",
        "Machine learning is"
    ]

    activations_dataset = {
        f'layer_{i}': [] for i in layers_to_extract
    }

    print(f"\nProcessing {len(sample_texts)} samples...")
    for text in sample_texts:
        input_ids = jnp.array([tokenizer.encode(text)])
        logits, activations = model.apply(params, input_ids, return_activations=True)

        # Extract last token activation from each layer
        for layer_idx in layers_to_extract:
            layer_name = f'layer_{layer_idx}'
            if layer_name in activations:
                last_token = activations[layer_name][:, -1, :]  # [1, hidden_dim]
                activations_dataset[layer_name].append(np.array(last_token[0]))

    # Save activations
    os.makedirs('./activations', exist_ok=True)
    for layer_name, acts_list in activations_dataset.items():
        activations_array = np.stack(acts_list)  # [n_samples, hidden_dim]
        filepath = f'./activations/{layer_name}_activations.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump({
                'activations': activations_array,
                'layer': layer_name,
                'n_samples': len(acts_list),
                'hidden_dim': activations_array.shape[1]
            }, f)

        print(f"Saved {layer_name}: {activations_array.shape} to {filepath}")

    print("\nActivations saved! Ready for SAE training.")


def example_load_hf_weights():
    """Example: Load actual HuggingFace weights and extract activations"""

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Using Real HuggingFace Weights")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-0.5B"

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load HF model
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Convert to JAX
    print("Converting to JAX...")
    config = QwenConfig()
    jax_params = convert_hf_to_jax_weights(hf_model)

    # Create model with hooks
    layers_to_extract = [6, 12, 18, 23]
    model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)

    # Test
    prompt = "The meaning of life is"
    input_ids = jnp.array([tokenizer.encode(prompt)])

    print(f"\nPrompt: '{prompt}'")
    print("Running inference with activation extraction...")

    logits, activations = model.apply({'params': jax_params}, input_ids, return_activations=True)

    print(f"\nExtracted activations from {len(activations)} layers:")
    for layer_name, acts in activations.items():
        print(f"  {layer_name}: {acts.shape}")

    # Get predicted token
    next_token_id = int(jnp.argmax(logits[0, -1, :]))
    next_token = tokenizer.decode([next_token_id])
    print(f"\nNext predicted token: '{next_token}'")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("QWEN MODEL ACTIVATION EXTRACTION EXAMPLES")
    print("=" * 70)

    # Run examples
    try:
        example_basic_extraction()
        example_generation_with_extraction()
        example_save_activations_for_sae()

        # This requires actual model download
        # example_load_hf_weights()

    except Exception as e:
        print(f"\nError in examples: {e}")
        print("Note: Some examples require downloading the actual Qwen model")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
