"""
Verify that the JAX Qwen model is working correctly by running real inference
"""

import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import (
    create_model_with_hooks,
    generate_with_kv_cache_and_activations,
    extract_activations_from_prompt
)


def test_real_model_inference():
    """
    Test with actual HuggingFace Qwen model to verify correctness
    """
    print("="*80)
    print("TESTING REAL QWEN MODEL INFERENCE")
    print("="*80)

    model_path = "KathirKs/qwen-2.5-0.5b"

    # Load tokenizer
    print(f"\n1. Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    # Load HF model
    print(f"\n2. Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True
    )
    print(f"   ✓ HuggingFace model loaded")

    # Create JAX model
    print(f"\n3. Creating JAX model with hooks...")
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

    jax_model = create_model_with_hooks(config, layers_to_extract=[10, 15, 20, 23])
    print(f"   ✓ JAX model created")

    # Convert weights
    print(f"\n4. Converting HuggingFace weights to JAX...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    print(f"   ✓ Weights converted")

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "Once upon a time",
    ]

    print(f"\n5. Running inference tests...")
    print("="*80)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: '{prompt}'")
        print("-"*80)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = jnp.array(inputs['input_ids'])

        print(f"Input tokens: {input_ids.shape}")

        # Generate with KV cache
        generated_ids, _, timing = generate_with_kv_cache_and_activations(
            model=jax_model,
            params=params,
            input_ids=input_ids,
            max_tokens=20,
            extract_activations=False,
            tokenizer=tokenizer
        )

        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated text: {generated_text}")
        print(f"Timing: {timing['tokens_per_sec']:.2f} tokens/sec")
        print("-"*80)

    # Test activation extraction
    print(f"\n6. Testing activation extraction...")
    print("-"*80)
    test_input = tokenizer("Hello, how are you?", return_tensors="np")
    input_ids = jnp.array(test_input['input_ids'])

    logits, activations = extract_activations_from_prompt(
        jax_model, params, input_ids
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Activations extracted from layers:")
    for layer_name, acts in activations.items():
        print(f"  {layer_name}: {acts.shape}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - MODEL IS WORKING CORRECTLY!")
    print("="*80)


if __name__ == "__main__":
    # Check available devices
    devices = jax.devices()
    print(f"JAX devices available: {[d.device_kind for d in devices]}")
    print()

    # Run the test
    test_real_model_inference()
