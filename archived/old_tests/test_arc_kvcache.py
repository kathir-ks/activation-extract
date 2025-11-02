"""
Quick test for arc_inference_jax.py with fixed KV cache
"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_fixed import QwenModelFixed
from arc_inference_jax import generate_tokens_jax, generate_outputs_with_batches


def test_arc_kvcache():
    """Test ARC inference with fixed KV cache"""
    print("="*60)
    print("Testing ARC Inference with Fixed KV Cache")
    print("="*60)

    # Load model
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\n1. Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    config = QwenConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        intermediate_size=hf_model.config.intermediate_size,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        use_fixed_cache=True
    )

    print("2. Initializing JAX model with fixed cache...")
    model = QwenModelFixed(config)

    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input)

    print("3. Converting HuggingFace weights to JAX...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    # Test 1: Simple generation
    print("\n" + "="*60)
    print("Test 1: Simple Token Generation")
    print("="*60)

    test_prompt = "The answer is"
    inputs = tokenizer(test_prompt, return_tensors="np")
    input_ids = jnp.array(inputs['input_ids'])

    print(f"Prompt: '{test_prompt}'")
    print(f"Input shape: {input_ids.shape}")

    start_time = time.time()
    generated_ids = generate_tokens_jax(
        model, params, input_ids,
        max_tokens=10,
        tokenizer=tokenizer
    )
    gen_time = time.time() - start_time

    output_text = tokenizer.decode(generated_ids[0])
    print(f"\n✓ Generation successful!")
    print(f"Output: '{output_text}'")
    print(f"Time: {gen_time:.2f}s")

    # Test 2: Batch generation interface
    print("\n" + "="*60)
    print("Test 2: Batch Generation Interface")
    print("="*60)

    test_prompts = [
        "Question: What is 2+2? Answer:",
        "Complete the pattern: 1, 2, 3,",
        "The capital of France is"
    ]

    print(f"Testing with {len(test_prompts)} prompts")

    start_time = time.time()
    outputs = generate_outputs_with_batches(
        model, params, tokenizer,
        prompts=test_prompts,
        batch_size=1,
        max_output_tokens=10
    )
    batch_time = time.time() - start_time

    print(f"\n✓ Batch generation successful!")
    print(f"Generated {len(outputs)} outputs in {batch_time:.2f}s")
    print(f"Average: {batch_time/len(outputs):.2f}s per prompt\n")

    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        generated_text = output.outputs[0].text
        print(f"{i+1}. Prompt: '{prompt[:40]}'")
        print(f"   Output: '{generated_text[:50]}'")

    print("\n" + "="*60)
    print("✓ All ARC inference tests passed!")
    print("="*60)

    return True


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    test_arc_kvcache()
