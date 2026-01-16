"""
Verify Model Implementation with Text Generation
Runs model on sample prompts and generates text to verify correctness
"""

import json
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import sys

# Import our model
sys.path.append('.')
from qwen2_jax_with_hooks import create_model_with_hooks
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights


def generate_text_simple(model, params, tokenizer, prompt, max_new_tokens=25):
    """Simple greedy generation for verification"""

    # Tokenize
    tokens = tokenizer(prompt, return_tensors='np')
    input_ids = jnp.array(tokens['input_ids'])

    generated = input_ids[0].tolist()

    # Initialize KV cache
    kv_caches = None

    # Prefill phase
    logits, kv_caches = model.apply(
        {'params': params},
        input_ids,
        kv_caches=None,
        position_offset=0,
        return_activations=False
    )

    # Get last token logits
    next_token_logits = logits[0, -1, :]
    next_token = jnp.argmax(next_token_logits).item()
    generated.append(next_token)

    # Decode phase
    for _ in range(max_new_tokens - 1):
        next_token_input = jnp.array([[next_token]])

        logits, kv_caches = model.apply(
            {'params': params},
            next_token_input,
            kv_caches=kv_caches,
            position_offset=len(generated) - 1,
            return_activations=False
        )

        next_token_logits = logits[0, -1, :]
        next_token = jnp.argmax(next_token_logits).item()
        generated.append(next_token)

        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=False)


def load_sample_prompts(dataset_path, tokenizer, max_samples=10):
    """Load sample prompts from dataset"""
    prompts = []

    # Import ARC formatting
    from arc24.prompting import create_prompts_from_task
    from arc24.encoders import GridCodeBlockEncoder, MinimalGridEncoder

    grid_encoder = GridCodeBlockEncoder(MinimalGridEncoder())

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            task = json.loads(line.strip())

            # Create prompts for this task
            task_prompts = create_prompts_from_task(
                task,
                grid_encoder,
                tokenizer,
                is_train_prompt=False,  # Inference mode
                prompt_version='output-from-examples-v1'
            )

            # Use first test case
            if task_prompts:
                expected_output = ''
                try:
                    if 'test' in task and task['test'] and 'output' in task['test'][0]:
                        expected_output = grid_encoder.to_text(task['test'][0]['output'])
                except (KeyError, IndexError):
                    pass

                prompts.append({
                    'task_id': task.get('task_id', f'task_{i}'),
                    'prompt': task_prompts[0],
                    'expected_output': expected_output
                })

    return prompts


def verify_model_generation(dataset_path, model_path, output_file, max_samples=10, max_new_tokens=25):
    """Run model generation and write results to file"""

    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=hf_config.tie_word_embeddings,
    )

    # Create model
    model = create_model_with_hooks(config, layers_to_extract=None)

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    import torch
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Load in float32 for JAX compatibility
    )

    # Load weights
    print("Converting weights to JAX...")
    params = convert_hf_to_jax_weights(hf_model, config)
    del hf_model  # Free memory

    # JIT compile the model
    print("JIT compiling model (this may take a moment)...")
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    _ = model.apply({'params': params}, dummy_input, return_activations=False)
    print("✓ Model compiled successfully\n")

    # Load sample prompts
    print(f"Loading {max_samples} sample prompts from {dataset_path}...")
    prompts = load_sample_prompts(dataset_path, tokenizer, max_samples)

    # Generate and write results
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MODEL GENERATION VERIFICATION\n")
        f.write("=" * 100 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size\n")
        f.write(f"Vocab size: {config.vocab_size}\n")
        f.write(f"Max new tokens: {max_new_tokens}\n")
        f.write("=" * 100 + "\n\n")

        for i, prompt_data in enumerate(prompts):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"SAMPLE {i+1}/{len(prompts)}: {prompt_data['task_id']}\n")
            f.write(f"{'=' * 100}\n\n")

            # Truncate prompt for display (keep first and last parts)
            prompt = prompt_data['prompt']
            if len(prompt) > 1000:
                f.write("INPUT PROMPT (truncated for display):\n")
                f.write("-" * 100 + "\n")
                f.write(prompt[:500] + "\n")
                f.write(f"\n... ({len(prompt) - 1000} characters omitted) ...\n\n")
                f.write(prompt[-500:] + "\n")
            else:
                f.write("INPUT PROMPT:\n")
                f.write("-" * 100 + "\n")
                f.write(prompt + "\n")

            f.write("\n" + "-" * 100 + "\n")
            f.write(f"GENERATING (max {max_new_tokens} tokens)...\n")
            f.write("-" * 100 + "\n\n")

            try:
                # Generate
                generated_text = generate_text_simple(
                    model, params, tokenizer, prompt, max_new_tokens
                )

                # Show only the new tokens (after the prompt)
                # Find where the original prompt ends
                prompt_tokens = tokenizer(prompt, return_tensors='np')['input_ids'][0]
                generated_tokens = tokenizer(generated_text, return_tensors='np')['input_ids'][0]

                # Decode just the new part
                new_tokens = generated_tokens[len(prompt_tokens):]
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

                f.write("GENERATED TEXT (new tokens only):\n")
                f.write(new_text + "\n\n")

                f.write("FULL GENERATED TEXT (prompt + new):\n")
                f.write(generated_text + "\n\n")

                if prompt_data['expected_output']:
                    f.write("EXPECTED OUTPUT (for reference):\n")
                    f.write(prompt_data['expected_output'] + "\n\n")

                f.write("✓ Generation completed successfully\n")

            except Exception as e:
                f.write(f"✗ Generation failed: {str(e)}\n\n")
                import traceback
                f.write(traceback.format_exc() + "\n")

    print(f"\n✓ Model generation verification written to: {output_file}")
    print(f"  Please review the file to verify:")
    print(f"  - Model generates coherent text")
    print(f"  - Output format matches expectations")
    print(f"  - No crashes or numerical errors")
    print(f"  - Generated grids follow ARC format")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify model implementation with generation")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-0.5B", help="Model path")
    parser.add_argument('--output_file', type=str, default="model_generation_verification.txt", help="Output file")
    parser.add_argument('--max_samples', type=int, default=10, help="Number of samples to test")
    parser.add_argument('--max_new_tokens', type=int, default=25, help="Max tokens to generate per sample")

    args = parser.parse_args()

    verify_model_generation(
        args.dataset_path,
        args.model_path,
        args.output_file,
        args.max_samples,
        args.max_new_tokens
    )
