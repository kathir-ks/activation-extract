"""Measure token generation performance and output quality"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_fixed import QwenModelFixed
from generate_parallel import generate_parallel
import time
import json

def count_tokens(tokenizer, text):
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def generate_sequential(model, params, tokenizer, prompts, max_tokens, temperature):
    """Sequential generation for comparison"""
    outputs = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
        input_ids = jnp.array(input_ids)

        # Simple greedy generation
        current_ids = input_ids
        for _ in range(max_tokens):I
            logits = model.apply(params, current_ids)
            next_token = jnp.argmax(logits[0, -1])
            current_ids = jnp.concatenate([current_ids, next_token.reshape(1, 1)], axis=1)

            # Stop at EOS
            if next_token == tokenizer.eos_token_id:
                break

        output_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        # Remove prompt from output
        output_text = output_text[len(prompt):]
        outputs.append(output_text)

    return outputs

def measure_performance(model, params, tokenizer, prompts, max_tokens, temperature, mode="parallel"):
    """Measure tokens per second for generation"""

    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {max_tokens}")
    print(f"{'='*60}\n")

    # Count input tokens
    input_token_counts = [count_tokens(tokenizer, p) for p in prompts]
    total_input_tokens = sum(input_token_counts)
    print(f"Input tokens: {input_token_counts} (total: {total_input_tokens})")

    # Generate with timing
    start_time = time.time()

    if mode == "parallel":
        outputs = generate_parallel(
            model, params, tokenizer, prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False
        )
    else:  # sequential
        outputs = generate_sequential(
            model, params, tokenizer, prompts,
            max_tokens=max_tokens,
            temperature=temperature
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Count output tokens
    output_token_counts = [count_tokens(tokenizer, o) for o in outputs]
    total_output_tokens = sum(output_token_counts)

    # Calculate metrics
    tokens_per_second = total_output_tokens / elapsed_time
    tokens_per_second_per_prompt = tokens_per_second / len(prompts)

    print(f"\n{'='*60}")
    print(f"PERFORMANCE RESULTS - {mode.upper()}")
    print(f"{'='*60}")
    print(f"Output tokens: {output_token_counts} (total: {total_output_tokens})")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Tokens/second: {tokens_per_second:.2f}")
    print(f"Tokens/second/prompt: {tokens_per_second_per_prompt:.2f}")
    print(f"Time per token: {elapsed_time/total_output_tokens*1000:.2f}ms")

    # Output quality analysis
    print(f"\n{'='*60}")
    print(f"OUTPUT QUALITY - {mode.upper()}")
    print(f"{'='*60}")

    results = []
    for i, (prompt, output, out_tokens) in enumerate(zip(prompts, outputs, output_token_counts)):
        print(f"\n--- Prompt {i+1} ({input_token_counts[i]} input tokens) ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"\n--- Output ({out_tokens} tokens) ---")
        print(output)
        print(f"\n--- Analysis ---")

        # Check if output is complete
        is_complete = out_tokens < max_tokens or tokenizer.eos_token in output
        is_empty = len(output.strip()) < 10
        has_grid = "```" in output or any(c.isdigit() for c in output)

        print(f"Complete: {is_complete}")
        print(f"Empty/truncated: {is_empty}")
        print(f"Contains grid/numbers: {has_grid}")

        results.append({
            "prompt": prompt,
            "output": output,
            "input_tokens": input_token_counts[i],
            "output_tokens": out_tokens,
            "complete": is_complete,
            "empty": is_empty,
            "has_grid": has_grid
        })

    return {
        "mode": mode,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_time": elapsed_time,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_prompt": tokens_per_second_per_prompt,
        "time_per_token_ms": elapsed_time/total_output_tokens*1000,
        "results": results
    }

def main():
    # Initialize TPU
    print("Initializing TPU...")
    try:
        jax.devices("tpu")
        print(f"Found {len(jax.devices('tpu'))} TPU cores")
    except:
        print("No TPU found, using default backend:", jax.default_backend())

    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    print("\nLoading model...")
    model_path = "KathirKs/qwen-2.5-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load HF model for weight conversion
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # Initialize JAX model
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

    jax_model = QwenModelFixed(config)

    # Initialize parameters
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    params = jax_model.init(rng, dummy_input)

    # Convert weights
    print("Converting weights...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    # Test prompts - using simple ones first
    test_prompts = [
        "Write a haiku about technology:",
        "Explain quantum computing in one sentence:",
        "What is the capital of France?",
        "Count from 1 to 10:"
    ]

    # Test with different token limits
    test_configs = [
        {"max_tokens": 20, "description": "Short (20 tokens)"},
        {"max_tokens": 50, "description": "Medium (50 tokens)"},
        {"max_tokens": 100, "description": "Long (100 tokens)"},
    ]

    all_results = {}

    for config in test_configs:
        max_tokens = config["max_tokens"]
        desc = config["description"]

        print(f"\n\n{'#'*80}")
        print(f"# TEST: {desc}")
        print(f"{'#'*80}\n")

        # Test parallel mode
        try:
            parallel_results = measure_performance(
                jax_model, params, tokenizer, test_prompts,
                max_tokens=max_tokens,
                temperature=0.0,
                mode="parallel"
            )
            all_results[f"parallel_{max_tokens}"] = parallel_results
        except Exception as e:
            print(f"\n❌ PARALLEL MODE FAILED: {e}")
            all_results[f"parallel_{max_tokens}"] = {"error": str(e)}

        # Test sequential mode for comparison
        try:
            sequential_results = measure_performance(
                jax_model, params, tokenizer, test_prompts,
                max_tokens=max_tokens,
                temperature=0.0,
                mode="sequential"
            )
            all_results[f"sequential_{max_tokens}"] = sequential_results
        except Exception as e:
            print(f"\n❌ SEQUENTIAL MODE FAILED: {e}")
            all_results[f"sequential_{max_tokens}"] = {"error": str(e)}

        # Compare if both succeeded
        if f"parallel_{max_tokens}" in all_results and f"sequential_{max_tokens}" in all_results:
            if "error" not in all_results[f"parallel_{max_tokens}"] and "error" not in all_results[f"sequential_{max_tokens}"]:
                par = all_results[f"parallel_{max_tokens}"]
                seq = all_results[f"sequential_{max_tokens}"]

                speedup = seq["elapsed_time"] / par["elapsed_time"]
                throughput_improvement = par["tokens_per_second"] / seq["tokens_per_second"]

                print(f"\n{'='*60}")
                print(f"COMPARISON - {desc}")
                print(f"{'='*60}")
                print(f"Sequential time: {seq['elapsed_time']:.2f}s")
                print(f"Parallel time: {par['elapsed_time']:.2f}s")
                print(f"Speedup: {speedup:.2f}x")
                print(f"Sequential tokens/sec: {seq['tokens_per_second']:.2f}")
                print(f"Parallel tokens/sec: {par['tokens_per_second']:.2f}")
                print(f"Throughput improvement: {throughput_improvement:.2f}x")

    # Save results
    output_file = "performance_results.json"
    with open(output_file, "w") as f:
        # Remove prompt/output text for cleaner summary
        summary = {}
        for key, value in all_results.items():
            if "error" not in value:
                summary[key] = {k: v for k, v in value.items() if k != "results"}
            else:
                summary[key] = value
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
