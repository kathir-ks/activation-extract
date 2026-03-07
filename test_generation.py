#!/usr/bin/env python3
"""
Quick sanity check: greedy text generation using the JAX Qwen2 implementation.

Loads the model in bfloat16, runs autoregressive generation on a few English
prompts, and prints the output. No extraction pipeline involved.

Usage:
    python test_generation.py --model_path KathirKs/qwen-2.5-0.5b --max_new_tokens 50
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qwen2_jax import QwenConfig, QwenModel, convert_hf_to_jax_weights


def greedy_generate(params, model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="np")  # [1, seq_len]
    input_ids = jnp.array(input_ids, dtype=jnp.int32)

    generated = []
    kv_caches = None
    position_offset = 0

    # Prefill: process the entire prompt at once
    logits, kv_caches = model.apply(params, input_ids, kv_caches=kv_caches,
                                    position_offset=position_offset)
    next_token = int(jnp.argmax(logits[0, -1, :]))
    generated.append(next_token)
    position_offset += input_ids.shape[1]

    # Decode: one token at a time with KV cache
    for _ in range(max_new_tokens - 1):
        token_input = jnp.array([[next_token]], dtype=jnp.int32)
        logits, kv_caches = model.apply(params, token_input, kv_caches=kv_caches,
                                        position_offset=position_offset)
        next_token = int(jnp.argmax(logits[0, -1, :]))
        generated.append(next_token)
        position_offset += 1

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="KathirKs/qwen-2.5-0.5b")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "In machine learning, a transformer is",
        "The quick brown fox jumps over the",
    ]

    print(f"Device: {jax.devices()[0]}")
    print(f"Model:  {args.model_path}")
    print(f"Dtype:  bfloat16\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model weights (bfloat16)...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config = QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        dtype=jnp.bfloat16,
    )

    converted = convert_hf_to_jax_weights(hf_model, config)
    params = {"params": converted}
    del hf_model
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    model = QwenModel(config)

    print("=" * 60)
    for prompt in prompts:
        t0 = time.time()
        output = greedy_generate(params, model, tokenizer, prompt, args.max_new_tokens)
        elapsed = time.time() - t0
        print(f"Prompt : {prompt}")
        print(f"Output : {output}")
        print(f"Time   : {elapsed:.2f}s")
        print("-" * 60)


if __name__ == "__main__":
    main()
