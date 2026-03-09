#!/usr/bin/env python3
"""Analyze token budget breakdown for ARC prompts."""

import json
import subprocess
import numpy as np
from transformers import AutoTokenizer
from arc24.encoders import create_grid_encoder
from arc24.prompting import create_prompts_from_task, get_prompt_templates

tokenizer = AutoTokenizer.from_pretrained("KathirKs/qwen-2.5-0.5b", trust_remote_code=True)
grid_encoder = create_grid_encoder("GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))")

print("Loading dataset from GCS...")
proc = subprocess.run(
    ["gcloud", "storage", "cat", "gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl"],
    capture_output=True, text=False, timeout=120
)
lines = proc.stdout.decode().strip().split("\n")

system_prompt, prompt_template, answer_template = get_prompt_templates("output-from-examples-v1")
sys_tokens = len(tokenizer.encode(system_prompt))
print(f"System prompt: {sys_tokens} tokens -- '{system_prompt}'")
print()

# Analyze 100 tasks
sample_indices = np.linspace(0, len(lines) - 1, 100, dtype=int)
all_grid_tokens = []
all_overhead = []
all_total = []
all_num_train = []

for idx in sample_indices:
    task = json.loads(lines[idx])
    tid = task.get("task_id", "unknown")

    try:
        prompts = create_prompts_from_task(
            task, grid_encoder=grid_encoder, tokenizer=tokenizer,
            is_train_prompt=False, prompt_version="output-from-examples-v1"
        )
        total = len(tokenizer.encode(prompts[0]))
    except Exception:
        continue

    # Count grid tokens
    train_data = task.get("train", [])
    train_samples = [{key: grid_encoder.to_text(grid) for key, grid in sample.items()} for sample in train_data]
    test_input = grid_encoder.to_text(task["test"][0]["input"])

    grid_tok = 0
    for sample in train_samples:
        grid_tok += len(tokenizer.encode(sample["input"]))
        grid_tok += len(tokenizer.encode(sample["output"]))
    grid_tok += len(tokenizer.encode(test_input))

    overhead = total - grid_tok
    all_grid_tokens.append(grid_tok)
    all_overhead.append(overhead)
    all_total.append(total)
    all_num_train.append(len(train_data))

all_grid_tokens = np.array(all_grid_tokens)
all_overhead = np.array(all_overhead)
all_total = np.array(all_total)
all_num_train = np.array(all_num_train)

print(f"Analyzed {len(all_total)} tasks")
print(f"{'='*60}")
print(f"{'Metric':<30} {'Mean':>8} {'Median':>8} {'P90':>8}")
print(f"{'='*60}")
print(f"{'Total tokens':<30} {all_total.mean():>8.0f} {np.median(all_total):>8.0f} {np.percentile(all_total,90):>8.0f}")
print(f"{'Grid data tokens':<30} {all_grid_tokens.mean():>8.0f} {np.median(all_grid_tokens):>8.0f} {np.percentile(all_grid_tokens,90):>8.0f}")
print(f"{'Overhead tokens':<30} {all_overhead.mean():>8.0f} {np.median(all_overhead):>8.0f} {np.percentile(all_overhead,90):>8.0f}")
print(f"{'Training examples':<30} {all_num_train.mean():>8.1f} {np.median(all_num_train):>8.0f} {np.percentile(all_num_train,90):>8.0f}")
print(f"{'='*60}")
print(f"{'Grid % of total':<30} {(all_grid_tokens/all_total).mean()*100:>7.1f}%")
print(f"{'Overhead % of total':<30} {(all_overhead/all_total).mean()*100:>7.1f}%")
print()

# How many fit in 2048?
print(f"Fit in 2048: {(all_total <= 2048).sum()}/{len(all_total)} ({(all_total <= 2048).mean()*100:.1f}%)")
print(f"Grid data alone fits in 2048: {(all_grid_tokens <= 2048).sum()}/{len(all_total)} ({(all_grid_tokens <= 2048).mean()*100:.1f}%)")
print(f"Grid data alone fits in 1800: {(all_grid_tokens <= 1800).sum()}/{len(all_total)} ({(all_grid_tokens <= 1800).mean()*100:.1f}%)")
print()

# If we strip ALL text and just keep grids
print("If we strip instructions and keep only grid data:")
print(f"  Under 1024: {(all_grid_tokens <= 1024).sum()} ({(all_grid_tokens <= 1024).mean()*100:.1f}%)")
print(f"  Under 2048: {(all_grid_tokens <= 2048).sum()} ({(all_grid_tokens <= 2048).mean()*100:.1f}%)")
print(f"  Over 2048:  {(all_grid_tokens > 2048).sum()} ({(all_grid_tokens > 2048).mean()*100:.1f}%)")
