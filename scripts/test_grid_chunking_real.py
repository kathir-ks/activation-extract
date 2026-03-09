#!/usr/bin/env python3
"""Test grid chunking with real tokenizer and grid encoder on a small sample."""

import json
import subprocess
import numpy as np
from transformers import AutoTokenizer
from arc24.encoders import create_grid_encoder
from core.grid_chunking import create_grid_chunks_from_dataset

tokenizer = AutoTokenizer.from_pretrained("KathirKs/qwen-2.5-0.5b", trust_remote_code=True)
grid_encoder = create_grid_encoder("GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))")

print("Loading dataset from local stream...")
with open("data/streams/stream_000.jsonl") as f:
    lines = f.readlines()

# Take a small sample
sample_size = min(20, len(lines))
sample_indices = np.linspace(0, len(lines) - 1, sample_size, dtype=int)
tasks = {}
for idx in sample_indices:
    task = json.loads(lines[idx])
    tid = task.get("task_id", f"task_{idx}")
    tasks[tid] = {"train": task["train"], "test": task["test"]}

print(f"\nLoaded {len(tasks)} tasks")

# Compare: standard prompt tokenization vs grid chunking
from arc24.prompting import create_prompts_from_task

prompt_lengths = []
for tid, task in tasks.items():
    try:
        prompts = create_prompts_from_task(
            task, grid_encoder=grid_encoder, tokenizer=tokenizer,
            is_train_prompt=False, prompt_version="output-from-examples-v1"
        )
        for p in prompts:
            prompt_lengths.append(len(tokenizer.encode(p)))
    except Exception:
        pass

prompt_lengths = np.array(prompt_lengths)
print(f"\nStandard prompt pipeline:")
print(f"  Prompts: {len(prompt_lengths)}")
print(f"  Mean tokens: {prompt_lengths.mean():.0f}")
print(f"  Median tokens: {np.median(prompt_lengths):.0f}")
print(f"  Fit in 2048: {(prompt_lengths <= 2048).sum()}/{len(prompt_lengths)} ({(prompt_lengths <= 2048).mean()*100:.1f}%)")

# Grid chunking pipeline
for chunk_size in [2048]:
    chunks, chunk_meta, stream_meta = create_grid_chunks_from_dataset(
        tasks=tasks,
        grid_encoder=grid_encoder,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        predictions_per_task=1,
        random_seed=42,
        verbose=True,
    )

    total_tokens = sum(m.num_tokens for m in chunk_meta)
    padding_tokens = sum(chunk_size - m.num_tokens for m in chunk_meta)
    print(f"\n  Summary for chunk_size={chunk_size}:")
    print(f"    Chunks: {len(chunks)}")
    print(f"    Total useful tokens: {total_tokens:,}")
    print(f"    Padding tokens: {padding_tokens}")
    print(f"    Token utilization: {total_tokens / (len(chunks) * chunk_size) * 100:.2f}%")
