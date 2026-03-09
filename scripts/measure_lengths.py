#!/usr/bin/env python3
"""Measure token length distribution across the dataset."""

import json
import numpy as np
import subprocess
from transformers import AutoTokenizer
from arc24.encoders import create_grid_encoder
from arc24.prompting import create_prompts_from_task
from arc24.data_augmentation import apply_data_augmentation, get_random_color_map, set_random_seed

SEP = "=" * 50

tokenizer = AutoTokenizer.from_pretrained("KathirKs/qwen-2.5-0.5b", trust_remote_code=True)
grid_encoder = create_grid_encoder("GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))")
set_random_seed(42)

# Load dataset from GCS
print("Loading dataset from GCS...")
proc = subprocess.run(
    ["gcloud", "storage", "cat", "gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl"],
    capture_output=True, text=False, timeout=120
)
lines = proc.stdout.decode().strip().split("\n")
print(f"Total tasks in dataset: {len(lines)}")

# Sample 1000 tasks evenly across dataset
sample_indices = np.linspace(0, len(lines) - 1, 1000, dtype=int)
tasks = {}
for i in sample_indices:
    data = json.loads(lines[i])
    task_id = data.get("task_id", data.get("id", f"task_{i}"))
    tasks[task_id] = data

# Create 1 prompt per task (no augmentation, measure raw lengths)
lengths = []
for task_id, task in tasks.items():
    try:
        prompts = create_prompts_from_task(
            task, grid_encoder=grid_encoder, tokenizer=tokenizer,
            is_train_prompt=False, prompt_version="output-from-examples-v1"
        )
        for p in prompts:
            tokens = tokenizer.encode(p)
            lengths.append(len(tokens))
    except Exception as e:
        pass

lengths = np.array(lengths)
print(f"\nSampled {len(lengths)} prompts from 1000 tasks")
print(SEP)
print(f"Min:    {lengths.min():>6}")
print(f"P10:    {int(np.percentile(lengths, 10)):>6}")
print(f"P25:    {int(np.percentile(lengths, 25)):>6}")
print(f"Median: {int(np.median(lengths)):>6}")
print(f"Mean:   {int(lengths.mean()):>6}")
print(f"P75:    {int(np.percentile(lengths, 75)):>6}")
print(f"P90:    {int(np.percentile(lengths, 90)):>6}")
print(f"P95:    {int(np.percentile(lengths, 95)):>6}")
print(f"P99:    {int(np.percentile(lengths, 99)):>6}")
print(f"Max:    {lengths.max():>6}")
print(SEP)
print(f"Under 512:   {(lengths <= 512).sum():>5} ({(lengths <= 512).mean()*100:>5.1f}%)")
print(f"Under 1024:  {(lengths <= 1024).sum():>5} ({(lengths <= 1024).mean()*100:>5.1f}%)")
print(f"Under 2048:  {(lengths <= 2048).sum():>5} ({(lengths <= 2048).mean()*100:>5.1f}%)")
print(f"Under 4096:  {(lengths <= 4096).sum():>5} ({(lengths <= 4096).mean()*100:>5.1f}%)")
print(f"Under 8192:  {(lengths <= 8192).sum():>5} ({(lengths <= 8192).mean()*100:>5.1f}%)")
print(f"Under 16384: {(lengths <= 16384).sum():>5} ({(lengths <= 16384).mean()*100:>5.1f}%)")
print(f"Under 32768: {(lengths <= 32768).sum():>5} ({(lengths <= 32768).mean()*100:>5.1f}%)")
print(SEP)
print(f"Exceed 2048:  {(lengths > 2048).sum():>5} ({(lengths > 2048).mean()*100:>5.1f}%)")
print(f"Exceed 4096:  {(lengths > 4096).sum():>5} ({(lengths > 4096).mean()*100:>5.1f}%)")
print(f"Exceed 8192:  {(lengths > 8192).sum():>5} ({(lengths > 8192).mean()*100:>5.1f}%)")

# Histogram
buckets = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
hist, _ = np.histogram(lengths, bins=buckets)
print(f"\nLength distribution (buckets):")
for i in range(len(hist)):
    bar = "#" * int(hist[i] / max(hist) * 40)
    print(f"  {buckets[i]:>5}-{buckets[i+1]:>5}: {hist[i]:>5} {bar}")
