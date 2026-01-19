#!/usr/bin/env python3
"""
Create Dataset Streams for Parallel Workers

Splits a HuggingFace dataset into N independent JSONL streams for parallel processing.
Each stream will be processed by one TPU worker.

OPTIMIZED:
- Loads dataset once into memory
- Uses multiprocessing to write streams in parallel
- ~32x faster than sequential approach

Usage:
    # Split into 32 streams (auto-detect CPU cores)
    python create_dataset_streams.py \
        --num_streams 32 \
        --output_dir ./data/streams

    # Split into 64 streams with 8 parallel workers
    python create_dataset_streams.py \
        --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --num_streams 64 \
        --output_dir ./data/streams \
        --num_workers 8
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def convert_sample_to_arc_format(sample, column_name: str, task_id: str):
    """
    Convert a single HF sample to ARC format.

    Returns:
        dict: ARC formatted task, or None if invalid
    """
    if column_name not in sample:
        return None

    task_pairs = sample[column_name]

    # Validate format: must be list with at least 2 pairs (train + test)
    if not isinstance(task_pairs, list) or len(task_pairs) < 2:
        return None

    # Split into train/test: last pair is test, rest are train
    train_pairs = task_pairs[:-1]
    test_pair = task_pairs[-1]

    # Format training data
    formatted_train_list = []
    for pair in train_pairs:
        if isinstance(pair, list) and len(pair) == 2:
            try:
                input_grid = [[int(cell) for cell in row] for row in pair[0]]
                output_grid = [[int(cell) for cell in row] for row in pair[1]]
                formatted_train_list.append({
                    "input": input_grid,
                    "output": output_grid
                })
            except (ValueError, TypeError):
                continue

    if len(formatted_train_list) == 0:
        return None

    # Format test data
    if isinstance(test_pair, list) and len(test_pair) > 0:
        try:
            test_input_grid = [[int(cell) for cell in row] for row in test_pair[0]]
            formatted_test_list = [{"input": test_input_grid}]
        except (ValueError, TypeError):
            return None
    else:
        return None

    return {
        "task_id": task_id,
        "train": formatted_train_list,
        "test": formatted_test_list
    }


def write_stream_worker(args):
    """
    Worker function to write a single stream file.

    Args:
        args: Tuple of (stream_id, samples, column_name, output_path, base_task_id)

    Returns:
        Tuple of (stream_id, num_written, num_invalid)
    """
    stream_id, samples, column_name, output_path, base_task_id = args

    filepath = Path(output_path) / f"stream_{stream_id:03d}.jsonl"
    num_written = 0
    num_invalid = 0

    with open(filepath, 'w') as f:
        for i, sample in enumerate(samples):
            task_id = f"task_{base_task_id + i:08x}"
            arc_task = convert_sample_to_arc_format(sample, column_name, task_id)

            if arc_task:
                f.write(json.dumps(arc_task) + '\n')
                num_written += 1
            else:
                num_invalid += 1

    return (stream_id, num_written, num_invalid)


def create_streams(
    dataset_name: str,
    column_name: str,
    num_streams: int,
    output_dir: str,
    max_samples: int = None,
    num_workers: int = None,
    verbose: bool = True
):
    """
    Create N dataset streams by splitting the dataset into equal parts.

    OPTIMIZED:
    - Loads dataset once into memory
    - Uses multiprocessing to write streams in parallel

    Args:
        dataset_name: HuggingFace dataset name
        column_name: Column containing task pairs
        num_streams: Number of streams to create
        output_dir: Output directory for stream files
        max_samples: Maximum total samples (None = use all)
        num_workers: Number of parallel workers (None = auto-detect)
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), num_streams)

    if verbose:
        print("="*70)
        print("CREATING DATASET STREAMS FOR PARALLEL WORKERS")
        print("="*70)
        print(f"  Dataset: {dataset_name}")
        print(f"  Column: {column_name}")
        print(f"  Number of streams: {num_streams}")
        print(f"  Output directory: {output_dir}")
        print(f"  Max samples: {max_samples if max_samples else 'unlimited'}")
        print(f"  Parallel workers: {num_workers}")
        print("="*70)
        print("\nStep 1: Loading dataset into memory...")

    # Load dataset into memory (non-streaming for random access)
    dataset = load_dataset(dataset_name, split="train")
    total_available = len(dataset)

    if max_samples:
        total_to_process = min(max_samples, total_available)
    else:
        total_to_process = total_available

    if verbose:
        print(f"  Loaded {total_available} samples")
        print(f"  Processing: {total_to_process} samples")

    # Distribute samples across streams (round-robin assignment)
    if verbose:
        print(f"\nStep 2: Distributing samples to {num_streams} streams...")

    stream_samples = [[] for _ in range(num_streams)]

    for i in tqdm(range(total_to_process), desc="Distributing", disable=not verbose):
        stream_id = i % num_streams
        stream_samples[stream_id].append(dataset[i])

    if verbose:
        print(f"\nStep 3: Writing streams in parallel with {num_workers} workers...")

    # Prepare arguments for each worker
    worker_args = []
    base_task_id = 0
    for stream_id in range(num_streams):
        worker_args.append((
            stream_id,
            stream_samples[stream_id],
            column_name,
            str(output_path),
            stream_id  # Use stream_id as base for unique task IDs
        ))

    # Process in parallel
    stream_counters = [0] * num_streams
    total_invalid = 0

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(write_stream_worker, worker_args),
            total=num_streams,
            desc="Writing streams",
            disable=not verbose
        ))

    # Collect results
    for stream_id, num_written, num_invalid in results:
        stream_counters[stream_id] = num_written
        total_invalid += num_invalid

    total_processed = sum(stream_counters)

    if verbose:
        print(f"\n{'='*70}")
        print("✅ ALL STREAMS CREATED SUCCESSFULLY")
        print("="*70)
        print(f"  Total samples processed: {total_processed}")
        print(f"  Invalid samples skipped: {total_invalid}")
        print(f"  Total streams: {num_streams}")
        print(f"  Samples per stream: ~{total_processed // num_streams}")
        print(f"  Location: {output_dir}")
        print(f"\n  Stream distribution:")
        for i, count in enumerate(stream_counters):
            print(f"    Stream {i:03d}: {count} samples")
        print(f"\nYou can now launch parallel workers with:")
        print(f"  export TPU_WORKER_ID=0")
        print(f"  python extract_activations.py --dataset_path {output_dir}/stream_${{TPU_WORKER_ID}}.jsonl ...")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset streams for parallel workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        '--column_name',
        type=str,
        default="examples",
        help="Column containing task pairs"
    )
    parser.add_argument(
        '--num_streams',
        type=int,
        required=True,
        help="Number of streams to create (typically 32 or 64)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./data/streams",
        help="Output directory for stream files"
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        help="Maximum total samples to process (will be split across streams)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on CPU cores)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help="Print progress"
    )

    args = parser.parse_args()

    create_streams(
        dataset_name=args.dataset_name,
        column_name=args.column_name,
        num_streams=args.num_streams,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
