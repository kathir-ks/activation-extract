#!/usr/bin/env python3
"""
Create Dataset Streams for Parallel Workers

Splits a HuggingFace dataset into N independent JSONL streams for parallel processing.
Each stream will be processed by one TPU worker.

OPTIMIZED: Loads dataset once and creates all streams in a single pass.

Usage:
    # Split into 32 streams
    python create_dataset_streams.py \
        --num_streams 32 \
        --output_dir ./data/streams

    # Split into 64 streams with custom dataset
    python create_dataset_streams.py \
        --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --num_streams 64 \
        --output_dir ./data/streams \
        --max_samples 200000
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


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


def create_streams(
    dataset_name: str,
    column_name: str,
    num_streams: int,
    output_dir: str,
    max_samples: int = None,
    verbose: bool = True
):
    """
    Create N dataset streams by splitting the dataset into equal parts.

    OPTIMIZED: Loads dataset once and writes to all stream files in a single pass.

    Args:
        dataset_name: HuggingFace dataset name
        column_name: Column containing task pairs
        num_streams: Number of streams to create
        output_dir: Output directory for stream files
        max_samples: Maximum total samples (None = use all)
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*70)
        print("CREATING DATASET STREAMS FOR PARALLEL WORKERS")
        print("="*70)
        print(f"  Dataset: {dataset_name}")
        print(f"  Column: {column_name}")
        print(f"  Number of streams: {num_streams}")
        print(f"  Output directory: {output_dir}")
        print(f"  Max samples: {max_samples if max_samples else 'unlimited'}")
        print("="*70)
        print("\nLoading dataset (streaming mode)...")

    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Open all stream files at once
    stream_files = []
    stream_counters = [0] * num_streams  # Track samples per stream

    for i in range(num_streams):
        filepath = output_path / f"stream_{i:03d}.jsonl"
        stream_files.append(open(filepath, 'w'))

    if verbose:
        print(f"Opened {num_streams} stream files for writing")
        print("Processing dataset in single pass...")

    total_processed = 0
    total_invalid = 0

    try:
        # Single pass through dataset
        pbar = tqdm(dataset, desc="Processing", disable=not verbose)
        for sample in pbar:
            # Check max_samples limit
            if max_samples and total_processed >= max_samples:
                break

            # Determine which stream this sample belongs to (round-robin)
            stream_id = total_processed % num_streams

            # Convert to ARC format
            task_id = f"task_{total_processed:08x}"
            arc_task = convert_sample_to_arc_format(sample, column_name, task_id)

            if arc_task:
                stream_files[stream_id].write(json.dumps(arc_task) + '\n')
                stream_counters[stream_id] += 1
                total_processed += 1

                if verbose and total_processed % 10000 == 0:
                    pbar.set_postfix({
                        'processed': total_processed,
                        'per_stream': f"~{total_processed // num_streams}"
                    })
            else:
                total_invalid += 1

    finally:
        # Close all files
        for f in stream_files:
            f.close()

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
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
