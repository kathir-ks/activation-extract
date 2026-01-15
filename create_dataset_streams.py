#!/usr/bin/env python3
"""
Create Dataset Streams for Parallel Workers

Splits a HuggingFace dataset into N independent JSONL streams for parallel processing.
Each stream will be processed by one TPU worker.

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
import os
from pathlib import Path
from convert_hf_to_arc_format import convert_hf_dataset_to_arc_format


def create_streams(
    dataset_name: str,
    column_name: str,
    num_streams: int,
    output_dir: str,
    max_samples: int = None,
    verbose: bool = True
):
    """
    Create N dataset streams by splitting the dataset into equal parts

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

    # Calculate samples per stream
    if max_samples:
        samples_per_stream = max_samples // num_streams
        remainder = max_samples % num_streams
    else:
        # If no max_samples, we'll process all data
        # HF streaming will handle this automatically
        samples_per_stream = None
        remainder = 0

    # Create each stream
    for stream_id in range(num_streams):
        if samples_per_stream:
            start_idx = stream_id * samples_per_stream
            # Distribute remainder across first few streams
            if stream_id < remainder:
                start_idx += stream_id
                end_idx = start_idx + samples_per_stream + 1
            else:
                start_idx += remainder
                end_idx = start_idx + samples_per_stream
        else:
            # No limits, but still need to shard
            # This will require streaming with skip/stride
            start_idx = stream_id
            end_idx = None

        output_file = output_path / f"stream_{stream_id:03d}.jsonl"

        if verbose:
            print(f"\n{'='*70}")
            print(f"Creating stream {stream_id}/{num_streams-1}")
            print(f"{'='*70}")
            if samples_per_stream:
                print(f"  Range: {start_idx} to {end_idx-1}")
                print(f"  Samples: {end_idx - start_idx}")
            print(f"  Output: {output_file}")
            print(f"{'='*70}")

        # Convert this slice of the dataset
        convert_hf_dataset_to_arc_format(
            dataset_name=dataset_name,
            column_name=column_name,
            output_filename=str(output_file),
            max_tasks=None,  # Don't limit per stream, use start/end
            start_index=start_idx,
            end_index=end_idx,
            verbose=verbose
        )

    if verbose:
        print(f"\n{'='*70}")
        print("✅ ALL STREAMS CREATED SUCCESSFULLY")
        print("="*70)
        print(f"  Total streams: {num_streams}")
        print(f"  Location: {output_dir}")
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
