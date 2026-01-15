#!/usr/bin/env python3
"""
Create sharded dataset for distributed processing

Splits a large JSONL dataset into multiple shards, with each shard containing
multiple files of ~100MB each. Each shard gets a metadata file for tracking usage.

Usage:
    python create_sharded_dataset.py \
        --input_file dataset.jsonl \
        --output_dir gs://bucket/sharded_dataset \
        --num_shards 8 \
        --chunk_size_mb 100
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import gcsfs
from tqdm import tqdm


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    if file_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        return fs.size(file_path) / (1024 * 1024)
    else:
        return os.path.getsize(file_path) / (1024 * 1024)


def count_lines(file_path: str) -> int:
    """Count lines in a file (local or GCS)"""
    if file_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(file_path, 'r') as f:
            return sum(1 for _ in f)
    else:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)


def create_sharded_dataset(
    input_file: str,
    output_dir: str,
    num_shards: int = 8,
    chunk_size_mb: float = 100.0,
    verbose: bool = True
):
    """
    Create sharded dataset with multiple chunks per shard

    Args:
        input_file: Input JSONL file (local or gs://)
        output_dir: Output directory (local or gs://)
        num_shards: Number of shards to create
        chunk_size_mb: Target size for each chunk file in MB
        verbose: Print progress
    """
    if verbose:
        print("="*70)
        print("Creating Sharded Dataset")
        print("="*70)
        print(f"  Input: {input_file}")
        print(f"  Output: {output_dir}")
        print(f"  Number of shards: {num_shards}")
        print(f"  Chunk size: {chunk_size_mb} MB")
        print("="*70)

    # Count total tasks
    if verbose:
        print("\n[1/4] Counting tasks...")
    total_tasks = count_lines(input_file)
    tasks_per_shard = total_tasks // num_shards

    if verbose:
        print(f"  Total tasks: {total_tasks:,}")
        print(f"  Tasks per shard: {tasks_per_shard:,}")

    # Setup file system
    is_gcs = output_dir.startswith("gs://")
    if is_gcs:
        fs = gcsfs.GCSFileSystem()
        # Create output directory
        if not fs.exists(output_dir):
            fs.makedirs(output_dir)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open input file
    if input_file.startswith("gs://"):
        fs_in = gcsfs.GCSFileSystem()
        input_handle = fs_in.open(input_file, 'r')
    else:
        input_handle = open(input_file, 'r')

    if verbose:
        print("\n[2/4] Creating shards...")

    try:
        # Process each shard
        task_idx = 0

        for shard_id in range(num_shards):
            shard_dir = f"{output_dir}/shard_{shard_id:03d}"

            # Create shard directory
            if is_gcs:
                fs.makedirs(shard_dir, exist_ok=True)
            else:
                Path(shard_dir).mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"\n  Processing shard {shard_id}/{num_shards}...")

            # Track chunks in this shard
            chunk_id = 0
            chunk_tasks = []
            chunk_size_bytes = 0
            chunk_target_bytes = int(chunk_size_mb * 1024 * 1024)

            shard_metadata = {
                "shard_id": shard_id,
                "total_shards": num_shards,
                "status": "available",  # available, in_use, completed
                "total_tasks": 0,
                "chunks": [],
                "assigned_to": None,
                "started_at": None,
                "completed_at": None
            }

            # Read tasks for this shard
            shard_task_count = 0

            with tqdm(total=tasks_per_shard, desc=f"    Shard {shard_id}",
                     disable=not verbose, leave=False) as pbar:

                while shard_task_count < tasks_per_shard and task_idx < total_tasks:
                    line = input_handle.readline()
                    if not line:
                        break

                    task = json.loads(line.strip())
                    task_bytes = len(line.encode('utf-8'))

                    # Check if we need to start a new chunk
                    if chunk_size_bytes + task_bytes > chunk_target_bytes and len(chunk_tasks) > 0:
                        # Write current chunk
                        chunk_file = f"{shard_dir}/chunk_{chunk_id:04d}.jsonl"
                        write_chunk(chunk_file, chunk_tasks, is_gcs, fs if is_gcs else None)

                        # Update metadata
                        shard_metadata["chunks"].append({
                            "chunk_id": chunk_id,
                            "file": f"chunk_{chunk_id:04d}.jsonl",
                            "num_tasks": len(chunk_tasks),
                            "size_mb": chunk_size_bytes / (1024 * 1024)
                        })

                        # Reset for next chunk
                        chunk_id += 1
                        chunk_tasks = []
                        chunk_size_bytes = 0

                    # Add task to current chunk
                    chunk_tasks.append(task)
                    chunk_size_bytes += task_bytes
                    shard_task_count += 1
                    task_idx += 1
                    pbar.update(1)

            # Write remaining tasks in last chunk
            if chunk_tasks:
                chunk_file = f"{shard_dir}/chunk_{chunk_id:04d}.jsonl"
                write_chunk(chunk_file, chunk_tasks, is_gcs, fs if is_gcs else None)

                shard_metadata["chunks"].append({
                    "chunk_id": chunk_id,
                    "file": f"chunk_{chunk_id:04d}.jsonl",
                    "num_tasks": len(chunk_tasks),
                    "size_mb": chunk_size_bytes / (1024 * 1024)
                })

            # Update shard metadata
            shard_metadata["total_tasks"] = shard_task_count
            shard_metadata["num_chunks"] = len(shard_metadata["chunks"])

            # Write shard metadata
            metadata_file = f"{shard_dir}/metadata.json"
            write_metadata(metadata_file, shard_metadata, is_gcs, fs if is_gcs else None)

            if verbose:
                print(f"    ✓ Created {len(shard_metadata['chunks'])} chunks "
                      f"({shard_task_count:,} tasks)")

        # Create master metadata
        if verbose:
            print("\n[3/4] Creating master metadata...")

        master_metadata = {
            "total_tasks": total_tasks,
            "num_shards": num_shards,
            "chunk_size_mb": chunk_size_mb,
            "shards": [
                {
                    "shard_id": i,
                    "directory": f"shard_{i:03d}",
                    "status": "available"
                }
                for i in range(num_shards)
            ]
        }

        master_file = f"{output_dir}/master_metadata.json"
        write_metadata(master_file, master_metadata, is_gcs, fs if is_gcs else None)

        if verbose:
            print("\n[4/4] Summary")
            print("="*70)
            print(f"  ✓ Created {num_shards} shards")
            print(f"  ✓ Total tasks: {total_tasks:,}")
            print(f"  ✓ Output: {output_dir}")
            print(f"\nShard structure:")
            print(f"  {output_dir}/")
            print(f"    ├── master_metadata.json")
            for i in range(min(3, num_shards)):
                print(f"    ├── shard_{i:03d}/")
                print(f"    │   ├── metadata.json")
                print(f"    │   ├── chunk_0000.jsonl")
                print(f"    │   └── chunk_XXXX.jsonl")
            if num_shards > 3:
                print(f"    └── ... ({num_shards - 3} more shards)")
            print("="*70)

    finally:
        input_handle.close()


def write_chunk(file_path: str, tasks: List[Dict], is_gcs: bool, fs=None):
    """Write chunk file"""
    if is_gcs:
        with fs.open(file_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
    else:
        with open(file_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')


def write_metadata(file_path: str, metadata: Dict, is_gcs: bool, fs=None):
    """Write metadata file"""
    if is_gcs:
        with fs.open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Create sharded dataset for distributed processing"
    )

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Input JSONL file (local or gs://)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Output directory (local or gs://)"
    )
    parser.add_argument(
        '--num_shards',
        type=int,
        default=8,
        help="Number of shards to create (default: 8)"
    )
    parser.add_argument(
        '--chunk_size_mb',
        type=float,
        default=100.0,
        help="Target chunk size in MB (default: 100)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print progress"
    )

    args = parser.parse_args()

    create_sharded_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        chunk_size_mb=args.chunk_size_mb,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
