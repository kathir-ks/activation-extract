#!/usr/bin/env python3
"""
Dataset Conversion CLI

Converts HuggingFace datasets to ARC-AGI format for activation extraction.

Example usage:
    # Convert barc0/200k_HEAVY dataset
    python convert_dataset.py \
        --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
        --output_file data/arc_tasks.jsonl \
        --max_tasks 10000

    # Create sharded dataset for distributed processing
    python convert_dataset.py \
        --input_file data/arc_tasks.jsonl \
        --output_dir data/sharded \
        --create_shards \
        --tasks_per_shard 1000
"""

import argparse
from data import convert_hf_dataset_to_arc_format, create_sharded_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert datasets to ARC format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert from HuggingFace
    hf_parser = subparsers.add_parser("from_hf", help="Convert from HuggingFace dataset")
    hf_parser.add_argument(
        "--dataset_name", type=str,
        default="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        help="HuggingFace dataset name"
    )
    hf_parser.add_argument(
        "--column_name", type=str, default="examples",
        help="Column containing task pairs"
    )
    hf_parser.add_argument(
        "--output_file", type=str, required=True,
        help="Output JSONL file path"
    )
    hf_parser.add_argument(
        "--max_tasks", type=int,
        help="Maximum number of tasks to convert"
    )
    hf_parser.add_argument(
        "--max_train_examples", type=int,
        help="Maximum training examples per task"
    )
    hf_parser.add_argument(
        "--start_index", type=int, default=0,
        help="Start index in dataset (for sharding)"
    )
    hf_parser.add_argument(
        "--end_index", type=int,
        help="End index in dataset (for sharding)"
    )
    hf_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    # Create sharded dataset
    shard_parser = subparsers.add_parser("shard", help="Create sharded dataset")
    shard_parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input JSONL file path"
    )
    shard_parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for sharded dataset"
    )
    shard_parser.add_argument(
        "--tasks_per_shard", type=int, default=1000,
        help="Number of tasks per shard"
    )
    shard_parser.add_argument(
        "--tasks_per_chunk", type=int, default=100,
        help="Number of tasks per chunk within a shard"
    )
    shard_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == "from_hf":
        print("="*70)
        print("Converting HuggingFace Dataset to ARC Format")
        print("="*70)
        
        task_count = convert_hf_dataset_to_arc_format(
            dataset_name=args.dataset_name,
            column_name=args.column_name,
            output_filename=args.output_file,
            max_tasks=args.max_tasks,
            max_train_examples=args.max_train_examples,
            start_index=args.start_index,
            end_index=args.end_index,
            verbose=not args.quiet
        )
        
        print(f"\nConverted {task_count} tasks to {args.output_file}")
        
    elif args.command == "shard":
        print("="*70)
        print("Creating Sharded Dataset")
        print("="*70)
        
        metadata = create_sharded_dataset(
            input_file=args.input_file,
            output_dir=args.output_dir,
            tasks_per_shard=args.tasks_per_shard,
            tasks_per_chunk=args.tasks_per_chunk,
            verbose=not args.quiet
        )
        
        print(f"\nCreated {metadata['num_shards']} shards in {args.output_dir}")
        
    else:
        print("Please specify a command: from_hf or shard")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
