#!/usr/bin/env python3
"""
Unified Activation Extraction CLI

Extracts activations from Qwen models on single or multi-host TPU setups.
Supports ARC and JSONL dataset formats.

Example usage:
    # Single-host extraction on ARC dataset
    python extract_activations.py \
        --model_path Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_path data/arc_tasks.jsonl \
        --output_dir ./activations

    # Multi-host extraction with GCS upload
    python extract_activations.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --dataset_path data/arc_tasks.jsonl \
        --output_dir ./activations \
        --use_multihost \
        --upload_to_gcs \
        --gcs_bucket my-bucket
"""

import argparse
from extraction import ExtractionConfig, run_extraction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract activations from Qwen models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model_path", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model path or local path"
    )
    model_group.add_argument(
        "--layers", type=int, nargs="+",
        help="Layer indices to extract (default: use DEFAULT_SAE_LAYERS)"
    )
    model_group.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length"
    )
    model_group.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for extraction"
    )
    
    # Dataset settings
    data_group = parser.add_argument_group("Dataset Settings")
    data_group.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to dataset file"
    )
    data_group.add_argument(
        "--dataset_type", type=str, default="arc",
        choices=["arc", "jsonl"],
        help="Dataset format type"
    )
    data_group.add_argument(
        "--max_samples", type=int,
        help="Maximum number of samples to process"
    )
    
    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output_dir", type=str, default="./activations",
        help="Output directory for activations"
    )
    output_group.add_argument(
        "--shard_size_gb", type=float, default=1.0,
        help="Target shard size in GB"
    )
    output_group.add_argument(
        "--no_compress", action="store_true",
        help="Disable shard compression"
    )
    
    # GCS settings
    gcs_group = parser.add_argument_group("GCS Settings")
    gcs_group.add_argument(
        "--upload_to_gcs", action="store_true",
        help="Upload shards to Google Cloud Storage"
    )
    gcs_group.add_argument(
        "--gcs_bucket", type=str,
        help="GCS bucket name"
    )
    gcs_group.add_argument(
        "--gcs_prefix", type=str, default="activations",
        help="GCS path prefix"
    )
    gcs_group.add_argument(
        "--delete_local", action="store_true",
        help="Delete local files after GCS upload"
    )
    
    # Multi-host settings
    multi_group = parser.add_argument_group("Multi-host Settings")
    multi_group.add_argument(
        "--use_multihost", action="store_true",
        help="Enable multi-host TPU mode"
    )
    multi_group.add_argument(
        "--machine_id", type=int, default=0,
        help="This machine's ID (0-indexed)"
    )
    multi_group.add_argument(
        "--total_machines", type=int, default=1,
        help="Total number of machines"
    )
    
    # Other settings
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = ExtractionConfig(
        # Model
        model_path=args.model_path,
        layers_to_extract=args.layers,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        
        # Dataset
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        
        # Output
        output_dir=args.output_dir,
        shard_size_gb=args.shard_size_gb,
        compress_shards=not args.no_compress,
        
        # GCS
        upload_to_gcs=args.upload_to_gcs,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        delete_local_after_upload=args.delete_local,
        
        # Multi-host
        use_multihost=args.use_multihost,
        machine_id=args.machine_id,
        total_machines=args.total_machines,
        
        # Other
        verbose=not args.quiet,
        random_seed=args.seed,
    )
    
    # Run extraction
    results = run_extraction(config)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"  Total processed: {results.get('total_processed', 0)}")
    print(f"  Total shards: {results.get('total_shards', 0)}")
    print(f"  Output directory: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
