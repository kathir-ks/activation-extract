#!/usr/bin/env python3
"""
Stream Status — View progress of multi-stream extraction workloads.

Usage:
    # View status from GCS manifest
    python stream_status.py gs://my-bucket/activations/stream_manifest.json

    # View from local manifest
    python stream_status.py ./stream_manifest.json

    # Create a new manifest
    python stream_status.py gs://my-bucket/activations/stream_manifest.json \
        --create --total_streams 32 --dataset_dir gs://my-bucket/datasets
"""

import argparse
import sys

from core.stream_manager import StreamManager


def main():
    parser = argparse.ArgumentParser(
        description="View or create stream extraction manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('manifest_path', type=str,
                        help="Path to manifest file (GCS or local)")
    parser.add_argument('--create', action='store_true',
                        help="Create a new manifest")
    parser.add_argument('--total_streams', type=int,
                        help="Total number of streams (for --create)")
    parser.add_argument('--dataset_dir', type=str,
                        help="Dataset directory/GCS path (for --create)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing manifest (for --create)")

    args = parser.parse_args()

    sm = StreamManager(args.manifest_path)

    if args.create:
        if not args.total_streams or not args.dataset_dir:
            print("Error: --total_streams and --dataset_dir required for --create")
            sys.exit(1)
        sm.create_manifest(
            total_streams=args.total_streams,
            dataset_dir=args.dataset_dir,
            overwrite=args.overwrite,
        )
        return

    # Show status
    status = sm.get_status_summary()
    if 'error' in status:
        print(f"Error: {status['error']}")
        sys.exit(1)

    print("=" * 60)
    print("STREAM EXTRACTION STATUS")
    print("=" * 60)
    print(f"  Dataset: {status['dataset_dir']}")
    print(f"  Manifest: {args.manifest_path}")
    print()
    print(f"  Total:       {status['total']}")
    print(f"  ✅ Completed: {status['completed']}")
    print(f"  🔄 In Progress: {status['in_progress']}")
    print(f"  ⏳ Pending:    {status['pending']}")
    print(f"  Progress:    {status['pct_complete']}%")
    print()

    if status['pods_active']:
        print("  Active pods:")
        for pod, stream_ids in status['pods_active'].items():
            print(f"    {pod}: streams {stream_ids}")
    print("=" * 60)


if __name__ == "__main__":
    main()
