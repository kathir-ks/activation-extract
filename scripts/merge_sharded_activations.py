#!/usr/bin/env python3
"""
Merge split activation shards from paired hosts into full hidden_dim shards.

Due to FSDP mesh topology on v5litepod-64, each host pair (host_00+host_01,
host_02+host_03, etc.) stores complementary 448-dim halves of the same samples.
This script concatenates them into full 896-dim vectors.

Usage (on a single machine):
    python3 scripts/merge_sharded_activations.py \
        --src_gcs gs://arc-data-europe-west4/activations/layer19_gridchunk_50k_v5litepod-64 \
        --dst_gcs gs://arc-data-europe-west4/activations/layer19_merged_50k \
        --layer_index 19 \
        --num_pairs 8

Multi-worker (distribute across TPU workers):
    python3 scripts/merge_sharded_activations.py \
        --src_gcs gs://... --dst_gcs gs://... \
        --worker_id 0 --num_workers 16
"""

import argparse
import gzip
import os
import pickle
import subprocess
import sys
import time
import numpy as np
from pathlib import Path


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gcs_ls(gcs_path, pattern="shard_*.pkl.gz"):
    """List files in GCS matching pattern."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", os.path.join(gcs_path, pattern)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return sorted([l.strip() for l in result.stdout.strip().split("\n") if l.strip()])


def gcs_cp(src, dst):
    """Copy file to/from GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "cp", src, dst],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("gcloud storage cp failed: %s" % result.stderr)


def gcs_exists(gcs_path):
    """Check if a GCS object exists."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True
    )
    return result.returncode == 0 and gcs_path.split("/")[-1] in result.stdout


def merge_shard_pair(src_gcs, dst_gcs, pair_idx, shard_name, layer_index, tmp_dir="/tmp/merge"):
    """Merge one shard from a host pair into a full hidden_dim shard."""
    host_a = "host_%02d" % (pair_idx * 2)
    host_b = "host_%02d" % (pair_idx * 2 + 1)

    # Skip if already merged
    out_prefix = "pair_%02d" % pair_idx
    out_path = os.path.join(dst_gcs, out_prefix, shard_name)
    if gcs_exists(out_path):
        return -1, None  # Signal: skipped

    path_a = os.path.join(src_gcs, host_a, shard_name)
    path_b = os.path.join(src_gcs, host_b, shard_name)

    local_a = os.path.join(tmp_dir, "%s_%s" % (host_a, shard_name))
    local_b = os.path.join(tmp_dir, "%s_%s" % (host_b, shard_name))

    os.makedirs(tmp_dir, exist_ok=True)

    # Download both halves
    gcs_cp(path_a, local_a)
    gcs_cp(path_b, local_b)

    # Load
    with gzip.open(local_a, "rb") as f:
        data_a = pickle.load(f)
    with gzip.open(local_b, "rb") as f:
        data_b = pickle.load(f)

    os.remove(local_a)
    os.remove(local_b)

    # Merge activations
    acts_a = data_a[layer_index]
    acts_b = data_b[layer_index]

    if len(acts_a) != len(acts_b):
        log("WARNING: pair %d %s: mismatched sample counts (%d vs %d)" %
            (pair_idx, shard_name, len(acts_a), len(acts_b)))
        min_len = min(len(acts_a), len(acts_b))
        acts_a = acts_a[:min_len]
        acts_b = acts_b[:min_len]

    merged = []
    for a, b in zip(acts_a, acts_b):
        if a["sample_idx"] != b["sample_idx"]:
            log("WARNING: sample_idx mismatch: %d vs %d" % (a["sample_idx"], b["sample_idx"]))
            continue
        combined = np.concatenate([a["activation"], b["activation"]], axis=-1)
        merged.append({
            "sample_idx": a["sample_idx"],
            "activation": combined,
            "shape": combined.shape,
            "text_preview": a["text_preview"],
        })

    # Write merged shard with fast compression
    # Output goes to pair_XX/ subdirectory to keep provenance
    out_prefix = "pair_%02d" % pair_idx
    out_path = os.path.join(dst_gcs, out_prefix, shard_name)
    local_out = os.path.join(tmp_dir, "merged_%s" % shard_name)

    with gzip.open(local_out, "wb", compresslevel=1) as f:
        pickle.dump({layer_index: merged}, f)

    gcs_cp(local_out, out_path)
    os.remove(local_out)

    return len(merged), merged[0]["activation"].shape if merged else None


def main():
    parser = argparse.ArgumentParser(description="Merge split activation shards")
    parser.add_argument("--src_gcs", required=True, help="Source GCS path with host_XX/ subdirs")
    parser.add_argument("--dst_gcs", required=True, help="Destination GCS path for merged shards")
    parser.add_argument("--layer_index", type=int, default=19)
    parser.add_argument("--num_pairs", type=int, default=8, help="Number of host pairs (default: 8)")
    parser.add_argument("--worker_id", type=int, default=0, help="This worker's ID (for distributed)")
    parser.add_argument("--num_workers", type=int, default=1, help="Total workers (for distributed)")
    parser.add_argument("--tmp_dir", default="/tmp/merge", help="Temp directory for downloads")
    parser.add_argument("--max_shards", type=int, default=0, help="Max shards per pair (0=all, for testing)")
    args = parser.parse_args()

    log("Merge worker %d/%d starting" % (args.worker_id, args.num_workers))
    log("Source: %s" % args.src_gcs)
    log("Destination: %s" % args.dst_gcs)

    # Enumerate all shard-pair tasks: (pair_idx, shard_name)
    all_tasks = []
    for pair_idx in range(args.num_pairs):
        host_a = "host_%02d" % (pair_idx * 2)
        shards = gcs_ls(os.path.join(args.src_gcs, host_a))
        shard_names = sorted([os.path.basename(s) for s in shards])
        if args.max_shards > 0:
            shard_names = shard_names[:args.max_shards]
        for shard_name in shard_names:
            all_tasks.append((pair_idx, shard_name))

    log("Total shard-pairs to merge: %d" % len(all_tasks))

    # Distribute tasks across workers (round-robin)
    my_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_workers == args.worker_id]
    log("This worker handles %d shard-pairs" % len(my_tasks))

    # Process
    total_samples = 0
    skipped = 0
    processed = 0
    t_start = time.time()
    for task_i, (pair_idx, shard_name) in enumerate(my_tasks):
        t0 = time.time()
        n_samples, shape = merge_shard_pair(
            args.src_gcs, args.dst_gcs, pair_idx, shard_name,
            args.layer_index, args.tmp_dir
        )
        elapsed = time.time() - t0
        if n_samples == -1:
            skipped += 1
            continue
        total_samples += n_samples
        processed += 1
        if processed % 5 == 0 or processed == 1:
            remaining = len(my_tasks) - task_i - 1
            avg_time = (time.time() - t_start) / (task_i + 1)
            eta = avg_time * remaining
            log("  [%d/%d] pair_%02d/%s: %d samples, shape=%s (%.1fs, ETA %.0fs, skipped %d)" %
                (task_i + 1, len(my_tasks), pair_idx, shard_name, n_samples, shape, elapsed, eta, skipped))

    total_time = time.time() - t_start
    log("Done! Merged %d samples (%d shards processed, %d skipped) in %.1fs (%.1f min)" %
        (total_samples, processed, skipped, total_time, total_time / 60))


if __name__ == "__main__":
    main()
