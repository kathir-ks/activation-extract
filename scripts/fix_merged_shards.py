#!/usr/bin/env python3
"""
Fix merged shards: delete bad 1792-dim shards and copy 896-dim source shards directly.

Source shards 0001-0294: 448-dim (paired merge produces correct 896-dim)
Source shards 0295-0631: 896-dim (should NOT be concatenated, copy directly)

For 896-dim shards, each host has independent data. We store them in host_XX/
subdirectories in the merged output, which the SAE data loader already supports.
"""

import argparse
import os
import subprocess
import time


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gcs_ls(gcs_path, pattern="shard_*.pkl.gz"):
    result = subprocess.run(
        ["gcloud", "storage", "ls", os.path.join(gcs_path, pattern)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return sorted([l.strip() for l in result.stdout.strip().split("\n") if l.strip()])


def gcs_rm(gcs_path):
    result = subprocess.run(
        ["gcloud", "storage", "rm", gcs_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


def gcs_cp(src, dst):
    result = subprocess.run(
        ["gcloud", "storage", "cp", src, dst],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("cp failed: %s" % result.stderr)


FIRST_896_SHARD = 295  # shard_0295 is the first 896-dim shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_gcs", default="gs://arc-data-europe-west4/activations/layer19_gridchunk_50k_v5litepod-64")
    parser.add_argument("--dst_gcs", default="gs://arc-data-europe-west4/activations/layer19_merged_50k")
    parser.add_argument("--num_hosts", type=int, default=16)
    parser.add_argument("--num_pairs", type=int, default=8)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--delete_bad", action="store_true", help="Delete bad 1792-dim shards from pair_XX dirs")
    parser.add_argument("--copy_896", action="store_true", help="Copy 896-dim shards to host_XX dirs")
    args = parser.parse_args()

    log("Fix worker %d/%d" % (args.worker_id, args.num_workers))

    if args.delete_bad:
        log("Phase 1: Deleting bad shards from pair_XX dirs...")
        # Build list of bad shards to delete
        all_bad = []
        for pair_idx in range(args.num_pairs):
            pair_dir = "%s/pair_%02d" % (args.dst_gcs, pair_idx)
            shards = gcs_ls(pair_dir)
            for s in shards:
                shard_name = os.path.basename(s)
                # Extract shard number
                try:
                    num = int(shard_name.replace("shard_", "").replace(".pkl.gz", ""))
                except ValueError:
                    continue
                if num >= FIRST_896_SHARD:
                    all_bad.append(s)

        # Distribute across workers
        my_bad = [s for i, s in enumerate(all_bad) if i % args.num_workers == args.worker_id]
        log("Total bad shards: %d, this worker: %d" % (len(all_bad), len(my_bad)))

        deleted = 0
        for i, path in enumerate(my_bad):
            gcs_rm(path)
            deleted += 1
            if deleted % 50 == 0:
                log("  Deleted %d/%d" % (deleted, len(my_bad)))
        log("Phase 1 done: deleted %d bad shards" % deleted)

    if args.copy_896:
        log("Phase 2: Copying 896-dim shards to host_XX dirs...")
        # Build list of (src, dst) copy tasks
        all_copies = []
        for host_idx in range(args.num_hosts):
            host_src = "%s/host_%02d" % (args.src_gcs, host_idx)
            host_dst = "%s/host_%02d" % (args.dst_gcs, host_idx)
            shards = gcs_ls(host_src)
            for s in shards:
                shard_name = os.path.basename(s)
                try:
                    num = int(shard_name.replace("shard_", "").replace(".pkl.gz", ""))
                except ValueError:
                    continue
                if num >= FIRST_896_SHARD:
                    all_copies.append((s, "%s/%s" % (host_dst, shard_name)))

        # Distribute across workers
        my_copies = [c for i, c in enumerate(all_copies) if i % args.num_workers == args.worker_id]
        log("Total copies: %d, this worker: %d" % (len(all_copies), len(my_copies)))

        copied = 0
        for i, (src, dst) in enumerate(my_copies):
            try:
                gcs_cp(src, dst)
                copied += 1
            except RuntimeError as e:
                log("  ERROR copying %s: %s" % (src, e))
            if copied % 50 == 0 and copied > 0:
                log("  Copied %d/%d" % (copied, len(my_copies)))
        log("Phase 2 done: copied %d shards" % copied)


if __name__ == "__main__":
    main()
