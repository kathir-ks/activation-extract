#!/usr/bin/env python3
"""
Phase 3: Shape / connected-component analysis of SAE features on ARC grids.

Questions:
  1. Do any SAE features fire preferentially on *non-background* shape tokens
     (i.e. cells that belong to a small connected component of a non-0 color)?
  2. Do any features abstract a shape as a single entity — i.e. fire on the
     whole shape (or its first-encountered cell) regardless of the color it is
     painted in, or regardless of where it sits in the grid?
  3. Which features are "edge detectors" (fire on a cell adjacent to a
     background cell or to a different color)?

Approach:
  - For each grid block, parse 4-connected components on non-background
    (nonzero) cells. Tag every digit-cell token with (component_id,
    component_size, component_color, is_edge). Background components (color 0)
    are also tracked separately.
  - Bucket tokens by (component_color, is_edge, size bucket) and by shape
    signature (color-independent bitmap hash, rotated/flipped to canonical).
  - For each feature, compute:
      * mean activation by edge vs interior
      * mean activation by component-size bucket
      * entropy of the shape-signature distribution it fires on — low entropy
        means the feature is shape-selective
      * top shape signatures it fires on

Runs on CPU, reads bundle from GCS.
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


BACKGROUND = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", src, dst], check=True)


def load_bundle(bundle_gcs: str, tmp_dir: str) -> Dict[str, Any]:
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, os.path.basename(bundle_gcs))
    if not os.path.exists(local):
        log(f"Downloading bundle: {bundle_gcs}")
        gcs_cp(bundle_gcs, local)
    with gzip.open(local, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Connected-component parsing
# ---------------------------------------------------------------------------

def parse_components(grid: List[List[int]], include_background: bool = False
                     ) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """4-connected components on a grid, per color.

    Returns:
        comp_id: int array [rows, cols], -1 for skipped cells
        comps:   dict comp_id -> {color, cells:[(r,c)], rows, cols, bitmap, size, bbox}
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    g = np.array(grid, dtype=np.int8)
    comp_id = np.full((rows, cols), -1, dtype=np.int32)
    comps: Dict[int, Dict[str, Any]] = {}
    next_id = 0

    for r in range(rows):
        for c in range(cols):
            if comp_id[r, c] != -1:
                continue
            color = int(g[r, c])
            if color == BACKGROUND and not include_background:
                continue
            # BFS
            stack = [(r, c)]
            cells: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if comp_id[rr, cc] != -1:
                    continue
                if int(g[rr, cc]) != color:
                    continue
                comp_id[rr, cc] = next_id
                cells.append((rr, cc))
                stack.extend([(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)])
            if not cells:
                continue
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            bitmap = np.zeros((h, w), dtype=np.uint8)
            for rr, cc in cells:
                bitmap[rr - r0, cc - c0] = 1
            comps[next_id] = {
                "color": color,
                "cells": cells,
                "rows": h,
                "cols": w,
                "bitmap": bitmap,
                "size": len(cells),
                "bbox": (r0, c0, r1, c1),
            }
            next_id += 1
    return comp_id, comps


def canonical_shape_signature(bitmap: np.ndarray) -> bytes:
    """Color-independent signature: take min over all D4 transforms of the
    binary bitmap. Small components collide on simple shapes across orientations."""
    variants = []
    b = bitmap
    for _ in range(4):
        variants.append(b.tobytes() + b":" + f"{b.shape[0]}x{b.shape[1]}".encode())
        variants.append(np.fliplr(b).tobytes() + b":" + f"{b.shape[0]}x{b.shape[1]}".encode())
        b = np.rot90(b)
    return min(variants)


def is_edge_cell(g: np.ndarray, r: int, c: int) -> bool:
    rows, cols = g.shape
    color = int(g[r, c])
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            return True  # grid boundary counts as edge
        if int(g[nr, nc]) != color:
            return True
    return False


def size_bucket(sz: int) -> str:
    if sz == 1:
        return "1"
    if sz <= 3:
        return "2-3"
    if sz <= 6:
        return "4-6"
    if sz <= 12:
        return "7-12"
    if sz <= 30:
        return "13-30"
    return "31+"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Shape-level SAE feature analysis on ARC")
    parser.add_argument("--bundle_gcs", required=True)
    parser.add_argument("--output_gcs", required=True)
    parser.add_argument("--tmp_dir", default="/tmp/arc_sae_shape")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_count", type=int, default=20,
                        help="Ignore features that fire fewer than this many times")
    parser.add_argument("--min_shape_fires", type=int, default=10,
                        help="Minimum fires on a given shape signature to count it")
    parser.add_argument("--shape_top_sig", type=int, default=3,
                        help="Report this many top shape signatures per feature")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    log("=" * 70)
    log("ARC SAE Shape Analysis — Phase 3")
    log("=" * 70)
    bundle = load_bundle(args.bundle_gcs, args.tmp_dir)
    meta = bundle["meta"]
    dict_size = meta["sae_dict_size"]
    log(f"Bundle: {meta.get('sae_gcs_prefix')} dict={dict_size} "
        f"tasks={meta.get('num_tasks')}")

    # Parse components for every grid block in every task
    # Build: for each (task_idx, block_idx) -> comp_id array + comps dict
    all_comps: Dict[Tuple[int, int], Tuple[np.ndarray, Dict[int, Dict[str, Any]]]] = {}
    # block_np[(task_idx, block_idx)] = np.int8 grid
    block_np: Dict[Tuple[int, int], np.ndarray] = {}

    for ti, task in enumerate(bundle["tasks"]):
        for bi, b in enumerate(task["blocks"]):
            g = np.array(b["grid"], dtype=np.int8)
            block_np[(ti, bi)] = g
            comp_id_map, comps = parse_components(b["grid"])
            all_comps[(ti, bi)] = (comp_id_map, comps)

    # Accumulators per feature
    # fire_count[f], edge_fire[f], interior_fire[f]
    fire_count = np.zeros(dict_size, dtype=np.int64)
    edge_fire = np.zeros(dict_size, dtype=np.int64)
    interior_fire = np.zeros(dict_size, dtype=np.int64)
    size_bucket_fire: Dict[str, np.ndarray] = {
        b: np.zeros(dict_size, dtype=np.int64)
        for b in ["1", "2-3", "4-6", "7-12", "13-30", "31+"]
    }

    # Shape signatures: map signature -> int id
    sig_to_id: Dict[bytes, int] = {}
    sig_examples: Dict[int, Dict[str, Any]] = {}
    # per-feature counter of sig_id -> count
    feat_sig_fire: Dict[int, Counter] = defaultdict(Counter)
    # global counter of sig_id -> count (to know which are common)
    sig_fire_total: Counter = Counter()
    sig_token_count: Counter = Counter()  # how many tokens total belong to each sig

    total_digit_tokens = 0
    t0 = time.time()

    for ti, task in enumerate(bundle["tasks"]):
        tokens = task["tokens"]
        sparse = task["sparse_features"]
        for tok, pairs in zip(tokens, sparse):
            if not tok["is_single_digit"] or not tok["is_in_grid"]:
                continue
            bi = tok["block_idx"]
            r = tok["row"]
            c = tok["col"]
            if r is None or c is None:
                continue
            comp_id_map, comps = all_comps[(ti, bi)]
            grid = block_np[(ti, bi)]
            cid = int(comp_id_map[r, c])
            color = int(grid[r, c])

            if cid == -1:
                # Background (color 0) with include_background=False
                bucket = "bg"
                is_edge = is_edge_cell(grid, r, c)
                sig_id = None
                sz = None
            else:
                comp = comps[cid]
                sig_bytes = canonical_shape_signature(comp["bitmap"])
                if sig_bytes not in sig_to_id:
                    sig_to_id[sig_bytes] = len(sig_to_id)
                    sig_examples[sig_to_id[sig_bytes]] = {
                        "rows": int(comp["rows"]),
                        "cols": int(comp["cols"]),
                        "size": int(comp["size"]),
                        "bitmap": comp["bitmap"].tolist(),
                    }
                sig_id = sig_to_id[sig_bytes]
                sz = comp["size"]
                bucket = size_bucket(sz)
                is_edge = is_edge_cell(grid, r, c)
                sig_token_count[sig_id] += 1

            total_digit_tokens += 1

            for f_idx, _v in pairs:
                f = int(f_idx)
                fire_count[f] += 1
                if is_edge:
                    edge_fire[f] += 1
                else:
                    interior_fire[f] += 1
                if bucket in size_bucket_fire:
                    size_bucket_fire[bucket][f] += 1
                if sig_id is not None:
                    feat_sig_fire[f][sig_id] += 1
                    sig_fire_total[sig_id] += 1

        if (ti + 1) % 25 == 0 or ti + 1 == len(bundle["tasks"]):
            log(f"  processed {ti+1}/{len(bundle['tasks'])} tasks "
                f"({total_digit_tokens} digit tokens, {time.time()-t0:.1f}s)")

    log(f"Total digit tokens processed: {total_digit_tokens}")
    log(f"Unique shape signatures: {len(sig_to_id)}")

    # Active features
    active_mask = fire_count >= args.min_count
    num_active = int(active_mask.sum())
    log(f"Active features (>= {args.min_count} fires): {num_active}")

    # Edge specificity: (edge_fire - interior_fire) / fire_count
    edge_share = np.where(fire_count > 0, edge_fire / np.maximum(fire_count, 1), 0.0)
    interior_share = np.where(fire_count > 0, interior_fire / np.maximum(fire_count, 1), 0.0)

    # Rank edge-selective features
    edge_ranked = []
    for f in np.where(active_mask)[0]:
        if edge_share[f] >= 0.85:
            edge_ranked.append({
                "feature_idx": int(f),
                "edge_share": float(edge_share[f]),
                "fire_count": int(fire_count[f]),
            })
    edge_ranked.sort(key=lambda d: (-d["edge_share"], -d["fire_count"]))

    interior_ranked = []
    for f in np.where(active_mask)[0]:
        if interior_share[f] >= 0.85:
            interior_ranked.append({
                "feature_idx": int(f),
                "interior_share": float(interior_share[f]),
                "fire_count": int(fire_count[f]),
            })
    interior_ranked.sort(key=lambda d: (-d["interior_share"], -d["fire_count"]))

    log(f"Edge-selective features (>=0.85): {len(edge_ranked)}")
    log(f"Interior-selective features (>=0.85): {len(interior_ranked)}")

    # Size-bucket selectivity: share of fires in each bucket
    size_buckets = ["1", "2-3", "4-6", "7-12", "13-30", "31+"]
    size_share = np.zeros((len(size_buckets), dict_size), dtype=np.float32)
    denom = np.maximum(fire_count, 1)
    for i, bk in enumerate(size_buckets):
        size_share[i] = size_bucket_fire[bk] / denom

    size_selective: Dict[str, List[Dict[str, Any]]] = {}
    for i, bk in enumerate(size_buckets):
        ranked = []
        for f in np.where(active_mask)[0]:
            share = float(size_share[i, f])
            if share >= 0.5:
                ranked.append({
                    "feature_idx": int(f),
                    "share": share,
                    "fire_count": int(fire_count[f]),
                })
        ranked.sort(key=lambda d: (-d["share"], -d["fire_count"]))
        size_selective[bk] = ranked[: args.top_k]
        log(f"  size bucket {bk}: {len(ranked)} selective features (>=0.5)")

    # Shape-selective features: low entropy over shape signatures, but only
    # among features with enough shape-bearing fires.
    shape_selective = []
    for f in np.where(active_mask)[0]:
        counter = feat_sig_fire.get(int(f))
        if not counter:
            continue
        total = sum(counter.values())
        if total < args.min_shape_fires:
            continue
        probs = np.array(list(counter.values()), dtype=np.float64) / total
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        # effective # of shapes: 2^entropy
        eff = float(2 ** entropy)
        top_sigs = counter.most_common(args.shape_top_sig)
        top_share = top_sigs[0][1] / total
        shape_selective.append({
            "feature_idx": int(f),
            "shape_fires": int(total),
            "entropy": entropy,
            "effective_shapes": eff,
            "top_share": float(top_share),
            "top_sigs": [
                {
                    "sig_id": int(sid),
                    "count": int(cnt),
                    "rows": sig_examples[sid]["rows"],
                    "cols": sig_examples[sid]["cols"],
                    "size": sig_examples[sid]["size"],
                }
                for sid, cnt in top_sigs
            ],
        })
    shape_selective.sort(key=lambda d: (-d["top_share"], d["entropy"]))
    top_shape_selective = shape_selective[: args.top_k * 2]
    log(f"Shape-scored features: {len(shape_selective)} "
        f"(top shape-selective: {len(top_shape_selective)})")

    # Report top shape signatures globally
    top_global_sigs = sig_fire_total.most_common(args.top_k)
    top_global_sigs_report = []
    for sid, cnt in top_global_sigs:
        ex = sig_examples[sid]
        top_global_sigs_report.append({
            "sig_id": int(sid),
            "fires": int(cnt),
            "tokens": int(sig_token_count[sid]),
            "rows": ex["rows"],
            "cols": ex["cols"],
            "size": ex["size"],
            "bitmap": ex["bitmap"],
        })

    out = {
        "meta": meta,
        "analysis_config": {
            "top_k": args.top_k,
            "min_count": args.min_count,
            "min_shape_fires": args.min_shape_fires,
        },
        "stats": {
            "total_digit_tokens": int(total_digit_tokens),
            "num_active_features": num_active,
            "num_shape_signatures": len(sig_to_id),
            "num_edge_selective": len(edge_ranked),
            "num_interior_selective": len(interior_ranked),
        },
        "edge_selective": edge_ranked[: args.top_k],
        "interior_selective": interior_ranked[: args.top_k],
        "size_selective_by_bucket": size_selective,
        "top_shape_selective_features": top_shape_selective,
        "top_global_shape_signatures": top_global_sigs_report,
    }

    out_local = os.path.join(args.tmp_dir, "shape_analysis.json")
    with open(out_local, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Wrote {out_local} ({os.path.getsize(out_local)/1e3:.1f} KB)")

    # Raw per-feature arrays too
    arrays_local = os.path.join(args.tmp_dir, "shape_arrays.pkl.gz")
    with gzip.open(arrays_local, "wb") as f:
        pickle.dump({
            "meta": meta,
            "fire_count": fire_count,
            "edge_fire": edge_fire,
            "interior_fire": interior_fire,
            "size_bucket_fire": {k: v for k, v in size_bucket_fire.items()},
            "sig_examples": sig_examples,
            "sig_fire_total": dict(sig_fire_total),
            "feat_sig_fire": {k: dict(v) for k, v in feat_sig_fire.items()},
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Wrote {arrays_local} ({os.path.getsize(arrays_local)/1e6:.2f} MB)")

    log(f"Uploading to {args.output_gcs}/...")
    gcs_cp(out_local, f"{args.output_gcs}/shape_analysis.json")
    gcs_cp(arrays_local, f"{args.output_gcs}/shape_arrays.pkl.gz")
    log("Upload complete.")
    log("=" * 70)
    log("Phase 3 DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
