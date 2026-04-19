#!/usr/bin/env python3
"""
Phase 2: Color-level analysis of SAE features on ARC tasks.

Loads a bundle produced by arc_sae_collect.py and asks: which SAE features are
selective for which ARC color (digit 0..9)? For each feature we compute its
mean activation over each color class (token is a single digit D inside a grid)
and derive a selectivity score: (mean on D) / (mean over all colors).

Outputs per-color top-K features, a mean-activation matrix [n_features x 10],
and a set of "color detectors" where one color dominates (>=threshold share).

Runs on CPU, reads the bundle from GCS.

Usage:
    python3 scripts/arc_sae_color_analysis.py \
        --bundle_gcs gs://arc-data-europe-west4/sae_analysis/v2_collect/collected.pkl.gz \
        --output_gcs gs://arc-data-europe-west4/sae_analysis/v2_color \
        --top_k 20 --dominance 0.5
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import time
from typing import Any, Dict, List

import numpy as np


NUM_COLORS = 10  # ARC digits 0..9


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Color selectivity analysis for SAE features")
    parser.add_argument("--bundle_gcs", required=True)
    parser.add_argument("--output_gcs", required=True)
    parser.add_argument("--tmp_dir", default="/tmp/arc_sae_color")
    parser.add_argument("--top_k", type=int, default=20, help="Top features per color to report")
    parser.add_argument("--dominance", type=float, default=0.5,
                        help="Min share of activation that must come from a single color "
                             "for the feature to count as a color detector")
    parser.add_argument("--min_count", type=int, default=20,
                        help="Ignore features that fire fewer than this many times")
    parser.add_argument("--only_in_grid", action="store_true", default=True,
                        help="Restrict to tokens inside a grid block (default: True)")
    parser.add_argument("--include_non_grid", dest="only_in_grid", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    log("=" * 70)
    log("ARC SAE Color Analysis — Phase 2")
    log("=" * 70)
    bundle = load_bundle(args.bundle_gcs, args.tmp_dir)
    meta = bundle["meta"]
    dict_size = meta["sae_dict_size"]
    log(f"Bundle: {meta.get('sae_gcs_prefix')} arch={meta.get('sae_arch')} "
        f"dict={dict_size} step={meta.get('sae_step')} tasks={meta.get('num_tasks')}")

    # Accumulators
    # sum_by_color[c, f] = sum of activations for feature f on digit-color c tokens
    # count_by_color[c]  = number of digit-color c tokens seen
    # fire_count[f]      = how many tokens (across colors) fired f
    sum_by_color = np.zeros((NUM_COLORS, dict_size), dtype=np.float64)
    count_by_color = np.zeros(NUM_COLORS, dtype=np.int64)
    fire_count = np.zeros(dict_size, dtype=np.int64)
    fire_count_by_color = np.zeros((NUM_COLORS, dict_size), dtype=np.int64)
    feature_total_activation = np.zeros(dict_size, dtype=np.float64)

    total_tokens = 0
    digit_tokens = 0
    t0 = time.time()

    for ti, task in enumerate(bundle["tasks"]):
        tokens = task["tokens"]
        sparse = task["sparse_features"]
        for tok, pairs in zip(tokens, sparse):
            total_tokens += 1
            if not tok["is_single_digit"]:
                continue
            if args.only_in_grid and not tok["is_in_grid"]:
                continue
            c = tok["digit_value"]
            if c is None or not (0 <= c < NUM_COLORS):
                continue
            digit_tokens += 1
            count_by_color[c] += 1
            for f_idx, v in pairs:
                f = int(f_idx)
                sum_by_color[c, f] += v
                fire_count[f] += 1
                fire_count_by_color[c, f] += 1
                feature_total_activation[f] += v
        if (ti + 1) % 5 == 0 or ti + 1 == len(bundle["tasks"]):
            log(f"  processed {ti+1}/{len(bundle['tasks'])} tasks "
                f"({digit_tokens} digit tokens, {time.time()-t0:.1f}s)")

    log(f"Totals: {total_tokens} tokens, {digit_tokens} digit tokens")
    log(f"Tokens per color: {count_by_color.tolist()}")

    # Mean activation per (color, feature): guard against divide-by-zero
    safe_counts = np.maximum(count_by_color, 1).astype(np.float64)[:, None]
    mean_by_color = sum_by_color / safe_counts  # [10, dict_size]

    # Color selectivity: for each feature, find the color whose share of
    # total (sum over colors) activation is largest.
    total_per_feat = sum_by_color.sum(axis=0) + 1e-12  # [dict_size]
    share_by_color = sum_by_color / total_per_feat[None, :]  # [10, dict_size]

    dominant_color = np.argmax(share_by_color, axis=0)  # [dict_size]
    dominant_share = share_by_color[dominant_color, np.arange(dict_size)]  # [dict_size]

    # Filter features by minimum fire count
    active_mask = fire_count >= args.min_count
    num_active = int(active_mask.sum())
    log(f"Active features (fire_count >= {args.min_count}): {num_active} / {dict_size}")

    # Color detectors: dominant share above threshold AND feature is active
    detectors = []
    for f in np.where(active_mask)[0]:
        ds = float(dominant_share[f])
        if ds >= args.dominance:
            detectors.append({
                "feature_idx": int(f),
                "dominant_color": int(dominant_color[f]),
                "dominant_share": ds,
                "fire_count": int(fire_count[f]),
                "mean_by_color": mean_by_color[:, f].tolist(),
                "share_by_color": share_by_color[:, f].tolist(),
                "total_activation": float(feature_total_activation[f]),
            })
    detectors.sort(key=lambda d: (-d["dominant_share"], -d["fire_count"]))

    detectors_by_color: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(NUM_COLORS)}
    for d in detectors:
        detectors_by_color[d["dominant_color"]].append(d)

    log(f"Total color detectors (share >= {args.dominance}): {len(detectors)}")
    for c in range(NUM_COLORS):
        log(f"  color {c}: {len(detectors_by_color[c])} detectors")

    # Top-K features per color by mean activation (only among active features)
    top_k_by_color: Dict[int, List[Dict[str, Any]]] = {}
    for c in range(NUM_COLORS):
        scores = mean_by_color[c].copy()
        scores[~active_mask] = -np.inf
        top_idx = np.argsort(-scores)[: args.top_k]
        rows = []
        for f in top_idx:
            if scores[f] == -np.inf:
                break
            rows.append({
                "feature_idx": int(f),
                "mean_activation": float(mean_by_color[c, f]),
                "dominant_color": int(dominant_color[f]),
                "dominant_share": float(dominant_share[f]),
                "fire_count": int(fire_count[f]),
                "fire_count_this_color": int(fire_count_by_color[c, f]),
            })
        top_k_by_color[c] = rows

    # Save outputs
    out_json = {
        "meta": meta,
        "analysis_config": {
            "top_k": args.top_k,
            "dominance": args.dominance,
            "min_count": args.min_count,
            "only_in_grid": args.only_in_grid,
        },
        "stats": {
            "total_tokens": int(total_tokens),
            "digit_tokens": int(digit_tokens),
            "tokens_per_color": count_by_color.tolist(),
            "num_active_features": num_active,
            "num_detectors": len(detectors),
            "detectors_per_color": {c: len(detectors_by_color[c]) for c in range(NUM_COLORS)},
        },
        "top_k_by_color": {str(c): top_k_by_color[c] for c in range(NUM_COLORS)},
        "detectors_by_color": {str(c): detectors_by_color[c][: args.top_k]
                               for c in range(NUM_COLORS)},
    }
    out_local = os.path.join(args.tmp_dir, "color_analysis.json")
    with open(out_local, "w") as f:
        json.dump(out_json, f, indent=2)
    log(f"Wrote {out_local} ({os.path.getsize(out_local)/1e3:.1f} KB)")

    # Raw arrays pickle (for downstream analysis)
    arrays_local = os.path.join(args.tmp_dir, "color_arrays.pkl.gz")
    with gzip.open(arrays_local, "wb") as f:
        pickle.dump({
            "meta": meta,
            "mean_by_color": mean_by_color.astype(np.float32),
            "share_by_color": share_by_color.astype(np.float32),
            "sum_by_color": sum_by_color.astype(np.float32),
            "count_by_color": count_by_color,
            "fire_count": fire_count,
            "fire_count_by_color": fire_count_by_color,
            "feature_total_activation": feature_total_activation.astype(np.float32),
            "dominant_color": dominant_color,
            "dominant_share": dominant_share.astype(np.float32),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Wrote {arrays_local} ({os.path.getsize(arrays_local)/1e6:.2f} MB)")

    log(f"Uploading to {args.output_gcs}/...")
    gcs_cp(out_local, f"{args.output_gcs}/color_analysis.json")
    gcs_cp(arrays_local, f"{args.output_gcs}/color_arrays.pkl.gz")
    log("Upload complete.")

    # Pretty printout of top detectors per color
    log("-" * 70)
    log("Top 5 color detectors per color (feature_idx, dom_share, fire_count):")
    for c in range(NUM_COLORS):
        top5 = detectors_by_color[c][:5]
        if not top5:
            log(f"  color {c}: (none)")
            continue
        items = ", ".join(
            f"f{d['feature_idx']}({d['dominant_share']:.2f},{d['fire_count']})"
            for d in top5
        )
        log(f"  color {c}: {items}")
    log("=" * 70)
    log("Phase 2 DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
