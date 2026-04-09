#!/usr/bin/env python3
"""
Diagnose SAE quality issues. Checks:
1. Checkpoint weight statistics (corruption detection)
2. Early vs late checkpoint comparison
3. Quick eval on small data sample
4. Training log analysis (if available)

Runs on CPU, no TPU needed.
"""

import json
import os
import subprocess
import sys
import time

import ml_dtypes  # noqa: F401
import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def download_checkpoint(gcs_bucket, gcs_prefix, step, tmp_dir):
    """Download a specific checkpoint step."""
    step_dir = f"step_{step:08d}"
    gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}/{step_dir}"
    local_path = os.path.join(tmp_dir, step_dir)
    if not os.path.exists(local_path):
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", gcs_path, tmp_dir],
            capture_output=True, check=True,
        )
    return local_path


def load_params_flat(ckpt_dir):
    """Load flat parameter arrays from checkpoint."""
    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        metadata = json.load(f)
    param_keys = metadata["param_keys"]
    params_dir = os.path.join(ckpt_dir, "params")
    params = {}
    for key in param_keys:
        a = np.load(os.path.join(params_dir, f"{key}.npy"))
        # Fix V2 dtype
        if a.dtype.kind == 'V' and a.dtype.itemsize == 2:
            a = a.view(ml_dtypes.bfloat16)
        params[key] = a
    return params, metadata


def param_stats(params):
    """Print stats for each parameter array."""
    for key, a in params.items():
        a_f32 = a.astype(np.float32)
        print(f"    {key:30s} shape={str(a.shape):20s} dtype={str(a.dtype):10s} "
              f"mean={a_f32.mean():.6f} std={a_f32.std():.6f} "
              f"min={a_f32.min():.4f} max={a_f32.max():.4f} "
              f"norm={np.linalg.norm(a_f32):.2f} "
              f"nan={np.isnan(a_f32).sum()} inf={np.isinf(a_f32).sum()}", flush=True)


def quick_eval(params, data, hidden_dim, dict_size, k):
    """Run a quick SAE forward pass using raw numpy (no JAX needed)."""
    # Extract weight matrices
    W_enc = params.get("W_enc", params.get("encoder/W_enc"))
    b_enc = params.get("b_enc", params.get("encoder/b_enc"))
    W_dec = params.get("W_dec", params.get("decoder/W_dec"))
    b_dec = params.get("b_dec", params.get("decoder/b_dec"))

    if W_enc is None:
        # Try flattened key names
        for key in params:
            if "W_enc" in key:
                W_enc = params[key]
            elif "b_enc" in key:
                b_enc = params[key]
            elif "W_dec" in key:
                W_dec = params[key]
            elif "b_dec" in key:
                b_dec = params[key]

    if W_enc is None:
        print("  ERROR: Could not find W_enc in params. Keys:", list(params.keys()), flush=True)
        return None

    W_enc = W_enc.astype(np.float32)
    b_enc = b_enc.astype(np.float32)
    W_dec = W_dec.astype(np.float32)
    b_dec = b_dec.astype(np.float32)

    log(f"  W_enc: {W_enc.shape}, W_dec: {W_dec.shape}, b_enc: {b_enc.shape}, b_dec: {b_dec.shape}")

    x = data.astype(np.float32)

    # Encode
    x_centered = x - b_dec
    z_pre = x_centered @ W_enc + b_enc

    # TopK activation
    topk_indices = np.argpartition(-z_pre, k, axis=-1)[:, :k]
    z = np.zeros_like(z_pre)
    for i in range(len(z)):
        vals = np.maximum(z_pre[i, topk_indices[i]], 0)  # ReLU
        z[i, topk_indices[i]] = vals

    # Decode
    x_hat = z @ W_dec + b_dec

    # Metrics
    mse = np.mean((x - x_hat) ** 2)
    variance = np.var(x)
    explained_var = 1.0 - np.var(x - x_hat) / max(variance, 1e-8)
    active = z != 0
    l0 = np.mean(np.sum(active, axis=-1))

    return {
        "mse": mse,
        "explained_variance": explained_var,
        "variance": variance,
        "l0": l0,
        "x_mean": np.mean(x),
        "x_std": np.std(x),
        "x_hat_mean": np.mean(x_hat),
        "x_hat_std": np.std(x_hat),
        "z_pre_mean": np.mean(z_pre),
        "z_pre_std": np.std(z_pre),
        "z_active_mean": np.mean(z[z != 0]) if np.any(z != 0) else 0,
        "residual_mean": np.mean(x - x_hat),
        "residual_std": np.std(x - x_hat),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_bucket", default="arc-data-europe-west4")
    parser.add_argument("--gcs_prefix", default="sae_checkpoints/layer19_topk_896d_v5e64")
    parser.add_argument("--data_gcs", default="gs://arc-data-europe-west4/activations/layer19_merged_50k")
    parser.add_argument("--layer_index", type=int, default=19)
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--dict_size", type=int, default=7168)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--tmp_dir", default="/tmp/sae_diag")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    log("=" * 70)
    log("SAE DIAGNOSTIC")
    log("=" * 70)

    # === 1. Check latest checkpoint weights ===
    log("")
    log("=== 1. LATEST CHECKPOINT WEIGHT STATISTICS ===")
    latest_dir = download_checkpoint(args.gcs_bucket, args.gcs_prefix, 200000, args.tmp_dir)
    params_200k, meta_200k = load_params_flat(latest_dir)
    log(f"  Step: {meta_200k['step']}, Tokens: {meta_200k.get('total_tokens', 'N/A')}")
    param_stats(params_200k)

    # Check for NaN/Inf
    has_nan = any(np.isnan(a.astype(np.float32)).any() for a in params_200k.values())
    has_inf = any(np.isinf(a.astype(np.float32)).any() for a in params_200k.values())
    log(f"\n  NaN in params: {has_nan}")
    log(f"  Inf in params: {has_inf}")

    # === 2. Check early checkpoint for comparison ===
    log("")
    log("=== 2. EARLY CHECKPOINT (step 5000) WEIGHT STATISTICS ===")
    early_dir = download_checkpoint(args.gcs_bucket, args.gcs_prefix, 5000, args.tmp_dir)
    params_5k, meta_5k = load_params_flat(early_dir)
    log(f"  Step: {meta_5k['step']}")
    param_stats(params_5k)

    # === 3. Compare early vs late ===
    log("")
    log("=== 3. WEIGHT DRIFT: step 5000 → step 200000 ===")
    for key in params_200k:
        a_early = params_5k[key].astype(np.float32)
        a_late = params_200k[key].astype(np.float32)
        diff = a_late - a_early
        cos_sim = np.sum(a_early.flatten() * a_late.flatten()) / (
            np.linalg.norm(a_early) * np.linalg.norm(a_late) + 1e-8
        )
        print(f"    {key:30s} diff_norm={np.linalg.norm(diff):.4f} "
              f"cos_sim={cos_sim:.6f} "
              f"early_norm={np.linalg.norm(a_early):.4f} "
              f"late_norm={np.linalg.norm(a_late):.4f}", flush=True)

    # === 4. Load a small data sample and run quick eval ===
    log("")
    log("=== 4. QUICK EVAL (numpy, no JAX) ===")

    # Download 1 shard
    import gzip
    import pickle

    shard_path = f"{args.data_gcs}/pair_00/shard_0001.pkl.gz"
    local_shard = os.path.join(args.tmp_dir, "test_shard.pkl.gz")
    log(f"  Downloading: {shard_path}")
    gcs_cp(shard_path, local_shard)

    with gzip.open(local_shard, "rb") as f:
        shard_data = pickle.load(f)

    layer_data = shard_data.get(args.layer_index) or shard_data.get(str(args.layer_index))
    acts = np.concatenate([s["activation"] for s in layer_data], axis=0)
    log(f"  Loaded: {acts.shape} dtype={acts.dtype}")

    # Take first 2048 tokens for quick eval
    sample = acts[:2048]
    log(f"  Eval sample: {sample.shape}")

    # Run eval with step 200000
    log(f"\n  --- Step 200000 ---")
    metrics_200k = quick_eval(params_200k, sample, args.hidden_dim, args.dict_size, args.k)
    if metrics_200k:
        for k, v in metrics_200k.items():
            log(f"    {k}: {v:.6f}")

    # Run eval with step 5000
    log(f"\n  --- Step 5000 ---")
    metrics_5k = quick_eval(params_5k, sample, args.hidden_dim, args.dict_size, args.k)
    if metrics_5k:
        for k, v in metrics_5k.items():
            log(f"    {k}: {v:.6f}")

    # === 5. Check a mid-training checkpoint ===
    log("")
    log("=== 5. MID-TRAINING CHECKPOINT (step 50000) ===")
    mid_dir = download_checkpoint(args.gcs_bucket, args.gcs_prefix, 50000, args.tmp_dir)
    params_50k, meta_50k = load_params_flat(mid_dir)
    log(f"  Step: {meta_50k['step']}")
    param_stats(params_50k)

    log(f"\n  --- Step 50000 eval ---")
    metrics_50k = quick_eval(params_50k, sample, args.hidden_dim, args.dict_size, args.k)
    if metrics_50k:
        for k, v in metrics_50k.items():
            log(f"    {k}: {v:.6f}")

    # === Summary ===
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    if metrics_5k and metrics_50k and metrics_200k:
        log(f"  {'Step':>10s} {'MSE':>10s} {'ExplVar':>10s} {'L0':>8s} {'x_std':>8s} {'xhat_std':>10s}")
        for step, m in [("5000", metrics_5k), ("50000", metrics_50k), ("200000", metrics_200k)]:
            log(f"  {step:>10s} {m['mse']:>10.4f} {m['explained_variance']:>10.4f} "
                f"{m['l0']:>8.1f} {m['x_std']:>8.4f} {m['x_hat_std']:>10.4f}")

    log("")
    log("Diagnostic complete.")


if __name__ == "__main__":
    main()
