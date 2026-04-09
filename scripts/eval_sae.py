#!/usr/bin/env python3
"""
Evaluate a trained SAE on activation data. Runs on a single TPU worker.

Downloads checkpoint from GCS, loads activation shards, runs forward passes,
computes and prints metrics.

Usage (on a TPU worker in the same region as GCS):
    python3 scripts/eval_sae.py \
        --gcs_bucket arc-data-europe-west4 \
        --gcs_prefix sae_checkpoints/layer19_topk_896d_v5e64 \
        --data_gcs gs://arc-data-europe-west4/activations/layer19_merged_50k \
        --layer_index 19 \
        --hidden_dim 896 --dict_size 7168 --k 32 \
        --num_eval_shards 20 --batch_size 1024
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import time

import ml_dtypes  # noqa: F401 — register bfloat16
import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gcs_ls(gcs_path, pattern="shard_*.pkl.gz"):
    r = subprocess.run(
        ["gcloud", "storage", "ls", os.path.join(gcs_path, pattern)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    return sorted([l.strip() for l in r.stdout.strip().split("\n") if l.strip()])


def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def load_shard_activations(local_path, layer_index):
    """Load activations from a gzipped pickle shard."""
    with gzip.open(local_path, "rb") as f:
        data = pickle.load(f)
    layer_data = data.get(layer_index) or data.get(str(layer_index))
    if layer_data is None:
        return None
    acts = [sample["activation"] for sample in layer_data]
    return np.concatenate(acts, axis=0)  # [total_tokens, hidden_dim]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAE")
    parser.add_argument("--gcs_bucket", default="arc-data-europe-west4")
    parser.add_argument("--gcs_prefix", default="sae_checkpoints/layer19_topk_896d_v5e64")
    parser.add_argument("--checkpoint_step", type=int, default=0, help="0=latest")
    parser.add_argument("--data_gcs", default="gs://arc-data-europe-west4/activations/layer19_merged_50k")
    parser.add_argument("--layer_index", type=int, default=19)
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--dict_size", type=int, default=7168)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--architecture", default="topk")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_eval_shards", type=int, default=20, help="Number of shards to eval on")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--tmp_dir", default="/tmp/sae_eval")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    log("=" * 60)
    log("SAE Evaluation")
    log("=" * 60)
    log(f"Checkpoint: gs://{args.gcs_bucket}/{args.gcs_prefix}")
    log(f"Data: {args.data_gcs}")
    log(f"Model: {args.architecture} (hidden={args.hidden_dim}, dict={args.dict_size}, k={args.k})")
    log(f"Eval shards: {args.num_eval_shards}, batch_size: {args.batch_size}")

    # === Step 1: Download checkpoint from GCS ===
    log("Step 1: Downloading checkpoint from GCS...")
    ckpt_dir = os.path.join(args.tmp_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get latest step
    if args.checkpoint_step == 0:
        marker_path = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/latest_step.json"
        local_marker = os.path.join(args.tmp_dir, "latest_step.json")
        gcs_cp(marker_path, local_marker)
        with open(local_marker) as f:
            step = json.load(f)["step"]
    else:
        step = args.checkpoint_step
    log(f"  Checkpoint step: {step}")

    step_dir = f"step_{step:08d}"
    gcs_step = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/{step_dir}"
    local_step = os.path.join(ckpt_dir, step_dir)
    if not os.path.exists(local_step):
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", gcs_step, ckpt_dir],
            capture_output=True, check=True,
        )
    log(f"  Downloaded to {local_step}")

    # === Step 2: Load model and restore params ===
    log("Step 2: Initializing JAX and loading model...")
    import jax
    import jax.numpy as jnp

    log(f"  JAX backend: {jax.default_backend()}, devices: {jax.device_count()}")

    from sae.configs.base import SAEConfig
    from sae.models.registry import create_sae
    from sae.evaluation.metrics import compute_metrics

    sae_config = SAEConfig(
        hidden_dim=args.hidden_dim,
        dict_size=args.dict_size,
        architecture=args.architecture,
        k=args.k,
        dtype=args.dtype,
    )
    model = create_sae(sae_config)

    # Init with dummy input to get param structure
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, args.hidden_dim), dtype=dtype)
    variables = model.init(rng, dummy)

    # Load checkpoint params
    with open(os.path.join(local_step, "metadata.json")) as f:
        metadata = json.load(f)
    param_keys = metadata["param_keys"]
    params_dir = os.path.join(local_step, "params")
    params_flat = [np.load(os.path.join(params_dir, f"{key}.npy")) for key in param_keys]

    # Restore param tree (handle bfloat16 saved as V2 void bytes)
    tree_def = jax.tree.structure(variables["params"])

    def safe_jnp_array(a):
        # .npy files with bfloat16 load as |V2 (void) when ml_dtypes isn't
        # available at save time. Reinterpret and convert through float32.
        if a.dtype.kind == 'V' and a.dtype.itemsize == 2:
            a = a.view(ml_dtypes.bfloat16)
        if hasattr(a.dtype, 'name') and a.dtype.name == 'bfloat16':
            return jnp.array(a.astype(np.float32)).astype(jnp.bfloat16)
        return jnp.array(a)

    params = jax.tree.unflatten(tree_def, [safe_jnp_array(a) for a in params_flat])

    # Load dead neuron tracking
    dead_path = os.path.join(local_step, "dead_neuron_steps.npy")
    dead_neuron_steps = np.load(dead_path) if os.path.exists(dead_path) else None

    log(f"  Model loaded: {sum(p.size for p in jax.tree.leaves(params)):,} parameters")
    log(f"  Checkpoint metadata: step={metadata['step']}, tokens={metadata.get('total_tokens', 'N/A')}")

    # === Step 3: Load evaluation data ===
    log("Step 3: Loading evaluation data from GCS...")

    # List all shards from pair_XX/host_XX subdirs
    all_shards = []
    all_entries = subprocess.run(
        ["gcloud", "storage", "ls", args.data_gcs + "/"],
        capture_output=True, text=True,
    ).stdout.strip().split("\n")

    sub_dirs = []
    for e in all_entries:
        e = e.strip().rstrip("/")
        if not e:
            continue
        dirname = e.split("/")[-1]
        if any(p in dirname for p in ("host_", "pair_")):
            sub_dirs.append(e)

    log(f"  Found {len(sub_dirs)} subdirectories")
    if sub_dirs:
        for sdir in sub_dirs:
            files = gcs_ls(sdir)
            all_shards.extend(files)
    else:
        all_shards = gcs_ls(args.data_gcs)

    log(f"  Total shards available: {len(all_shards)}")

    # Sample eval shards (spread across different subdirs)
    np.random.seed(42)
    eval_indices = np.random.choice(len(all_shards), min(args.num_eval_shards, len(all_shards)), replace=False)
    eval_shards = [all_shards[i] for i in sorted(eval_indices)]
    log(f"  Selected {len(eval_shards)} shards for evaluation")

    # Download and concatenate activations
    all_activations = []
    total_tokens = 0
    for i, shard_path in enumerate(eval_shards):
        shard_name = os.path.basename(shard_path)
        local_shard = os.path.join(args.tmp_dir, f"eval_{i}_{shard_name}")
        gcs_cp(shard_path, local_shard)
        acts = load_shard_activations(local_shard, args.layer_index)
        if acts is not None:
            all_activations.append(acts)
            total_tokens += acts.shape[0]
        os.remove(local_shard)
        if (i + 1) % 5 == 0:
            log(f"  Loaded {i + 1}/{len(eval_shards)} shards ({total_tokens:,} tokens)")

    eval_data = np.concatenate(all_activations, axis=0)
    log(f"  Eval data: {eval_data.shape} ({eval_data.dtype})")

    # === Step 4: Run evaluation ===
    log("Step 4: Running evaluation...")

    @jax.jit
    def eval_batch(params, x):
        x_hat, z, info = model.apply({"params": params}, x)
        return x_hat, z

    # Process in batches
    n = eval_data.shape[0]
    all_mse = []
    all_explained_var = []
    all_l0 = []
    all_active_per_feature = np.zeros(args.dict_size, dtype=np.float64)
    batch_count = 0

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = jnp.array(eval_data[start:end], dtype=dtype)
        x_hat, z = eval_batch(params, batch)

        # Compute per-batch metrics
        metrics = compute_metrics(batch, x_hat, z)
        all_mse.append(metrics["mse"])
        all_explained_var.append(metrics["explained_variance"])
        all_l0.append(metrics["l0"])

        # Track per-feature activation
        active = np.array(z != 0, dtype=np.float32)
        all_active_per_feature += active.sum(axis=0)
        batch_count += 1

        if batch_count % 10 == 0:
            log(f"  Batch {batch_count}: MSE={metrics['mse']:.6f}, "
                f"Explained var={metrics['explained_variance']:.4f}, L0={metrics['l0']:.1f}")

    # === Step 5: Aggregate and report ===
    log("")
    log("=" * 60)
    log("EVALUATION RESULTS")
    log("=" * 60)

    avg_mse = np.mean(all_mse)
    avg_explained_var = np.mean(all_explained_var)
    avg_l0 = np.mean(all_l0)

    # Feature utilization
    feature_density = all_active_per_feature / n
    ever_active = all_active_per_feature > 0
    dead_frac = 1.0 - ever_active.mean()

    log(f"  Checkpoint step:       {metadata['step']}")
    log(f"  Total tokens evaluated: {n:,}")
    log(f"  Eval shards:           {len(eval_shards)}")
    log(f"")
    log(f"  --- Reconstruction Quality ---")
    log(f"  MSE:                   {avg_mse:.6f}")
    log(f"  Explained variance:    {avg_explained_var:.4f}")
    log(f"  Normalized MSE:        {avg_mse / max(np.var(eval_data.astype(np.float32)), 1e-8):.6f}")
    log(f"")
    log(f"  --- Sparsity ---")
    log(f"  L0 (active features):  {avg_l0:.1f} / {args.dict_size}")
    log(f"  L0 fraction:           {avg_l0 / args.dict_size:.4f}")
    log(f"  k (target):            {args.k}")
    log(f"")
    log(f"  --- Feature Utilization ---")
    log(f"  Dead features:         {dead_frac:.1%} ({int(dead_frac * args.dict_size)} / {args.dict_size})")
    log(f"  Active features:       {ever_active.sum()} / {args.dict_size}")
    log(f"  Mean feature density:  {feature_density[ever_active].mean():.6f}")
    log(f"  Median feature density:{np.median(feature_density[ever_active]):.6f}")
    log(f"  Max feature density:   {feature_density.max():.6f}")

    if dead_neuron_steps is not None:
        dead_count = (dead_neuron_steps >= 10000).sum()
        log(f"")
        log(f"  --- Dead Neuron Tracking (from checkpoint) ---")
        log(f"  Dead (>10K steps):     {dead_count} / {args.dict_size} ({dead_count/args.dict_size:.1%})")
        log(f"  Max inactive steps:    {dead_neuron_steps.max()}")

    # Feature density histogram
    log(f"")
    log(f"  --- Feature Density Distribution ---")
    active_densities = feature_density[ever_active]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(active_densities, p) if len(active_densities) > 0 else 0
        log(f"  P{p:02d}: {val:.6f}")

    log(f"")
    log("=" * 60)
    log("Evaluation complete.")


if __name__ == "__main__":
    main()
