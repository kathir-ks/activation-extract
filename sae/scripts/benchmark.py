#!/usr/bin/env python3
"""Benchmark SAE training throughput on CPU vs TPU.

Downloads a small subset of activation shards from GCS and measures
training speed (tokens/sec, step time) on the specified backend.

Example usage:

    # Benchmark on CPU using GCS activations
    python -m sae.scripts.benchmark \
        --gcs_path gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/host_00 \
        --backend cpu \
        --num_steps 50

    # Benchmark on TPU (default)
    python -m sae.scripts.benchmark \
        --gcs_path gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/host_00 \
        --num_steps 100

    # Compare both backends
    python -m sae.scripts.benchmark \
        --gcs_path gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/host_00 \
        --compare \
        --num_steps 50
"""

import argparse
import os
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SAE training throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gcs_path", type=str, required=True,
                        help="GCS path to pickle shards")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["cpu", "tpu", "gpu"],
                        help="JAX backend (default: auto-detect)")
    parser.add_argument("--compare", action="store_true",
                        help="Run on both CPU and TPU, then compare")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of training steps to benchmark")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--architecture", type=str, default="vanilla",
                        choices=["vanilla", "topk", "gated", "jumprelu"])
    parser.add_argument("--expansion", type=int, default=16,
                        help="Dictionary expansion factor over hidden_dim")
    parser.add_argument("--layer_index", type=int, default=12)
    parser.add_argument("--max_shards", type=int, default=2,
                        help="Max shards to load (limits download)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps excluded from timing")
    return parser.parse_args()


def run_benchmark(backend, args):
    """Run a single benchmark on the specified backend. Returns metrics dict."""
    # Force backend
    os.environ["JAX_PLATFORMS"] = backend

    # Need fresh JAX import after setting env var in compare mode,
    # but JAX platform is set at first import. For compare mode,
    # we run each backend in a subprocess instead.
    import jax
    import jax.numpy as jnp
    import numpy as np

    actual_backend = jax.default_backend()
    print(f"\n{'='*60}")
    print(f"Benchmark: {actual_backend.upper()}")
    print(f"{'='*60}")
    print(f"  Devices: {jax.device_count()}")

    # 1. Load data from GCS
    print(f"  Loading shards from {args.gcs_path} ...")
    from sae.data.pickle_source import PickleShardSource

    source = PickleShardSource(
        shard_dir=".",  # unused when gcs_path is set
        layer_index=args.layer_index,
        gcs_path=args.gcs_path,
    )

    # Limit shards for benchmark
    if args.max_shards and len(source._shard_files) > args.max_shards:
        source._shard_files = source._shard_files[:args.max_shards]
        print(f"  Limited to {args.max_shards} shards for benchmark")

    hidden_dim = source.hidden_dim
    dict_size = hidden_dim * args.expansion
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dict size: {dict_size} ({args.expansion}x)")

    # 2. Pre-load vectors into memory
    print("  Pre-loading activation vectors into memory...")
    t_load = time.time()
    vectors = []
    for vec in source.iter_vectors():
        vectors.append(vec)
        if len(vectors) >= args.batch_size * (args.num_steps + args.warmup_steps + 5):
            break
    vectors = np.stack(vectors).astype(np.float32)
    load_time = time.time() - t_load
    print(f"  Loaded {len(vectors):,} vectors in {load_time:.1f}s")

    # 3. Setup model
    from sae.configs.base import SAEConfig
    from sae.models.registry import create_sae
    from sae.training.lr_schedule import create_optimizer
    from sae.training.train_state import SAETrainState
    from sae.training.distributed import create_sae_mesh
    from sae.configs.training import TrainingConfig
    from functools import partial

    # Use float32 on CPU
    dtype = "float32" if actual_backend == "cpu" else "bfloat16"

    sae_config = SAEConfig(
        hidden_dim=hidden_dim,
        dict_size=dict_size,
        architecture=args.architecture,
        dtype=dtype,
        k=min(64, dict_size),
    )
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=3e-4,
        lr_warmup_steps=0,
        lr_decay="constant",
    )

    model = create_sae(sae_config)
    rng = jax.random.PRNGKey(42)
    model_dtype = jnp.float32 if dtype == "float32" else jnp.bfloat16
    dummy = jnp.zeros((1, hidden_dim), dtype=model_dtype)
    variables = model.init(rng, dummy)

    optimizer = create_optimizer(train_config)
    state = SAETrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        dead_neuron_steps=jnp.zeros(dict_size, dtype=jnp.int32),
        total_tokens=0,
    )

    mesh = create_sae_mesh("auto")

    # JIT compile train step
    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch):
        def loss_fn(params):
            total_loss, loss_dict = model.apply(
                {"params": params}, batch, method=model.compute_loss
            )
            return total_loss, loss_dict
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_dict

    # 4. Run benchmark
    print(f"  Running {args.warmup_steps} warmup + {args.num_steps} timed steps...")
    total_steps = args.warmup_steps + args.num_steps
    step_times = []

    for step in range(total_steps):
        start = step * args.batch_size
        end = start + args.batch_size
        if end > len(vectors):
            # Wrap around
            batch_np = np.concatenate([vectors[start:], vectors[:end - len(vectors)]])
        else:
            batch_np = vectors[start:end]

        batch = jnp.array(batch_np, dtype=model_dtype)

        t0 = time.time()
        state, loss_dict = train_step(state, batch)
        # Force sync (JAX is async)
        jax.block_until_ready(state.params)
        dt = time.time() - t0

        if step >= args.warmup_steps:
            step_times.append(dt)

        if step < args.warmup_steps:
            status = "warmup"
        else:
            status = f"step {step - args.warmup_steps + 1}/{args.num_steps}"

        if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            loss_val = float(loss_dict.get("total", 0))
            print(f"    [{status}] loss={loss_val:.6f} time={dt*1000:.1f}ms")

    # 5. Compute stats
    step_times = np.array(step_times)
    tokens_per_step = args.batch_size
    tokens_per_sec = tokens_per_step / step_times

    results = {
        "backend": actual_backend,
        "hidden_dim": hidden_dim,
        "dict_size": dict_size,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
        "dtype": dtype,
        "num_devices": jax.device_count(),
        "num_steps": args.num_steps,
        "mean_step_ms": float(np.mean(step_times) * 1000),
        "median_step_ms": float(np.median(step_times) * 1000),
        "p95_step_ms": float(np.percentile(step_times, 95) * 1000),
        "mean_tokens_per_sec": float(np.mean(tokens_per_sec)),
        "median_tokens_per_sec": float(np.median(tokens_per_sec)),
        "total_tokens": int(tokens_per_step * args.num_steps),
        "total_time_sec": float(np.sum(step_times)),
    }

    print(f"\n  Results ({actual_backend.upper()}):")
    print(f"    Mean step time:     {results['mean_step_ms']:.1f} ms")
    print(f"    Median step time:   {results['median_step_ms']:.1f} ms")
    print(f"    P95 step time:      {results['p95_step_ms']:.1f} ms")
    print(f"    Mean tokens/sec:    {results['mean_tokens_per_sec']:,.0f}")
    print(f"    Median tokens/sec:  {results['median_tokens_per_sec']:,.0f}")
    print(f"    Total time:         {results['total_time_sec']:.1f}s")
    print(f"    Total tokens:       {results['total_tokens']:,}")

    return results


def main():
    args = parse_args()

    if args.compare:
        # Run CPU benchmark in subprocess to avoid JAX platform lock
        import subprocess
        import json

        results = {}

        for backend in ["cpu", "tpu"]:
            print(f"\n{'#'*60}")
            print(f"# Running {backend.upper()} benchmark")
            print(f"{'#'*60}")

            cmd = [
                sys.executable, "-m", "sae.scripts.benchmark",
                "--gcs_path", args.gcs_path,
                "--backend", backend,
                "--num_steps", str(args.num_steps),
                "--batch_size", str(args.batch_size),
                "--architecture", args.architecture,
                "--expansion", str(args.expansion),
                "--layer_index", str(args.layer_index),
                "--max_shards", str(args.max_shards),
                "--warmup_steps", str(args.warmup_steps),
            ]

            proc = subprocess.run(cmd, capture_output=False, text=True)
            if proc.returncode != 0:
                print(f"  {backend.upper()} benchmark failed (exit code {proc.returncode})")
                continue

        print(f"\n{'='*60}")
        print("Comparison complete. See individual results above.")
        print(f"{'='*60}")

    else:
        backend = args.backend or "auto"
        if backend == "auto":
            # Don't set env var, let JAX auto-detect
            import jax
            backend = jax.default_backend()
        else:
            os.environ["JAX_PLATFORMS"] = backend

        run_benchmark(backend, args)


if __name__ == "__main__":
    main()
