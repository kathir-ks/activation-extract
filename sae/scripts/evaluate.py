#!/usr/bin/env python3
"""Evaluate a trained SAE on held-out activations.

Example:
    python -m sae.scripts.evaluate \
        --checkpoint_dir ./sae_checkpoints \
        --source_type pickle \
        --data_dir ./activations_eval/tpu_0 \
        --layer_index 12
"""

import argparse
import json

import jax
import jax.numpy as jnp

from sae.configs.base import SAEConfig
from sae.evaluation.metrics import compute_metrics
from sae.models.registry import create_sae
from sae.training.checkpointing import load_checkpoint, restore_params
from sae.data.registry import create_source


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAE")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--source_type", type=str, default="pickle")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--layer_index", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint_dir, step=args.step)
    if ckpt is None:
        print(f"No checkpoint found in {args.checkpoint_dir}")
        return

    metadata = ckpt["metadata"]
    sae_config = SAEConfig(**metadata["sae_config"])

    # Create model and restore params
    model = create_sae(sae_config)
    rng = jax.random.PRNGKey(0)
    eval_dtype = jnp.bfloat16 if sae_config.dtype == "bfloat16" else jnp.float32
    dummy = jnp.zeros((1, sae_config.hidden_dim), dtype=eval_dtype)
    variables = model.init(rng, dummy)
    params = restore_params(ckpt, variables["params"])

    # Create data source
    source_kwargs = {"layer_index": args.layer_index}
    if args.source_type == "pickle":
        source_kwargs["shard_dir"] = args.data_dir
        source_kwargs["shuffle_shards"] = False
    elif args.source_type == "numpy":
        source_kwargs = {"path": args.data_dir}
    source = create_source(args.source_type, **source_kwargs)

    # Evaluate
    all_metrics = []
    batch_count = 0

    print(f"Evaluating {sae_config.architecture} SAE (step {metadata['step']})...")
    print(f"  Dict size: {sae_config.dict_size}, Hidden dim: {sae_config.hidden_dim}")

    for batch_np in source.iter_batches(args.eval_batch_size):
        batch = jnp.array(batch_np)
        x_hat, z, _ = model.apply({"params": params}, batch)
        metrics = compute_metrics(batch, x_hat, z)
        all_metrics.append(metrics)

        batch_count += 1
        if batch_count >= args.num_batches:
            break

    if not all_metrics:
        print("No data to evaluate.")
        return

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        agg[key] = sum(values) / len(values)

    print(f"\nResults over {batch_count} batches:")
    print(f"  MSE:                {agg['mse']:.6f}")
    print(f"  Explained Variance: {agg['explained_variance']:.4f}")
    print(f"  Normalized MSE:     {agg['normalized_mse']:.6f}")
    print(f"  L0:                 {agg['l0']:.1f}")
    print(f"  L0 fraction:        {agg['l0_frac']:.4f}")
    print(f"  Dead neuron frac:   {agg['dead_neuron_frac']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"step": metadata["step"], "num_batches": batch_count, **agg}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
