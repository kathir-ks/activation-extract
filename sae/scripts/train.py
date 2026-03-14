#!/usr/bin/env python3
"""CLI entry point for SAE training.

Example usage:

    # Vanilla SAE on pickle shards (from activation-extract pipeline)
    python -m sae.scripts.train \
        --architecture vanilla \
        --hidden_dim 896 \
        --dict_size 14336 \
        --source_type pickle \
        --data_dir ./activations/tpu_0 \
        --layer_index 12 \
        --batch_size 4096 \
        --num_steps 100000

    # TopK SAE on numpy files
    python -m sae.scripts.train \
        --architecture topk \
        --hidden_dim 896 \
        --dict_size 14336 \
        --k 64 \
        --source_type numpy \
        --data_dir ./activations/layer_12.npy \
        --batch_size 4096

    # Using a preset
    python -m sae.scripts.train --preset qwen-0.5b-vanilla --data_dir ./activations
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on pre-extracted activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset (overrides individual args)
    parser.add_argument("--preset", type=str, default=None,
                        help="Use a preset config (e.g., 'qwen-0.5b-vanilla')")

    # SAE architecture
    sae = parser.add_argument_group("SAE Architecture")
    sae.add_argument("--architecture", type=str, default="vanilla",
                     choices=["vanilla", "topk", "gated", "jumprelu"])
    sae.add_argument("--hidden_dim", type=int, default=896)
    sae.add_argument("--dict_size", type=int, default=14336)
    sae.add_argument("--l1_coeff", type=float, default=5e-3)
    sae.add_argument("--k", type=int, default=64, help="Top-k value (topk arch)")
    sae.add_argument("--dtype", type=str, default="bfloat16",
                     choices=["float32", "bfloat16"])
    sae.add_argument("--normalize_decoder", action="store_true", default=True)
    sae.add_argument("--no_normalize_decoder", action="store_false",
                     dest="normalize_decoder")

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument("--source_type", type=str, default="pickle",
                      choices=["pickle", "numpy", "safetensors", "hf_dataset"])
    data.add_argument("--data_dir", type=str, required=True,
                      help="Path to activation data (dir or file)")
    data.add_argument("--layer_index", type=int, default=12)
    data.add_argument("--tensor_name", type=str, default="activations",
                      help="Tensor key for safetensors/npz")
    data.add_argument("--hf_dataset", type=str, default=None,
                      help="HuggingFace dataset name (for hf_dataset source)")
    data.add_argument("--hf_column", type=str, default="activations")

    # Training
    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size", type=int, default=4096)
    train.add_argument("--num_steps", type=int, default=100_000)
    train.add_argument("--learning_rate", type=float, default=3e-4)
    train.add_argument("--lr_warmup_steps", type=int, default=1000)
    train.add_argument("--lr_decay", type=str, default="cosine",
                       choices=["cosine", "constant", "linear"])
    train.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "adamw"])
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--shuffle_buffer_size", type=int, default=262144)

    # Dead neurons
    train.add_argument("--dead_neuron_resample_steps", type=int, default=25000)
    train.add_argument("--dead_neuron_window", type=int, default=10000)

    # Logging & checkpointing
    io = parser.add_argument_group("Logging & Checkpointing")
    io.add_argument("--log_every", type=int, default=100)
    io.add_argument("--eval_every", type=int, default=1000)
    io.add_argument("--log_dir", type=str, default="./sae_logs")
    io.add_argument("--checkpoint_dir", type=str, default="./sae_checkpoints")
    io.add_argument("--checkpoint_every", type=int, default=5000)
    io.add_argument("--keep_last_n_checkpoints", type=int, default=3)
    io.add_argument("--no_resume", action="store_true",
                     help="Don't resume from checkpoint")

    # Distributed
    dist = parser.add_argument_group("Distributed")
    dist.add_argument("--mesh_type", type=str, default="auto",
                      choices=["auto", "1d", "data_parallel"])
    dist.add_argument("--enable_barrier_sync", action="store_true",
                      help="Enable socket barrier sync for multi-host SSH launch")
    dist.add_argument("--barrier_port", type=int, default=5555,
                      help="Port for barrier sync server")
    dist.add_argument("--barrier_controller_host", type=str, default=None,
                      help="IP of barrier controller (auto-detect if not set)")

    # Backend
    backend = parser.add_argument_group("Backend")
    backend.add_argument("--backend", type=str, default=None,
                         choices=["cpu", "tpu", "gpu"],
                         help="Force JAX backend (default: auto-detect)")

    # GCS
    gcs = parser.add_argument_group("GCS Data")
    gcs.add_argument("--gcs_path", type=str, default=None,
                     help="GCS path to pickle shards (e.g. gs://bucket/activations/host_00)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Force JAX backend before any JAX imports/operations
    if args.backend:
        import os
        os.environ["JAX_PLATFORMS"] = args.backend

    # Barrier sync BEFORE JAX init (prevents SSH stagger issues)
    barrier_client = None
    barrier_server = None
    if args.enable_barrier_sync:
        import os
        from core.barrier_sync import (
            BarrierServer, BarrierClient,
            get_worker0_internal_ip, get_worker_id, get_num_workers,
        )
        worker_id = get_worker_id()
        num_workers = get_num_workers()
        controller_host = args.barrier_controller_host or get_worker0_internal_ip()

        # Worker 0 starts the barrier server
        if worker_id == 0:
            barrier_server = BarrierServer(num_workers=num_workers, port=args.barrier_port)
            barrier_server.start_background()

        barrier_client = BarrierClient(
            controller_host=controller_host,
            worker_id=worker_id,
            port=args.barrier_port,
        )
        barrier_client.wait_at_barrier("pre_jax_init")

    from sae.configs.base import SAEConfig
    from sae.configs.training import TrainingConfig
    from sae.training.trainer import SAETrainer
    from sae.training.distributed import initialize_distributed

    # Initialize distributed (auto-detects single vs multi-host)
    host_info = initialize_distributed(verbose=True)

    if barrier_client:
        barrier_client.wait_at_barrier("post_jax_init")

    # Build configs
    if args.preset:
        from sae.configs.presets import PRESETS
        if args.preset not in PRESETS:
            print(f"Unknown preset: {args.preset}. Available: {list(PRESETS.keys())}")
            sys.exit(1)
        sae_config, train_config = PRESETS[args.preset](layer=args.layer_index)
        # Override data source
        train_config.source_kwargs = _build_source_kwargs(args)
        train_config.source_type = args.source_type
    else:
        sae_config = SAEConfig(
            hidden_dim=args.hidden_dim,
            dict_size=args.dict_size,
            architecture=args.architecture,
            l1_coeff=args.l1_coeff,
            k=args.k,
            dtype=args.dtype,
            normalize_decoder=args.normalize_decoder,
        )
        train_config = TrainingConfig(
            source_type=args.source_type,
            source_kwargs=_build_source_kwargs(args),
            layer_index=args.layer_index,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_decay=args.lr_decay,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            shuffle_buffer_size=args.shuffle_buffer_size,
            dead_neuron_resample_steps=args.dead_neuron_resample_steps,
            dead_neuron_window=args.dead_neuron_window,
            log_every=args.log_every,
            eval_every=args.eval_every,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            keep_last_n_checkpoints=args.keep_last_n_checkpoints,
            mesh_type=args.mesh_type,
        )

    # Create trainer and run
    trainer = SAETrainer(sae_config, train_config)
    trainer.setup(resume=not args.no_resume)
    trainer.train()

    # Clean up barrier sync
    if barrier_client:
        barrier_client.wait_at_barrier("training_complete")
    if barrier_server:
        barrier_server.stop()


def _build_source_kwargs(args) -> dict:
    """Build source constructor kwargs from CLI args."""
    if args.source_type == "pickle":
        kwargs = {
            "shard_dir": args.data_dir,
            "layer_index": args.layer_index,
        }
        if args.gcs_path:
            kwargs["gcs_path"] = args.gcs_path
        return kwargs
    elif args.source_type == "numpy":
        return {"path": args.data_dir}
    elif args.source_type == "safetensors":
        return {
            "path": args.data_dir,
            "tensor_name": args.tensor_name,
        }
    elif args.source_type == "hf_dataset":
        return {
            "dataset_name": args.hf_dataset or args.data_dir,
            "column": args.hf_column,
        }
    return {}


if __name__ == "__main__":
    main()
