#!/usr/bin/env python3
"""
Activation Extraction for ARC Dataset on TPU

Refactored version using shared core utilities.

This script extracts activations from ARC-format datasets (JSONL) and is optimized
for single or multi-host TPU processing.

Usage:
    # Single JSONL file
    python extract_activations.py \
        --dataset_path dataset.jsonl \
        --model_path KathirKs/qwen-2.5-0.5b \
        --gcs_bucket your-bucket \
        --upload_to_gcs

    # Sharded dataset (automatic shard claiming)
    python extract_activations.py \
        --use_sharded_dataset \
        --sharded_dataset_dir gs://bucket/sharded_dataset \
        --model_path KathirKs/qwen-2.5-0.5b \
        --gcs_bucket your-bucket \
        --upload_to_gcs

    # Multi-host TPU v5e-64
    python extract_activations.py \
        --use_sharded_dataset \
        --sharded_dataset_dir gs://bucket/sharded_dataset \
        --multihost --num_hosts 4 --host_id 0 \
        --coordinator_address "10.0.0.1:8476" \
        --model_path KathirKs/qwen-2.5-0.5b \
        --gcs_bucket your-bucket
"""

import jax
import jax.numpy as jnp
import json
import numpy as np
import argparse
import os
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Import core utilities
from core import (
    initialize_multihost,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded,
    pad_sequences,
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    create_prompts_from_dataset,
    ActivationStorage,
    P
)

# Import model utilities
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks

# Import ARC-specific modules
from arc24.data_augmentation import set_random_seed
from arc24.encoders import create_grid_encoder


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction"""
    # Distributed config
    machine_id: int = 0
    total_machines: int = 1

    # Model config
    model_path: str = "KathirKs/qwen-2.5-0.5b"

    # Dataset config
    dataset_path: Optional[str] = None
    max_tasks: Optional[int] = None

    # Sharded dataset support
    use_sharded_dataset: bool = False
    sharded_dataset_dir: Optional[str] = None
    preferred_shard_id: Optional[int] = None

    # Prompt config
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    predictions_per_task: int = 8

    # Extraction config
    layers_to_extract: Optional[List[int]] = None
    batch_size: int = 4
    max_seq_length: int = 2048

    # Output config
    output_dir: str = './activations'

    # GCS upload config
    upload_to_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = 'activations'
    shard_size_gb: float = 1.0
    compress_shards: bool = True
    delete_local_after_upload: bool = False

    # Multi-host TPU config
    multihost: bool = False
    coordinator_address: Optional[str] = None
    host_id: int = 0
    num_hosts: int = 1
    mesh_type: str = '2d'

    # Other
    random_seed: Optional[int] = 42
    verbose: bool = True

    def __post_init__(self):
        # Validate configuration
        if self.use_sharded_dataset and not self.sharded_dataset_dir:
            raise ValueError("--sharded_dataset_dir required when using --use_sharded_dataset")
        if not self.use_sharded_dataset and not self.dataset_path:
            raise ValueError("--dataset_path required when not using sharded dataset")
        if self.upload_to_gcs and not self.gcs_bucket:
            raise ValueError("--gcs_bucket required when --upload_to_gcs is set")

        # Update GCS prefix with machine/host info if needed
        if self.upload_to_gcs:
            prefix = self.gcs_prefix
            if self.total_machines > 1:
                prefix += f"_machine_{self.machine_id:02d}"
            if self.multihost:
                prefix += f"_host_{self.host_id:02d}"
            self.gcs_prefix = prefix


def process_batch(
    model,
    params,
    sequences: List,
    sample_indices: List[int],
    prompts_data: List[Dict],
    storage: ActivationStorage,
    layers_to_extract: List[int],
    pad_token_id: int,
    batch_size: Optional[int] = None,
    max_seq_length: Optional[int] = None
):
    """Process a single batch of sequences"""
    # Pad batch dimension if needed
    actual_batch_size = len(sequences)
    if batch_size is not None and actual_batch_size < batch_size:
        pad_count = batch_size - actual_batch_size
        sequences = sequences + [sequences[-1]] * pad_count

    # Pad sequences to fixed length
    padded = pad_sequences(sequences, pad_token_id=pad_token_id, fixed_length=max_seq_length)
    input_ids = jnp.array(padded)

    # Forward pass (JIT-compiled)
    activations = extract_activations_sharded(model, params, input_ids)

    # Async device→host transfer for all layers
    host_activations = {}
    for layer_idx in layers_to_extract:
        layer_key = f'layer_{layer_idx}'
        if layer_key in activations:
            host_activations[layer_key] = jax.device_get(activations[layer_key])

    # Process only actual samples
    for i, sample_idx in enumerate(sample_indices):
        prompt_data = prompts_data[sample_idx]
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in host_activations:
                layer_act = host_activations[layer_key][i]
                layer_act_np = np.array(layer_act)
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=layer_act_np,
                    sample_idx=sample_idx,
                    text_preview=f"Task: {prompt_data['task_id']}, Prompt: {prompt_data['prompt'][:100]}"
                )


def main():
    parser = argparse.ArgumentParser(description="Extract activations from ARC dataset")

    # Distributed args
    parser.add_argument('--machine_id', type=int, default=0)
    parser.add_argument('--total_machines', type=int, default=1)

    # Model args
    parser.add_argument('--model_path', type=str, default="KathirKs/qwen-2.5-0.5b")

    # Dataset args
    parser.add_argument('--dataset_path', type=str, help="Path to JSONL dataset")
    parser.add_argument('--max_tasks', type=int, help="Maximum tasks to process")

    # Sharded dataset args
    parser.add_argument('--use_sharded_dataset', action='store_true')
    parser.add_argument('--sharded_dataset_dir', type=str)
    parser.add_argument('--preferred_shard_id', type=int)

    # Prompt args
    parser.add_argument('--grid_encoder', type=str)
    parser.add_argument('--prompt_version', type=str)
    parser.add_argument('--predictions_per_task', type=int)

    # Extraction args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--max_seq_length', type=int, default=2048)

    # Output args
    parser.add_argument('--output_dir', type=str)

    # GCS args
    parser.add_argument('--upload_to_gcs', action='store_true')
    parser.add_argument('--gcs_bucket', type=str)
    parser.add_argument('--gcs_prefix', type=str)
    parser.add_argument('--shard_size_gb', type=float)
    parser.add_argument('--compress_shards', action='store_true', default=None)
    parser.add_argument('--no_compress_shards', action='store_false', dest='compress_shards')
    parser.add_argument('--delete_local_after_upload', action='store_true')

    # Multi-host args
    parser.add_argument('--multihost', action='store_true')
    parser.add_argument('--coordinator_address', type=str)
    parser.add_argument('--host_id', type=int)
    parser.add_argument('--num_hosts', type=int)
    parser.add_argument('--mesh_type', type=str)

    # Other args
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Build config
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    cfg = ExtractionConfig(**config_dict)

    # Enable JIT logging
    jax.config.update('jax_log_compiles', True)

    print("="*70)
    print(f"ACTIVATION EXTRACTION - MACHINE {cfg.machine_id}")
    if cfg.multihost:
        print(f"MULTI-HOST MODE - Host {cfg.host_id}/{cfg.num_hosts-1}")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)

    # Initialize multi-host if needed
    if cfg.multihost:
        initialize_multihost(
            cfg.coordinator_address,
            cfg.num_hosts,
            cfg.host_id,
            cfg.verbose
        )

    # Set random seed
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)

    # Load dataset
    shard_manager = None
    claimed_shard_id = None

    if cfg.use_sharded_dataset:
        worker_id = f"machine{cfg.machine_id}_host{cfg.host_id if cfg.multihost else 0}"
        tasks, claimed_shard_id, shard_manager = load_arc_dataset_from_shard(
            cfg.sharded_dataset_dir,
            worker_id,
            cfg.preferred_shard_id,
            cfg.verbose
        )
    else:
        tasks = load_arc_dataset_jsonl(
            cfg.dataset_path,
            cfg.max_tasks,
            cfg.machine_id,
            cfg.total_machines,
            cfg.verbose
        )

    # Auto-detect model config
    print(f"\nDetecting model configuration from {cfg.model_path}...")
    try:
        hf_config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
        config = QwenConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
        )
        print(f"  ✓ Detected {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

        # Default layers to extract
        if cfg.layers_to_extract is None:
            cfg.layers_to_extract = list(range(config.num_hidden_layers))
            print(f"  ✓ Extracting from all {len(cfg.layers_to_extract)} layers")
        else:
            print(f"  ✓ Extracting from {len(cfg.layers_to_extract)} specified layers")
    except Exception as e:
        raise RuntimeError(f"Failed to detect model config: {e}")

    # Load model
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    print(f"Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    print(f"Creating JAX model with hooks...")
    jax_model = create_model_with_hooks(config, layers_to_extract=cfg.layers_to_extract)

    print(f"Converting weights to JAX...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    del hf_model

    # Create device mesh if multi-host
    mesh = None
    if cfg.multihost:
        mesh = create_device_mesh(cfg.mesh_type, cfg.verbose)
        sharding_strategy = create_sharding_strategy(mesh)
        print(f"\nSharding parameters...")
        params = shard_params(params, sharding_strategy)

    # Create grid encoder
    print(f"\nCreating grid encoder: {cfg.grid_encoder}")
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    # Create prompts
    print(f"\nCreating prompts...")
    prompts_data = create_prompts_from_dataset(
        tasks,
        grid_encoder,
        tokenizer,
        cfg.prompt_version,
        cfg.predictions_per_task,
        cfg.random_seed,
        cfg.verbose
    )

    # Tokenize
    print(f"\nTokenizing {len(prompts_data)} prompts...")
    sequences = [tokenizer.encode(p['prompt']) for p in prompts_data]

    # Initialize storage
    storage = ActivationStorage(
        output_dir=cfg.output_dir,
        upload_to_gcs=cfg.upload_to_gcs,
        gcs_bucket=cfg.gcs_bucket,
        gcs_prefix=cfg.gcs_prefix,
        shard_size_gb=cfg.shard_size_gb,
        compress_shards=cfg.compress_shards,
        delete_local_after_upload=cfg.delete_local_after_upload,
        verbose=cfg.verbose
    )

    # Process batches
    print(f"\nExtracting activations...")
    num_batches = (len(sequences) + cfg.batch_size - 1) // cfg.batch_size

    context = mesh if mesh is not None else open(os.devnull)
    with context:
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", disable=not cfg.verbose):
            start_idx = batch_idx * cfg.batch_size
            end_idx = min(start_idx + cfg.batch_size, len(sequences))

            batch_sequences = sequences[start_idx:end_idx]
            batch_sample_indices = list(range(start_idx, end_idx))

            process_batch(
                jax_model, params, batch_sequences, batch_sample_indices,
                prompts_data, storage, cfg.layers_to_extract,
                tokenizer.pad_token_id or 0,
                cfg.batch_size, cfg.max_seq_length
            )

    # Finalize
    storage.finalize()

    # Mark shard as completed if using sharded dataset
    if shard_manager is not None and claimed_shard_id is not None:
        if cfg.verbose:
            print(f"\nMarking shard {claimed_shard_id} as completed...")
        shard_manager.mark_completed(claimed_shard_id)

    print("\n" + "="*70)
    print(f"EXTRACTION COMPLETE!")
    if claimed_shard_id is not None:
        print(f"  Shard processed: {claimed_shard_id}")
    print(f"  Activations saved to: {cfg.output_dir}")
    if cfg.upload_to_gcs:
        print(f"  GCS path: gs://{cfg.gcs_bucket}/{cfg.gcs_prefix}/")
    print("="*70)


if __name__ == '__main__':
    main()
