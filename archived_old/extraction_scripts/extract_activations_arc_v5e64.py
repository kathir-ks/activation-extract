"""
Activation Extraction for Converted ARC Dataset on TPU v5e-64

This script extracts activations from converted ARC-format datasets
(JSONL format) and is optimized for TPU v5e-64 (multi-host).

TPU v5e-64 specs:
- 4 hosts × 8 chips/host = 32 total chips
- 16 GB HBM per chip
- Multi-host coordination via JAX distributed

Usage on v5e-64 (run on each host):
    python extract_activations_arc_v5e64.py \
        --machine_id 0 \
        --total_machines 1 \
        --multihost \
        --coordinator_address "10.0.0.1:8476" \
        --host_id 0 \
        --num_hosts 4 \
        --mesh_type 2d \
        --dataset_path arc_formatted_challenges.jsonl \
        --model_path KathirKs/qwen-2.5-7b \
        --batch_size 16 \
        --upload_to_gcs \
        --gcs_bucket your-bucket-name
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import argparse
from tqdm.auto import tqdm
import os
from pathlib import Path
import pickle
from functools import partial
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks

# Import ARC-specific modules
from arc24.data_augmentation import apply_data_augmentation, get_random_color_map, set_random_seed
from arc24.prompting import create_prompts_from_task
from arc24.encoders import create_grid_encoder

# Import multihost utilities from fineweb script
from extract_activations_fineweb_multihost import (
    ActivationStorage,
    pad_sequences,
    initialize_multihost,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded
)

# Import shard manager
from shard_manager import ShardManager, load_shard_chunks

# Alias for convenience
P = PartitionSpec


@dataclass
class ActivationExtractionConfig:
    """Configuration for activation extraction"""
    # Distributed config
    machine_id: int = 0
    total_machines: int = 1

    # Model config
    model_path: str = "KathirKs/qwen-2.5-7b"

    # Dataset config
    dataset_path: str = 'arc_formatted_challenges.jsonl'
    max_tasks: Optional[int] = None  # Max tasks to process

    # Sharded dataset support
    use_sharded_dataset: bool = False  # Use sharded dataset with auto shard claiming
    sharded_dataset_dir: Optional[str] = None  # Directory containing sharded dataset
    preferred_shard_id: Optional[int] = None  # Preferred shard ID (auto-select if None)

    # Prompt config
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    predictions_per_task: int = 8

    # Extraction config
    layers_to_extract: List[int] = None  # Will default based on model
    batch_size: int = 16  # Batch size for v5e-64
    max_seq_length: int = 2048  # Max sequence length for tokenization

    # Output config
    output_dir: str = './activations_arc_v5e64'

    # GCS upload config (REQUIRED for v5e-64)
    upload_to_gcs: bool = True
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = 'activations_arc_v5e64'
    shard_size_gb: float = 1.0
    compress_shards: bool = True
    delete_local_after_upload: bool = True

    # Multi-host TPU config (v5e-64)
    multihost: bool = False
    coordinator_address: Optional[str] = None
    host_id: int = 0
    num_hosts: int = 1
    mesh_type: str = '2d'  # '1d', '2d', or '3d'

    # Other
    random_seed: Optional[int] = 42
    verbose: bool = True
    use_data_parallel: bool = True

    def __post_init__(self):
        if self.upload_to_gcs and self.gcs_bucket is None:
            raise ValueError("gcs_bucket must be specified when upload_to_gcs=True")

        if self.multihost:
            if self.coordinator_address is None:
                raise ValueError("coordinator_address must be specified when multihost=True")
            if self.host_id >= self.num_hosts:
                raise ValueError(f"host_id ({self.host_id}) must be < num_hosts ({self.num_hosts})")

        # Create machine/host-specific GCS prefix
        if self.upload_to_gcs:
            prefix = f"{self.gcs_prefix}/machine_{self.machine_id:03d}"
            if self.multihost:
                prefix += f"_host_{self.host_id:02d}"
            self.gcs_prefix = prefix


def load_arc_dataset_from_shard(
    sharded_dataset_dir: str,
    worker_id: str,
    preferred_shard_id: Optional[int] = None,
    verbose: bool = False
) -> Tuple[Dict, int, ShardManager]:
    """
    Load ARC dataset from sharded dataset with automatic shard claiming

    Args:
        sharded_dataset_dir: Directory containing sharded dataset
        worker_id: Unique worker identifier
        preferred_shard_id: Preferred shard ID (auto-select if None)
        verbose: Print progress

    Returns:
        Tuple of (tasks_dict, shard_id, shard_manager)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading from Sharded Dataset")
        print(f"{'='*70}")
        print(f"  Dataset dir: {sharded_dataset_dir}")
        print(f"  Worker ID: {worker_id}")
        if preferred_shard_id is not None:
            print(f"  Preferred shard: {preferred_shard_id}")

    # Initialize shard manager
    shard_manager = ShardManager(sharded_dataset_dir, worker_id)

    # Claim a shard
    if verbose:
        print(f"\n  Claiming shard...")

    shard_info = shard_manager.claim_shard(preferred_shard_id)

    if shard_info is None:
        raise RuntimeError(
            f"No available shards in {sharded_dataset_dir}. "
            f"All shards may be in use or completed."
        )

    shard_id = shard_info["shard_id"]
    chunk_files = shard_info["chunks"]
    total_tasks = shard_info["metadata"]["total_tasks"]

    if verbose:
        print(f"  ✓ Claimed shard {shard_id}")
        print(f"    Tasks: {total_tasks:,}")
        print(f"    Chunks: {len(chunk_files)}")
        print(f"\n  Loading tasks from chunks...")

    # Load all tasks from chunks
    is_gcs = sharded_dataset_dir.startswith("gs://")
    task_list = load_shard_chunks(chunk_files, is_gcs)

    # Convert to dict format
    tasks = {}
    for task_obj in task_list:
        task_id = task_obj.get("task_id", f"task_{len(tasks):08x}")
        tasks[task_id] = {
            "train": task_obj["train"],
            "test": task_obj["test"]
        }

    if verbose:
        print(f"  ✓ Loaded {len(tasks):,} tasks from shard {shard_id}")
        print(f"{'='*70}\n")

    return tasks, shard_id, shard_manager


def load_arc_dataset_jsonl(dataset_path: str, max_tasks: Optional[int] = None,
                           machine_id: int = 0, total_machines: int = 1,
                           verbose: bool = False):
    """
    Load ARC dataset from JSONL format with machine-based sharding

    Args:
        dataset_path: Path to JSONL file
        max_tasks: Maximum tasks to load (per machine)
        machine_id: This machine's ID (0 to total_machines-1)
        total_machines: Total number of machines
        verbose: Print progress
    """
    if verbose:
        print(f"\nLoading ARC dataset from {dataset_path}...")
        print(f"  Machine {machine_id}/{total_machines-1}")
        print(f"  Max tasks per machine: {max_tasks if max_tasks else 'unlimited'}")

    tasks = {}
    task_count = 0

    with open(dataset_path, 'r') as f:
        for line_idx, line in enumerate(f):
            # Shard across machines (round-robin)
            if line_idx % total_machines != machine_id:
                continue

            try:
                task_obj = json.loads(line.strip())

                # Handle both formats:
                # 1. {"task_id": "...", "train": [...], "test": [...]}
                # 2. {"train": [...], "test": [...]} (generate task_id)

                if "task_id" in task_obj:
                    task_id = task_obj["task_id"]
                else:
                    task_id = f"task_{line_idx:08x}"

                # Remove task_id from task object (ARC format doesn't include it)
                task_data = {
                    "train": task_obj["train"],
                    "test": task_obj["test"]
                }

                tasks[task_id] = task_data
                task_count += 1

                if max_tasks is not None and task_count >= max_tasks:
                    break

            except json.JSONDecodeError:
                if verbose:
                    print(f"  Warning: Skipping invalid JSON line {line_idx}")
                continue

    if verbose:
        print(f"  ✓ Loaded {len(tasks)} tasks")

    return tasks


def create_prompts_from_dataset(
    tasks: Dict,
    grid_encoder,
    tokenizer,
    prompt_version: str,
    predictions_per_task: int,
    random_seed: Optional[int] = None,
    verbose: bool = False
):
    """Create prompts for all tasks with data augmentation"""
    if random_seed is not None:
        set_random_seed(random_seed)

    prompts = []

    for task_id, task in tqdm(tasks.items(), total=len(tasks),
                              desc='Creating prompts', disable=not verbose):
        # Data augmentation parameters
        from itertools import product
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        num_augmentations = len(data_augmentation_params)

        # Calculate how many times to repeat each augmentation
        repeats_per_aug = max(1, predictions_per_task // num_augmentations)

        for hflip, n_rot90 in data_augmentation_params:
            for _ in range(repeats_per_aug):
                color_map = get_random_color_map(change_background_probability=0.1)
                data_augmentation_kwargs = dict(hflip=hflip, n_rot90=n_rot90, color_map=color_map)
                augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)

                task_prompts = create_prompts_from_task(
                    augmented_task, grid_encoder=grid_encoder, tokenizer=tokenizer,
                    is_train_prompt=False, prompt_version=prompt_version
                )

                for idx, prompt in enumerate(task_prompts):
                    prompts.append({
                        'task_id': task_id,
                        'data_augmentation_kwargs': data_augmentation_kwargs,
                        'prompt': prompt,
                        'idx': idx
                    })

            # Break early if we have enough
            if len(prompts) >= predictions_per_task * len(task['test']):
                break

    return prompts


def process_batch(model, params, sequences, sample_indices, prompts_data,
                  storage, layers_to_extract, pad_token_id,
                  batch_size=None, max_seq_length=None):
    """Process a single batch of sequences with padding and optimized transfers"""
    # Pad batch dimension
    actual_batch_size = len(sequences)
    if batch_size is not None and actual_batch_size < batch_size:
        pad_count = batch_size - actual_batch_size
        sequences = sequences + [sequences[-1]] * pad_count

    # Pad sequences to fixed length
    padded = pad_sequences(sequences, pad_token_id=pad_token_id, fixed_length=max_seq_length)
    input_ids = jnp.array(padded)

    # Forward pass (JIT-compiled)
    activations = extract_activations_sharded(model, params, input_ids)

    # Vectorized async device→host transfer for all layers at once
    # This overlaps TPU computation with host transfers
    host_activations = {}
    for layer_idx in layers_to_extract:
        layer_key = f'layer_{layer_idx}'
        if layer_key in activations:
            # jax.device_get starts async transfer, returns immediately
            host_activations[layer_key] = jax.device_get(activations[layer_key])

    # Process only actual samples
    for i, sample_idx in enumerate(sample_indices):
        prompt_data = prompts_data[sample_idx]

        # Extract activations for each layer (already on host)
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in host_activations:
                layer_act = host_activations[layer_key][i]
                layer_act_np = np.array(layer_act)

                # Store with task_id and prompt
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=layer_act_np,
                    sample_idx=sample_idx,
                    text_preview=f"Task: {prompt_data['task_id']}, Prompt: {prompt_data['prompt'][:100]}"
                )


def main():
    parser = argparse.ArgumentParser(description="Extract activations from ARC dataset on v5e-64")

    # Distributed args
    parser.add_argument('--machine_id', type=int, default=0)
    parser.add_argument('--total_machines', type=int, default=1)

    # Model args
    parser.add_argument('--model_path', type=str, default="KathirKs/qwen-2.5-7b")

    # Dataset args
    parser.add_argument('--dataset_path', type=str, help="Path to JSONL dataset file")
    parser.add_argument('--max_tasks', type=int, default=None)

    # Sharded dataset args
    parser.add_argument('--use_sharded_dataset', action='store_true',
                       help="Use sharded dataset with automatic shard claiming")
    parser.add_argument('--sharded_dataset_dir', type=str,
                       help="Directory containing sharded dataset")
    parser.add_argument('--preferred_shard_id', type=int,
                       help="Preferred shard ID (auto-select if not specified)")

    # Prompt args
    parser.add_argument('--grid_encoder', type=str)
    parser.add_argument('--prompt_version', type=str)
    parser.add_argument('--predictions_per_task', type=int)

    # Extraction args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--max_seq_length', type=int, default=2048)

    # Output args
    parser.add_argument('--output_dir', type=str)

    # GCS args
    parser.add_argument('--upload_to_gcs', action='store_true')
    parser.add_argument('--gcs_bucket', type=str, required=True)
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
    parser.add_argument('--no_data_parallel', action='store_true')

    args = parser.parse_args()

    # Build config
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    if 'no_data_parallel' in config_dict:
        config_dict['use_data_parallel'] = not config_dict.pop('no_data_parallel')

    cfg = ActivationExtractionConfig(**config_dict)

    # Enable JIT compilation logging for performance verification
    jax.config.update('jax_log_compiles', True)
    if cfg.verbose:
        print("\n[Performance] JIT compilation logging enabled")
        print("[Performance] Watch for 'Compiling...' messages - should only appear once per unique shape")

    print("="*70)
    print(f"ARC ACTIVATION EXTRACTION ON TPU v5e-64 - MACHINE {cfg.machine_id}")
    if cfg.multihost:
        print(f"MULTI-HOST MODE - Host {cfg.host_id}/{cfg.num_hosts-1}")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)

    # Initialize multi-host if needed
    if cfg.multihost:
        num_devices = initialize_multihost(
            cfg.coordinator_address,
            cfg.num_hosts,
            cfg.host_id,
            cfg.verbose
        )
    else:
        devices = jax.devices()
        num_devices = len(devices)
        print(f"\nMachine {cfg.machine_id} - Found {num_devices} device(s): {[d.device_kind for d in devices]}")

    # Set random seed
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)

    # Load dataset
    shard_manager = None
    claimed_shard_id = None

    if cfg.use_sharded_dataset:
        # Load from sharded dataset
        if not cfg.sharded_dataset_dir:
            raise ValueError("--sharded_dataset_dir required when using --use_sharded_dataset")

        worker_id = f"machine{cfg.machine_id}_host{cfg.host_id if cfg.multihost else 0}"

        tasks, claimed_shard_id, shard_manager = load_arc_dataset_from_shard(
            cfg.sharded_dataset_dir,
            worker_id,
            cfg.preferred_shard_id,
            cfg.verbose
        )
    else:
        # Load from single JSONL file
        if not cfg.dataset_path:
            raise ValueError("--dataset_path required when not using sharded dataset")

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
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings
        )
        print(f"  ✓ Loaded config: {hf_config.model_type}")
        print(f"    Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")

        # Set default layers if not specified
        if cfg.layers_to_extract is None:
            # Extract from last few layers
            cfg.layers_to_extract = list(range(config.num_hidden_layers - 14, config.num_hidden_layers))
            print(f"    Using default layers: {cfg.layers_to_extract}")

    except Exception as e:
        print(f"  ⚠ Could not load config: {e}")
        raise

    # Load model and tokenizer
    print(f"\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=torch.float32,
        trust_remote_code=True
    )

    jax_model = create_model_with_hooks(config, layers_to_extract=cfg.layers_to_extract)
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    del hf_model

    # Create mesh and shard parameters
    mesh = None
    if cfg.use_data_parallel and num_devices > 1:
        print(f"\nSetting up model sharding across {num_devices} devices...")
        if cfg.multihost:
            print(f"  Multi-host: {cfg.num_hosts} hosts × {num_devices // cfg.num_hosts} devices/host")
            print(f"  Mesh type: {cfg.mesh_type}")

        mesh, _ = create_device_mesh(num_devices, cfg.mesh_type, cfg.num_hosts)
        print(f"  ✓ Mesh: {mesh.axis_names}, shape: {mesh.devices.shape}")

        sharding_rules = create_sharding_strategy(mesh)
        print(f"  ✓ Sharding strategy: {len(sharding_rules)} rules")

        print(f"  ⟳ Sharding parameters...")
        with mesh:
            params = shard_params(params, mesh, sharding_rules)
        print(f"  ✓ Parameters sharded")

    # Create grid encoder
    print(f"\nCreating grid encoder: {cfg.grid_encoder}")
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    # Create prompts
    print(f"\nCreating prompts...")
    prompts_data = create_prompts_from_dataset(
        tasks, grid_encoder, tokenizer, cfg.prompt_version,
        cfg.predictions_per_task, cfg.random_seed, cfg.verbose
    )
    print(f"  ✓ Created {len(prompts_data)} prompts")

    # Tokenize
    print("\nTokenizing prompts...")
    sequences = []
    for prompt_data in tqdm(prompts_data, desc="Tokenizing", disable=not cfg.verbose):
        inputs = tokenizer(
            prompt_data['prompt'],
            return_tensors="np",
            truncation=True,
            max_length=cfg.max_seq_length
        )
        sequences.append(inputs['input_ids'][0])

    # Create batches
    print(f"\nCreating batches (batch_size={cfg.batch_size})...")
    num_batches = (len(sequences) + cfg.batch_size - 1) // cfg.batch_size
    print(f"  ✓ {num_batches} batches")

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

    # Extract activations
    print(f"\nExtracting activations...")
    if mesh is not None:
        print(f"  Mode: Sharded ({cfg.mesh_type.upper()} mesh: {mesh.axis_names})")
    else:
        print(f"  Mode: Sequential")

    context = mesh if mesh is not None else None

    with (context if context is not None else open(os.devnull)) as _:
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
        if cfg.verbose:
            print(f"  ✓ Shard {claimed_shard_id} marked as completed")

    print("\n" + "="*70)
    print(f"MACHINE {cfg.machine_id} - EXTRACTION COMPLETE!")
    if claimed_shard_id is not None:
        print(f"  Shard processed: {claimed_shard_id}")
    print(f"  Activations saved to: {cfg.output_dir}")
    if cfg.upload_to_gcs:
        print(f"  GCS path: gs://{cfg.gcs_bucket}/{cfg.gcs_prefix}/")
    print("="*70)


if __name__ == '__main__':
    main()
