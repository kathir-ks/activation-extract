#!/usr/bin/env python3
"""
Multihost TPU Activation Extraction

For TPU pod slices (v5e-64, v5e-128, etc.) where multiple hosts
share a single model instance with sharded parameters.

Key differences from single-host extraction:
1. Uses jax.distributed for multi-host coordination
2. Data is distributed across hosts, then gathered
3. Only host 0 uploads results to GCS
4. Checkpointing is coordinated across hosts

Usage:
    # On TPU pod (all workers run same command via --worker=all)
    python multihost_extract.py \\
        --topology v5e-64 \\
        --dataset_path gs://bucket/dataset.jsonl \\
        --gcs_bucket your-bucket \\
        --upload_to_gcs

    # With explicit coordinator (for manual setup)
    python multihost_extract.py \\
        --topology v5e-64 \\
        --coordinator_address "10.0.0.1:8476" \\
        --host_id 0 \\
        --num_hosts 4 \\
        --dataset_path gs://bucket/dataset.jsonl \\
        --gcs_bucket your-bucket
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
import json
import numpy as np
import argparse
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
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
    ActivationStorage,
    P
)

# Import new multihost utilities
from core.jax_utils import (
    initialize_multihost_auto,
    get_host_info,
    distribute_data_across_hosts,
    gather_activations_to_primary,
    sync_hosts,
    is_primary_host,
    get_per_host_batch_indices,
)

from core.mesh_configs import (
    get_topology_config,
    create_mesh_for_topology,
    create_sharding_specs,
    detect_topology,
    TopologyConfig,
)

# Import model utilities
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks

# Import ARC-specific modules
from arc24.data_augmentation import set_random_seed
from arc24.encoders import create_grid_encoder
from core.dataset_utils import create_prompts_from_dataset

# Import barrier synchronization for multihost coordination
from core.barrier_sync import (
    init_barrier_sync,
    barrier,
    shutdown_barrier_sync,
    BarrierServer,
    BarrierClient,
)


@dataclass
class MultihostExtractionConfig:
    """Configuration for multihost TPU activation extraction"""
    
    # TPU Pod topology
    topology: str = 'v5e-64'  # v5e-64, v5e-128, v5e-256
    
    # Multihost settings (auto-detected if not specified)
    coordinator_address: Optional[str] = None
    host_id: Optional[int] = None
    num_hosts: Optional[int] = None
    
    # Model config
    model_path: str = "Qwen/Qwen2.5-0.5B"
    
    # Dataset config
    dataset_path: str = ""
    max_tasks: Optional[int] = None
    
    # Prompt config
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    predictions_per_task: int = 8
    
    # Extraction config
    layers_to_extract: Optional[List[int]] = None
    activation_type: str = 'residual'  # 'mlp', 'attn', or 'residual'
    batch_size: int = 32  # Global batch size across all hosts
    max_seq_length: int = 2048
    
    # Output config
    output_dir: str = './activations'
    
    # GCS upload config
    upload_to_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = 'activations/multihost'
    shard_size_gb: float = 1.0
    compress_shards: bool = True
    delete_local_after_upload: bool = False
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: str = './checkpoints'
    
    # Barrier synchronization for multihost coordination
    enable_barrier_sync: bool = True  # Enable socket-based barrier sync
    barrier_port: int = 5555
    barrier_controller_host: Optional[str] = None  # Auto-detected if None
    is_barrier_server: bool = False  # Set to True for SSH worker 0 only
    
    # Other
    random_seed: Optional[int] = 42
    verbose: bool = True
    
    def __post_init__(self):
        # Validate
        if self.upload_to_gcs and not self.gcs_bucket:
            raise ValueError("--gcs_bucket required when --upload_to_gcs is set")
        if not self.dataset_path:
            raise ValueError("--dataset_path is required")


class MultihostActivationStorage(ActivationStorage):
    """
    Storage handler for multihost TPU extraction.

    With FSDP, each host holds a shard of the activations. Every host
    runs its own ActivationStorage instance and uploads to a per-host
    GCS prefix (e.g. activations/host_00/, activations/host_01/).

    This avoids expensive cross-host gathers and lets each host stream
    its data to GCS independently.
    """

    def __init__(
        self,
        host_id: int = 0,
        num_hosts: int = 1,
        **kwargs
    ):
        self.host_id = host_id
        self.num_hosts = num_hosts
        self.is_primary = (host_id == 0)

        # Each host gets its own output_dir and gcs_prefix
        if num_hosts > 1:
            output_dir = kwargs.get('output_dir', './activations')
            kwargs['output_dir'] = os.path.join(output_dir, f'host_{host_id:02d}')

            gcs_prefix = kwargs.get('gcs_prefix', 'activations')
            kwargs['gcs_prefix'] = f"{gcs_prefix}/host_{host_id:02d}"

        # All hosts initialize storage (each writes its own shards)
        super().__init__(**kwargs)

    def finalize(self):
        """Finalize storage, then sync all hosts"""
        super().finalize()
        # Sync after all hosts have flushed their buffers
        sync_hosts("finalize_storage")


def load_checkpoint_multihost(checkpoint_dir: str, topology: str, host_id: int) -> Dict:
    """Load checkpoint file for this host if it exists"""
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{topology}_host_{host_id:02d}.json"
    )
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    return {}


def process_batch_multihost(
    model,
    params,
    sequences: List,
    sample_indices: List[int],
    prompts_data: List[Dict],
    storage: MultihostActivationStorage,
    layers_to_extract: List[int],
    pad_token_id: int,
    batch_size: int,
    max_seq_length: int,
    host_info: Dict,
    mesh: Optional[Mesh] = None,
    sharding_specs: Optional[Dict[str, NamedSharding]] = None,
):
    """Process a batch of sequences with multihost coordination"""
    
    # Pad batch dimension if needed
    actual_batch_size = len(sequences)
    if actual_batch_size < batch_size:
        pad_count = batch_size - actual_batch_size
        sequences = sequences + [sequences[-1]] * pad_count
    
    # Pad sequences to fixed length
    padded = pad_sequences(sequences, pad_token_id=pad_token_id, fixed_length=max_seq_length)
    input_ids = jnp.array(padded)
    
    # Shard input_ids along batch dimension if mesh is provided (FSDP)
    if mesh is not None and sharding_specs is not None and 'input' in sharding_specs:
        # FSDP: All workers hold a slice of the batch
        input_ids = jax.device_put(input_ids, sharding_specs['input'])
    
    # Forward pass (JIT-compiled, SPMD across all hosts)
    # JAX SPMD already synchronizes workers - no explicit barrier needed
    activations = extract_activations_sharded(model, params, input_ids)

    # Synchronize all hosts before gathering activations.
    # Without this, a fast host could start the next batch's device_put
    # while a slow host is still reading shards — causing a collective desync.
    from core.barrier_sync import barrier
    barrier("pre_gather")

    # ── FSDP path: each host extracts its own addressable shard ──────
    if mesh is not None and sharding_specs is not None:
        # Transfer this host's addressable shards to numpy
        host_activations = {}
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                # Sort shards by index to guarantee batch-dimension ordering
                local_shards = sorted(
                    activations[layer_key].addressable_shards,
                    key=lambda s: s.index,
                )
                local_data = np.concatenate(
                    [np.array(s.data) for s in local_shards], axis=0
                )
                host_activations[layer_key] = local_data

        # Map local shard rows back to global sample indices
        per_host_samples = actual_batch_size // host_info['num_hosts']
        local_start = host_info['host_id'] * per_host_samples
        local_end = min(local_start + per_host_samples, actual_batch_size)

        for i_local, global_i in enumerate(range(local_start, local_end)):
            if global_i >= len(sample_indices):
                break
            sample_idx = sample_indices[global_i]
            prompt_data = prompts_data[sample_idx]
            for layer_idx in layers_to_extract:
                layer_key = f'layer_{layer_idx}'
                if layer_key in host_activations:
                    storage.add_activation(
                        layer_idx=layer_idx,
                        activation=host_activations[layer_key][i_local],
                        sample_idx=sample_idx,
                        text_preview=f"Task: {prompt_data['task_id']}, Prompt: {prompt_data['prompt'][:100]}"
                    )

    # ── Legacy / single-host path: gather to primary ─────────────────
    else:
        gathered_activations = gather_activations_to_primary(activations)

        if gathered_activations is not None:
            host_activations = {}
            for layer_idx in layers_to_extract:
                layer_key = f'layer_{layer_idx}'
                if layer_key in gathered_activations:
                    host_activations[layer_key] = jax.device_get(
                        gathered_activations[layer_key]
                    )

            for i, sample_idx in enumerate(sample_indices):
                if i >= actual_batch_size:
                    break
                prompt_data = prompts_data[sample_idx]
                for layer_idx in layers_to_extract:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in host_activations:
                        storage.add_activation(
                            layer_idx=layer_idx,
                            activation=np.array(host_activations[layer_key][i]),
                            sample_idx=sample_idx,
                            text_preview=f"Task: {prompt_data['task_id']}, Prompt: {prompt_data['prompt'][:100]}"
                        )


def main():
    parser = argparse.ArgumentParser(description="Multihost TPU Activation Extraction")
    
    # Topology args
    parser.add_argument('--topology', type=str, default='v5e-64',
                        help="TPU topology (v5e-64, v5e-128, v5e-256)")
    
    # Multihost args (usually auto-detected)
    parser.add_argument('--coordinator_address', type=str, 
                        help="Coordinator address (auto-detected if not set)")
    parser.add_argument('--host_id', type=int, 
                        help="Host ID (auto-detected if not set)")
    parser.add_argument('--num_hosts', type=int,
                        help="Number of hosts (auto-detected if not set)")
    
    # Model args
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-0.5B")
    
    # Dataset args
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to JSONL dataset")
    parser.add_argument('--max_tasks', type=int, help="Maximum tasks to process")
    
    # Prompt args
    parser.add_argument('--grid_encoder', type=str)
    parser.add_argument('--prompt_version', type=str)
    parser.add_argument('--predictions_per_task', type=int)
    
    # Extraction args
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Global batch size across all hosts")
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--activation_type', type=str, default='residual',
                        choices=['residual', 'mlp', 'attn'])
    parser.add_argument('--max_seq_length', type=int, default=2048)
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='./activations')
    
    # GCS args
    parser.add_argument('--upload_to_gcs', action='store_true')
    parser.add_argument('--gcs_bucket', type=str)
    parser.add_argument('--gcs_prefix', type=str, default='activations/multihost')
    parser.add_argument('--shard_size_gb', type=float, default=1.0)
    parser.add_argument('--compress_shards', action='store_true', default=True)
    parser.add_argument('--delete_local_after_upload', action='store_true')
    
    # Checkpoint args
    parser.add_argument('--enable_checkpointing', action='store_true', default=True)
    parser.add_argument('--no_checkpointing', action='store_false', dest='enable_checkpointing')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    # Barrier sync args
    parser.add_argument('--enable_barrier_sync', action='store_true', default=True,
                        help="Enable socket-based barrier synchronization (default: True)")
    parser.add_argument('--no_barrier_sync', action='store_false', dest='enable_barrier_sync',
                        help="Disable socket-based barrier synchronization")
    parser.add_argument('--barrier_port', type=int, default=5555,
                        help="Port for barrier server (default: 5555)")
    parser.add_argument('--barrier_controller_host', type=str,
                        help="Barrier controller host (default: auto-detect Worker 0 IP)")
    parser.add_argument('--is_barrier_server', action='store_true', default=False,
                        help="Override: Force this worker to run the barrier server (auto-detected from CLOUD_TPU_TASK_ID if not set)")
    
    # Other args
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Build config
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    cfg = MultihostExtractionConfig(**config_dict)
    
    # Validate batch size is divisible by number of hosts (required for FSDP)
    # Note: num_hosts may not be known yet, validation will happen after initialization
    
    # =========================================================================
    # Step 0: Start barrier server BEFORE JAX init (if this is the server worker)
    # =========================================================================
    # The barrier server must start before JAX's distributed init because SSH workers
    # connect at different times and we need to synchronize them BEFORE JAX tries
    # to form a process group.
    
    from core.barrier_sync import BarrierServer, BarrierClient, get_worker_id, get_worker0_internal_ip, get_num_workers

    # Detect worker ID from environment (BEFORE any JAX initialization)
    # CRITICAL: This must happen before any JAX imports or distributed init
    # Google Cloud sets CLOUD_TPU_TASK_ID automatically when using --worker=all
    # GCE metadata agent-worker-number is the most reliable source for TPU VMs
    worker_id = get_worker_id()
    is_barrier_server = (worker_id == 0)

    # Allow explicit override via CLI flag (for manual setups)
    if args.is_barrier_server:
        is_barrier_server = True
        worker_id = 0

    # Auto-detect barrier controller host if not provided
    barrier_controller_host = cfg.barrier_controller_host
    if cfg.enable_barrier_sync and not barrier_controller_host:
        barrier_controller_host = get_worker0_internal_ip()
        if cfg.verbose:
            print(f"   Auto-detected barrier controller: {barrier_controller_host}")

    if cfg.verbose:
        print(f"\n📋 Worker ID: {worker_id}")
        print(f"   Is barrier server: {is_barrier_server}")
        print(f"   Barrier controller: {barrier_controller_host}")

    barrier_server = None
    barrier_client = None

    if cfg.enable_barrier_sync and is_barrier_server:
        # This is the designated barrier server (SSH worker 0)
        # Detect number of workers from environment before JAX init
        num_workers = get_num_workers()
        print(f"\n🚀 Starting barrier server on port {cfg.barrier_port} for {num_workers} workers...")
        barrier_server = BarrierServer(num_workers=num_workers, port=cfg.barrier_port)
        barrier_server.start_background(wait_ready=True, ready_timeout=30.0)
        print(f"✓ Barrier server ready on port {cfg.barrier_port}")
    
    # All workers create barrier client and wait at 'pre_jax_init' barrier
    if cfg.enable_barrier_sync and barrier_controller_host:
        # Give server a moment to start (important for timing)
        import time
        time.sleep(2.0)  # Small delay for all non-server workers
        
        barrier_client = BarrierClient(
            controller_host=barrier_controller_host,
            worker_id=worker_id,  # Use detected worker ID from environment
            port=cfg.barrier_port
        )
        
        # Set the global client for the barrier() convenience function
        from core import barrier_sync
        barrier_sync._barrier_client = barrier_client
        if barrier_server:
            barrier_sync._barrier_server = barrier_server
        
        print(f"⏳ Waiting at 'pre_jax_init' barrier...")
        if not barrier_client.wait_at_barrier("pre_jax_init", timeout=300):
            raise RuntimeError("Failed to synchronize at 'pre_jax_init' barrier")
        print(f"✓ Synchronized! Starting JAX init...")
    
    # =========================================================================
    # Step 1: Initialize multihost environment (NOW all workers start together)
    # =========================================================================
    
    if cfg.coordinator_address and cfg.host_id is not None and cfg.num_hosts:
        # Explicit multihost configuration
        host_info = {
            'host_id': cfg.host_id,
            'num_hosts': cfg.num_hosts,
            'coordinator_address': cfg.coordinator_address,
            'is_primary': cfg.host_id == 0,
        }
        initialize_multihost(
            cfg.coordinator_address,
            cfg.num_hosts,
            cfg.host_id,
            cfg.verbose
        )
        host_info['total_devices'] = jax.device_count()
        host_info['local_devices'] = jax.local_device_count()
    else:
        # Auto-detect from environment
        host_info = initialize_multihost_auto(verbose=cfg.verbose)
    
    # Only primary host prints detailed info
    if host_info['is_primary']:
        print("=" * 70)
        print(f"MULTIHOST TPU ACTIVATION EXTRACTION")
        print("=" * 70)
        print(f"Topology: {cfg.topology}")
        print(f"Hosts: {host_info['num_hosts']}")
        print(f"Total devices: {host_info['total_devices']}")
        print("=" * 70)
        print(json.dumps(asdict(cfg), indent=2, default=str))
        print("=" * 70)
    
    # Validate batch size is divisible by number of hosts (required for FSDP)
    if cfg.batch_size % host_info['num_hosts'] != 0:
        raise ValueError(
            f"Batch size ({cfg.batch_size}) must be divisible by number of hosts ({host_info['num_hosts']}) for FSDP. "
            f"Try batch_size={cfg.batch_size + (host_info['num_hosts'] - cfg.batch_size % host_info['num_hosts'])}"
        )
    
    # =========================================================================
    # Step 2: Create device mesh for topology
    # =========================================================================
    
    # Use auto-detected mesh (1D for single-host, 3D for multi-host)
    mesh = create_device_mesh('auto', verbose=host_info['is_primary'] and cfg.verbose)
    sharding_specs = create_sharding_strategy(mesh)
    
    # =========================================================================
    # Step 3: Load checkpoint if exists
    # =========================================================================
    
    start_sample_idx = 0
    if cfg.enable_checkpointing:
        checkpoint = load_checkpoint_multihost(
            cfg.checkpoint_dir, cfg.topology, host_info['host_id']
        )
        if checkpoint:
            start_sample_idx = checkpoint.get('last_processed_sample_idx', 0) + 1
            if host_info['is_primary']:
                print(f"\n📌 RESUMING FROM CHECKPOINT (Host {host_info['host_id']})")
                print(f"  Last processed sample: {checkpoint.get('last_processed_sample_idx', 0)}")
                print(f"  Starting from sample: {start_sample_idx}")

    # Sync after checkpoint loading
    sync_hosts("checkpoint_loaded")
    
    # =========================================================================
    # Step 4: Load dataset (each host loads full dataset, then filters)
    # =========================================================================
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"\nLoading dataset from {cfg.dataset_path}...")
    
    tasks = load_arc_dataset_jsonl(
        cfg.dataset_path,
        cfg.max_tasks,
        machine_id=0,
        total_machines=1,
        verbose=host_info['is_primary'] and cfg.verbose
    )
    
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)
    
    # =========================================================================
    # Step 5: Load model and convert to JAX
    # =========================================================================
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"\nLoading model {cfg.model_path}...")
    
    # Detect model config
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
        rope_theta=getattr(hf_config, 'rope_theta', 1000000.0),
    )
    
    # Default layers to extract
    if cfg.layers_to_extract is None:
        cfg.layers_to_extract = list(range(config.num_hidden_layers))
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"  Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        print(f"  Extracting from {len(cfg.layers_to_extract)} layers")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    
    # CRITICAL: Sync before model download
    # Model downloading from HuggingFace can take varying times on different hosts
    # Without this barrier, hosts will drift apart during download/loading
    if host_info['is_primary'] and cfg.verbose:
        print(f"\n⏳ Synchronizing hosts before model download...")
    
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        if not barrier("pre_model_download", timeout=300):
            raise RuntimeError("Failed to synchronize at 'pre_model_download' barrier")
    else:
        sync_hosts("pre_model_download")
        
    if host_info['is_primary'] and cfg.verbose:
        print(f"✓ All hosts ready, starting model download...")
    
    # Load HF model and convert (all hosts do this)
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Create JAX model with hooks
    jax_model = create_model_with_hooks(
        config, 
        layers_to_extract=cfg.layers_to_extract, 
        activation_type=cfg.activation_type
    )
    
    # Convert weights to JAX
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}
    del hf_model  # Free memory
    
    # CRITICAL: Sync all hosts after model loading using socket barrier
    # This ensures all hosts have loaded the model before any host starts executing
    # Without this, hosts that load faster will try to execute while slower hosts
    # are still loading, causing "unexpected peer in launch group" errors
    if host_info['is_primary'] and cfg.verbose:
        print(f"\n⏳ Synchronizing all hosts after model loading...")
    
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        if not barrier("model_loaded", timeout=600):  # 10 min for large models
            raise RuntimeError("Failed to synchronize at 'model_loaded' barrier")
    else:
        sync_hosts("model_loaded")
        
    if host_info['is_primary'] and cfg.verbose:
        print(f"✓ All hosts ready")
    
    # Shard parameters across mesh
    if host_info['is_primary'] and cfg.verbose:
        print(f"\nSharding parameters across {host_info['total_devices']} devices...")
    
    sharding_strategy = create_sharding_strategy(mesh)
    with mesh:
        params = shard_params(params, sharding_strategy)
    
    # Sync after sharding
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        barrier("sharding_complete", timeout=300)
    else:
        sync_hosts("sharding_complete")
    
    # =========================================================================
    # Step 6: Create prompts and tokenize
    # =========================================================================
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"\nCreating prompts...")
    
    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    prompts_data = create_prompts_from_dataset(
        tasks,
        grid_encoder,
        tokenizer,
        cfg.prompt_version,
        cfg.predictions_per_task,
        cfg.random_seed,
        verbose=host_info['is_primary'] and cfg.verbose
    )
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"Tokenizing {len(prompts_data)} prompts...")
    
    sequences = [tokenizer.encode(p['prompt']) for p in prompts_data]
    
    # =========================================================================
    # Step 7: Initialize storage (each host writes its own shards)
    # =========================================================================

    # Update GCS prefix with topology info
    gcs_prefix = f"{cfg.gcs_prefix}_{cfg.topology}"

    storage = MultihostActivationStorage(
        host_id=host_info['host_id'],
        num_hosts=host_info['num_hosts'],
        output_dir=cfg.output_dir,
        upload_to_gcs=cfg.upload_to_gcs,
        gcs_bucket=cfg.gcs_bucket,
        gcs_prefix=gcs_prefix,
        shard_size_gb=cfg.shard_size_gb,
        compress_shards=cfg.compress_shards,
        delete_local_after_upload=cfg.delete_local_after_upload,
        verbose=cfg.verbose
    )
    
    # =========================================================================
    # Step 8: Process batches with multihost coordination
    # =========================================================================
    
    if host_info['is_primary']:
        print(f"\n{'='*70}")
        print(f"EXTRACTING ACTIVATIONS")
        print(f"{'='*70}")
        print(f"  Total samples: {len(sequences)}")
        print(f"  Global batch size: {cfg.batch_size}")
        print(f"  Per-host batch size: {cfg.batch_size // host_info['num_hosts']}")
        print(f"  Starting from sample: {start_sample_idx}")
        print(f"{'='*70}")
    
    # Calculate batches for this host
    per_host_batch = cfg.batch_size // host_info['num_hosts']
    num_batches = (len(sequences) + cfg.batch_size - 1) // cfg.batch_size
    
    # CRITICAL: Sync before extraction starts
    if host_info['is_primary'] and cfg.verbose:
        print(f"\n⏳ Synchronizing hosts before extraction...")
    
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        if not barrier("extraction_start", timeout=300):
            raise RuntimeError("Failed to synchronize at 'extraction_start' barrier")
    else:
        sync_hosts("extraction_start")
    
    if host_info['is_primary'] and cfg.verbose:
        print(f"✓ All hosts synchronized, starting extraction!")
    
    last_saved_shard_count = 0
    
    with mesh:
        pbar = tqdm(
            range(num_batches), 
            desc=f"Host {host_info['host_id']}", 
            disable=not (host_info['is_primary'] and cfg.verbose)
        )
        
        for batch_idx in pbar:
            global_start = batch_idx * cfg.batch_size
            global_end = min(global_start + cfg.batch_size, len(sequences))
            
            # Skip if all samples in this batch have been processed
            if global_end <= start_sample_idx:
                continue
            
            # FSDP: All workers process the FULL batch, but sharded
            # We no longer slice the batch manually per worker
            host_start = global_start
            host_end = global_end
            
            # Everyone gets same batch - JAX handles sharding
            batch_sequences = sequences[global_start:global_end]
            batch_sample_indices = list(range(global_start, global_end))
            
            process_batch_multihost(
                jax_model, params, batch_sequences, batch_sample_indices,
                prompts_data, storage, cfg.layers_to_extract,
                tokenizer.pad_token_id or 0,
                cfg.batch_size,  # Use global batch size
                cfg.max_seq_length,
                host_info,
                mesh=mesh,
                sharding_specs=sharding_specs
            )
            
            # Checkpoint after shard creation (each host saves independently)
            if cfg.enable_checkpointing and storage.shard_count > last_saved_shard_count:
                checkpoint_data = {
                    'topology': cfg.topology,
                    'host_id': host_info['host_id'],
                    'last_processed_sample_idx': global_end - 1,
                    'total_samples_processed': global_end,
                    'total_shards': storage.shard_count,
                    'dataset_path': cfg.dataset_path,
                    'model_path': cfg.model_path,
                }
                # Each host saves its own checkpoint
                checkpoint_path = os.path.join(
                    cfg.checkpoint_dir,
                    f"checkpoint_{cfg.topology}_host_{host_info['host_id']:02d}.json"
                )
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                last_saved_shard_count = storage.shard_count

                if host_info['is_primary'] and cfg.verbose:
                    pbar.set_postfix({'shards': storage.shard_count})
    
    # =========================================================================
    # Step 9: Finalize
    # =========================================================================
    
    storage.finalize()
    
    # Save final checkpoint (each host independently)
    if cfg.enable_checkpointing:
        final_checkpoint = {
            'topology': cfg.topology,
            'host_id': host_info['host_id'],
            'last_processed_sample_idx': len(sequences) - 1,
            'total_samples_processed': len(sequences),
            'total_shards': storage.shard_count,
            'dataset_path': cfg.dataset_path,
            'model_path': cfg.model_path,
            'status': 'completed'
        }
        checkpoint_path = os.path.join(
            cfg.checkpoint_dir,
            f"checkpoint_{cfg.topology}_host_{host_info['host_id']:02d}.json"
        )
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(final_checkpoint, f, indent=2)

        # Sync after all hosts save their checkpoints
        if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
            barrier("final_checkpoint", timeout=300)
        else:
            sync_hosts("final_checkpoint")
    
    # Final sync using barrier
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        barrier("complete", timeout=300)
        shutdown_barrier_sync()
    else:
        sync_hosts("complete")
    
    if host_info['is_primary']:
        print("\n" + "=" * 70)
        print(f"✅ MULTIHOST EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Topology: {cfg.topology}")
        print(f"  Hosts: {host_info['num_hosts']}")
        print(f"  Total samples processed: {len(sequences)}")
        print(f"  Shards per host: {storage.shard_count}")
        print(f"  Total shards (all hosts): ~{storage.shard_count * host_info['num_hosts']}")
        print(f"  Activations saved to: {cfg.output_dir}")
        if cfg.upload_to_gcs:
            print(f"  GCS path: gs://{cfg.gcs_bucket}/{gcs_prefix}/host_*/")
        print("=" * 70)


if __name__ == '__main__':
    main()
