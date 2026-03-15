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
    P,
    create_dynamic_batches,
    pad_batch_to_bucket,
    DynamicBatch,
    create_grid_chunks_from_dataset,
    save_chunks_cache,
    load_chunks_cache,
    get_chunk_cache_path,
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
    max_seq_length: int = 5120  # Fixed sequence length for grid chunks
    pipeline: str = 'prompt'  # 'prompt' or 'grid_chunking'
    fsdp_size: Optional[int] = None  # FSDP axis size (default: min(2, local_devices))
    
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
    checkpoint_gcs_bucket: Optional[str] = None   # GCS bucket for durable checkpoints
    checkpoint_gcs_prefix: str = 'checkpoints'    # GCS prefix for checkpoint files
    
    # Barrier synchronization for multihost coordination
    enable_barrier_sync: bool = True  # Enable socket-based barrier sync
    barrier_port: int = 5555
    barrier_controller_host: Optional[str] = None  # Auto-detected if None
    is_barrier_server: bool = False  # Set to True for SSH worker 0 only
    
    # Other
    random_seed: Optional[int] = 42
    verbose: bool = True
    
    # Stream mode (process multiple streams sequentially)
    stream_mode: bool = False
    stream_manifest: Optional[str] = None      # GCS/local path to manifest
    stream_range_start: Optional[int] = None   # First stream ID for this pod
    stream_range_end: Optional[int] = None     # Last stream ID for this pod
    pod_id: Optional[str] = None               # Pod identifier for manifest
    
    def __post_init__(self):
        # Validate
        if self.upload_to_gcs and not self.gcs_bucket:
            raise ValueError("--gcs_bucket required when --upload_to_gcs is set")
        if not self.dataset_path and not self.stream_mode:
            raise ValueError("--dataset_path is required (or use --stream_mode)")
        if self.stream_mode and not self.stream_manifest:
            raise ValueError("--stream_manifest required when --stream_mode is set")


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


def _checkpoint_filename(topology: str, host_id: int) -> str:
    """Consistent checkpoint filename across save/load."""
    return f"checkpoint_{topology}_host_{host_id:02d}.json"


def load_checkpoint_multihost(
    checkpoint_dir: str,
    topology: str,
    host_id: int,
    gcs_bucket: Optional[str] = None,
    gcs_prefix: str = 'checkpoints',
) -> Dict:
    """Load checkpoint file for this host, with GCS fallback.

    Tries local first, then GCS. This ensures recovery after preemption
    (which destroys local disk) as long as checkpoints were persisted to GCS.
    """
    filename = _checkpoint_filename(topology, host_id)

    # Try local first
    local_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                data = json.load(f)
                data['_source'] = 'local'
                return data
        except Exception as e:
            print(f"Warning: Failed to load local checkpoint: {e}")

    # Fall back to GCS
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}/{filename}"
        try:
            import fsspec
            fs = fsspec.filesystem('gs')
            if fs.exists(gcs_path):
                with fs.open(gcs_path, 'r') as f:
                    data = json.load(f)
                    data['_source'] = 'gcs'
                    print(f"  📥 Loaded checkpoint from GCS: {gcs_path}")
                    return data
        except Exception as e:
            print(f"Warning: Failed to load GCS checkpoint: {e}")

    return {}


def save_checkpoint_multihost(
    checkpoint_data: Dict,
    checkpoint_dir: str,
    topology: str,
    host_id: int,
    gcs_bucket: Optional[str] = None,
    gcs_prefix: str = 'checkpoints',
):
    """Save checkpoint locally AND to GCS for preemption safety."""
    filename = _checkpoint_filename(topology, host_id)

    # Save locally
    os.makedirs(checkpoint_dir, exist_ok=True)
    local_path = os.path.join(checkpoint_dir, filename)
    with open(local_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Save to GCS
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}/{filename}"
        try:
            import fsspec
            fs = fsspec.filesystem('gs')
            with fs.open(gcs_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint to GCS: {e}")



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
    actual_lengths: Optional[List[int]] = None,
):
    """Process a batch of sequences with multihost coordination.

    Args:
        actual_lengths: If provided, only store activations for the first
            actual_lengths[i] tokens of each sequence (skip padding tokens).
    """

    # Pad batch dimension if needed
    actual_batch_size = len(sequences)
    if actual_batch_size < batch_size:
        pad_count = batch_size - actual_batch_size
        sequences = sequences + [sequences[-1]] * pad_count
        if actual_lengths is not None:
            actual_lengths = actual_lengths + [actual_lengths[-1]] * pad_count

    # Pad sequences to fixed length (bucket size or max_seq_length)
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
        # Gather activations along model axis so each host gets full hidden_dim.
        # Without this, each host only has a slice (e.g. 448 of 896 with model=2).
        from jax.sharding import PartitionSpec as P
        data_axis = mesh.axis_names[0]
        act_sharding = NamedSharding(mesh, P(data_axis, None, None))

        # Transfer this host's addressable shards to numpy
        host_activations = {}
        local_shard_indices = None
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                # Re-shard to P(data, None, None) — triggers all-gather on model axis
                full_act = jax.device_put(activations[layer_key], act_sharding)
                # After re-sharding, local devices may hold identical replicas
                # (e.g. 4 chips on one host all have the same data shard).
                # Deduplicate by taking only unique shard indices.
                local_shards = full_act.addressable_shards
                seen_indices = {}
                for s in local_shards:
                    if s.index not in seen_indices:
                        seen_indices[s.index] = s
                unique_shards = sorted(seen_indices.values(), key=lambda s: s.index)
                local_data = np.concatenate(
                    [np.array(s.data) for s in unique_shards], axis=0
                )
                host_activations[layer_key] = local_data
                # Capture shard indices from the first layer
                if local_shard_indices is None:
                    local_shard_indices = sorted(set(s.index for s in local_shards))

        # Map local shard rows back to global sample indices.
        # Use shard index tuples to determine which data-axis positions
        # this host owns, rather than assuming host_id maps to data position.
        first_layer_key = f'layer_{layers_to_extract[0]}'
        n_local_samples = host_activations[first_layer_key].shape[0]
        # Shard index tuple's first element is the batch slice (data axis position)
        local_batch_positions = sorted(set(idx[0] for idx in local_shard_indices))
        # Each data position corresponds to (batch_size / data_axis_size) samples
        data_axis_size = mesh.shape[data_axis]
        samples_per_position = actual_batch_size // data_axis_size
        global_indices = []
        for pos in local_batch_positions:
            start = pos * samples_per_position
            end = min(start + samples_per_position, actual_batch_size)
            global_indices.extend(range(start, end))

        # Log diagnostic info on first batch
        if storage.shard_count == 0 and not hasattr(process_batch_multihost, '_logged'):
            import sys
            print(f"[Host {host_info['host_id']}] FSDP extraction diagnostics:")
            print(f"  Local shards: {n_local_samples} samples, hidden_dim={host_activations[first_layer_key].shape[-1]}")
            print(f"  Data axis positions: {local_batch_positions}")
            print(f"  Global sample indices: {global_indices}")
            sys.stdout.flush()
            process_batch_multihost._logged = True

        for i_local, global_i in enumerate(global_indices):
            if global_i >= len(sample_indices):
                break
            sample_idx = sample_indices[global_i]
            prompt_data = prompts_data[sample_idx]
            for layer_idx in layers_to_extract:
                layer_key = f'layer_{layer_idx}'
                if layer_key in host_activations:
                    act = host_activations[layer_key][i_local]
                    # Crop padding: only store activations for actual tokens
                    if actual_lengths is not None and global_i < len(actual_lengths):
                        seq_len = actual_lengths[global_i]
                        act = act[:seq_len]
                    storage.add_activation(
                        layer_idx=layer_idx,
                        activation=act,
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
                        act = np.array(host_activations[layer_key][i])
                        # Crop padding: only store activations for actual tokens
                        if actual_lengths is not None and i < len(actual_lengths):
                            seq_len = actual_lengths[i]
                            act = act[:seq_len]
                        storage.add_activation(
                            layer_idx=layer_idx,
                            activation=act,
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
    parser.add_argument('--dataset_path', type=str, default="",
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
    parser.add_argument('--max_seq_length', type=int, default=5120,
                        help="Fixed sequence length for chunks/batches (default: 5120)")
    parser.add_argument('--pipeline', type=str, default='prompt',
                        choices=['prompt', 'grid_chunking'],
                        help="Data pipeline: 'prompt' (full prompts) or 'grid_chunking' (grid-only chunks for SAE)")
    parser.add_argument('--fsdp_size', type=int, default=None,
                        help="FSDP axis size for 3D mesh (default: min(2, local_devices))")

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
    parser.add_argument('--checkpoint_gcs_bucket', type=str, default=None,
                        help="GCS bucket for durable checkpoints (survives preemption)")
    parser.add_argument('--checkpoint_gcs_prefix', type=str, default='checkpoints',
                        help="GCS prefix for checkpoint files")
    
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
    
    # Stream mode args
    parser.add_argument('--stream_mode', action='store_true', default=False,
                        help="Process multiple streams sequentially from manifest")
    parser.add_argument('--stream_manifest', type=str, default=None,
                        help="GCS/local path to stream manifest JSON")
    parser.add_argument('--stream_range_start', type=int, default=None,
                        help="First stream ID assigned to this pod")
    parser.add_argument('--stream_range_end', type=int, default=None,
                        help="Last stream ID assigned to this pod")
    parser.add_argument('--pod_id', type=str, default=None,
                        help="Pod identifier for manifest tracking")
    
    args = parser.parse_args()
    
    # Build config
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    # Allow empty dataset_path in stream_mode
    if args.stream_mode and not args.dataset_path:
        config_dict.setdefault('dataset_path', '')
    cfg = MultihostExtractionConfig(**config_dict)
    
    # =========================================================================
    # Stream mode: process multiple streams sequentially
    # =========================================================================
    if cfg.stream_mode:
        from core.stream_manager import StreamManager
        sm = StreamManager(cfg.stream_manifest, verbose=cfg.verbose)
        
        stream_range = None
        if cfg.stream_range_start is not None and cfg.stream_range_end is not None:
            stream_range = (cfg.stream_range_start, cfg.stream_range_end)
        
        pod_id = cfg.pod_id or cfg.topology
        stream_count = 0
        
        while True:
            stream = sm.claim_next_stream(pod_id, stream_range)
            if stream is None:
                break
            
            stream_count += 1
            stream_id = stream['stream_id']
            dataset_path = stream['dataset_path']
            
            print(f"\n{'='*70}")
            print(f"STREAM MODE: Processing stream {stream_id} ({dataset_path})")
            print(f"{'='*70}")
            
            # Override dataset_path for this stream
            cfg.dataset_path = dataset_path
            
            try:
                run_extraction(cfg)
                sm.mark_stream_complete(stream_id)
            except Exception as e:
                print(f"\n❌ Stream {stream_id} failed: {e}")
                import traceback
                traceback.print_exc()
                # Don't mark complete — it stays in_progress for retry
                break
        
        if stream_count > 0:
            status = sm.get_status_summary()
            print(f"\n✅ Stream mode complete. Processed {stream_count} streams.")
            print(f"   Overall: {status['completed']}/{status['total']} completed ({status['pct_complete']}%)")
        return
    
    # Single-stream mode (original behavior)
    run_extraction(cfg)


def run_extraction(cfg):
    """Run extraction for a single dataset with the given config."""
    
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
    if cfg.is_barrier_server:
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
    mesh = create_device_mesh('auto', verbose=host_info['is_primary'] and cfg.verbose, fsdp_size=cfg.fsdp_size)
    sharding_specs = create_sharding_strategy(mesh)
    
    # =========================================================================
    # Step 3: Load checkpoint if exists
    # =========================================================================
    
    start_sample_idx = 0
    resume_shard_count = 0
    resume_activation_count = 0
    if cfg.enable_checkpointing:
        # Use --gcs_bucket as fallback for checkpoint GCS bucket if not set explicitly
        ckpt_gcs_bucket = cfg.checkpoint_gcs_bucket or cfg.gcs_bucket
        checkpoint = load_checkpoint_multihost(
            cfg.checkpoint_dir, cfg.topology, host_info['host_id'],
            gcs_bucket=ckpt_gcs_bucket,
            gcs_prefix=cfg.checkpoint_gcs_prefix,
        )
        if checkpoint and checkpoint.get('status') != 'completed':
            start_sample_idx = checkpoint.get('last_processed_sample_idx', 0) + 1
            resume_shard_count = checkpoint.get('total_shards', 0)
            resume_activation_count = checkpoint.get('total_activations', 0)
            if host_info['is_primary']:
                source = checkpoint.get('_source', 'unknown')
                print(f"\n📌 RESUMING FROM CHECKPOINT (Host {host_info['host_id']}, source: {source})")
                print(f"  Last processed sample: {checkpoint.get('last_processed_sample_idx', 0)}")
                print(f"  Starting from sample: {start_sample_idx}")
                print(f"  Resuming from shard: {resume_shard_count}")
        elif checkpoint and checkpoint.get('status') == 'completed':
            if host_info['is_primary']:
                print(f"\n✅ Checkpoint shows extraction already completed. Skipping.")
            # Use a sentinel so the barrier below can verify ALL hosts agree
            start_sample_idx = -1  # signals "completed"

    # Validate all hosts resume from the same point.
    # Each host uses start_sample_idx in the barrier name — if any host has a
    # different value (e.g. stale local checkpoint vs fresh GCS checkpoint),
    # the barrier will timeout because not all hosts reach the same barrier name.
    if host_info['is_primary']:
        print(f"  Checkpoint sync: host {host_info['host_id']} resuming from sample {start_sample_idx}")
    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        barrier_name = f"checkpoint_loaded_at_{start_sample_idx}"
        passed = barrier(barrier_name, timeout=120)
        if not passed:
            msg = (
                f"\n{'='*60}\n"
                f"CHECKPOINT MISMATCH DETECTED (Host {host_info['host_id']})\n"
                f"{'='*60}\n"
                f"This host wants to resume from sample {start_sample_idx},\n"
                f"but other hosts are at a different point.\n\n"
                f"This usually means one host loaded a stale local checkpoint\n"
                f"while others loaded from GCS (or vice versa).\n\n"
                f"Fix: Delete local checkpoints on all hosts and retry:\n"
                f"  rm -rf {cfg.checkpoint_dir}/checkpoint_*.json\n"
                f"The pipeline will then load from GCS consistently.\n"
                f"{'='*60}\n"
            )
            print(msg, flush=True)
            raise RuntimeError(f"Checkpoint mismatch: host {host_info['host_id']} at sample {start_sample_idx}")
    else:
        sync_hosts("checkpoint_loaded")

    # If ALL hosts agreed on "completed", exit now (after the barrier)
    if start_sample_idx == -1:
        return

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
        dtype=jnp.bfloat16,
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
        dtype=torch.bfloat16,
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
    # Step 6: Create prompts/chunks and tokenize
    # =========================================================================

    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    if cfg.pipeline == 'grid_chunking':
        # Grid chunking pipeline: strip text, continuous grid token stream, fixed chunks
        if host_info['is_primary'] and cfg.verbose:
            print(f"\nUsing grid chunking pipeline (chunk_size={cfg.max_seq_length})")

        # Try loading from cache first (saves ~2 hours on restart/recovery)
        ckpt_gcs_bucket = cfg.checkpoint_gcs_bucket or cfg.gcs_bucket
        cache_path = get_chunk_cache_path(
            gcs_bucket=ckpt_gcs_bucket,
            gcs_prefix=cfg.checkpoint_gcs_prefix,
            task_ids=list(tasks.keys()),
            chunk_size=cfg.max_seq_length,
            predictions_per_task=cfg.predictions_per_task,
            random_seed=cfg.random_seed,
        )

        if host_info['is_primary'] and cfg.verbose:
            print(f"  Chunk cache path: {cache_path}")

        cached = load_chunks_cache(
            cache_path,
            verbose=host_info['is_primary'] and cfg.verbose,
        )

        if cached is not None:
            chunks, chunk_metadata, stream_metadata = cached
            if host_info['is_primary'] and cfg.verbose:
                print(f"  Using cached chunks ({len(chunks)} chunks, skipped data pipeline)")
        else:
            if host_info['is_primary'] and cfg.verbose:
                print(f"  No cache found, building chunks from scratch...")
            chunks, chunk_metadata, stream_metadata = create_grid_chunks_from_dataset(
                tasks=tasks,
                grid_encoder=grid_encoder,
                tokenizer=tokenizer,
                chunk_size=cfg.max_seq_length,
                predictions_per_task=cfg.predictions_per_task,
                random_seed=cfg.random_seed,
                verbose=host_info['is_primary'] and cfg.verbose,
            )
            # Primary host saves cache for future restarts
            if host_info['is_primary'] and ckpt_gcs_bucket:
                try:
                    import sys
                    print(f"  Saving chunk cache ({len(chunks)} chunks) to {cache_path}...")
                    sys.stdout.flush()
                    save_chunks_cache(
                        chunks, chunk_metadata, stream_metadata,
                        cache_path, verbose=cfg.verbose,
                    )
                    print(f"  Chunk cache saved successfully")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  Warning: Failed to save chunk cache: {e}")
                    import traceback; traceback.print_exc()
                    sys.stdout.flush()

        sequences = chunks
        prompts_data = [
            {'task_id': f'chunk_{m.chunk_idx}', 'prompt': f'grid_chunk_{m.chunk_idx}'}
            for m in chunk_metadata
        ]
        dynamic_batches = None
        use_fixed_batching = True
    else:
        # Standard prompt pipeline
        if host_info['is_primary'] and cfg.verbose:
            print(f"\nCreating prompts...")

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

        # Dynamic batching: sort by length, group into buckets
        if host_info['is_primary'] and cfg.verbose:
            print(f"\nCreating dynamic batches (max_seq_length={cfg.max_seq_length})...")

        dynamic_batches, sorted_sequences, sorted_prompts = create_dynamic_batches(
            sequences,
            prompts_data,
            max_seq_length=cfg.max_seq_length,
            num_hosts=host_info['num_hosts'],
            verbose=host_info['is_primary'] and cfg.verbose,
        )

        sequences = sorted_sequences
        prompts_data = sorted_prompts
        use_fixed_batching = False

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
        verbose=cfg.verbose,
        resume_from_shard=resume_shard_count,
        resume_from_activations=resume_activation_count,
    )

    # =========================================================================
    # Step 8: Process batches with multihost coordination
    # =========================================================================

    if use_fixed_batching:
        total_sequences = len(sequences)
        num_batches = (total_sequences + cfg.batch_size - 1) // cfg.batch_size
    else:
        total_sequences = sum(len(b.sequences) for b in dynamic_batches)
        num_batches = len(dynamic_batches)

    if host_info['is_primary']:
        pipeline_label = "grid chunking, fixed batching" if use_fixed_batching else "dynamic batching"
        print(f"\n{'='*70}")
        print(f"EXTRACTING ACTIVATIONS ({pipeline_label})")
        print(f"{'='*70}")
        print(f"  Total samples: {total_sequences}")
        print(f"  Total batches: {num_batches}")
        print(f"  Starting from sample: {start_sample_idx}")
        print(f"{'='*70}")

    # CRITICAL: Sync before extraction starts
    if host_info['is_primary'] and cfg.verbose:
        print(f"\n  Synchronizing hosts before extraction...")

    if cfg.enable_barrier_sync and host_info['num_hosts'] > 1:
        if not barrier("extraction_start", timeout=1800):
            raise RuntimeError("Failed to synchronize at 'extraction_start' barrier")
    else:
        sync_hosts("extraction_start")

    if host_info['is_primary'] and cfg.verbose:
        print(f"  All hosts synchronized, starting extraction!")

    last_saved_shard_count = 0
    global_sample_offset = 0  # Tracks position in the sorted sequence list

    def _save_checkpoint_if_needed(global_end, pbar=None, bucket_size=None):
        nonlocal last_saved_shard_count
        if cfg.enable_checkpointing and storage.shard_count > last_saved_shard_count:
            ckpt_gcs_bucket = cfg.checkpoint_gcs_bucket or cfg.gcs_bucket
            save_checkpoint_multihost(
                checkpoint_data={
                    'topology': cfg.topology,
                    'host_id': host_info['host_id'],
                    'last_processed_sample_idx': global_end - 1,
                    'total_samples_processed': global_end,
                    'total_shards': storage.shard_count,
                    'total_activations': storage.total_activations,
                    'dataset_path': cfg.dataset_path,
                    'model_path': cfg.model_path,
                },
                checkpoint_dir=cfg.checkpoint_dir,
                topology=cfg.topology,
                host_id=host_info['host_id'],
                gcs_bucket=ckpt_gcs_bucket,
                gcs_prefix=cfg.checkpoint_gcs_prefix,
            )
            last_saved_shard_count = storage.shard_count
            if host_info['is_primary'] and cfg.verbose and pbar:
                postfix = {'shards': storage.shard_count}
                if bucket_size:
                    postfix['bucket'] = bucket_size
                pbar.set_postfix(postfix)

    with mesh:
        if use_fixed_batching:
            # Grid chunking: simple fixed-size batching
            pbar = tqdm(
                range(num_batches),
                desc=f"Host {host_info['host_id']}",
                disable=not (host_info['is_primary'] and cfg.verbose)
            )
            for batch_idx in pbar:
                global_start = batch_idx * cfg.batch_size
                global_end = min(global_start + cfg.batch_size, total_sequences)
                batch_sequences = sequences[global_start:global_end]

                if global_end <= start_sample_idx:
                    continue

                batch_sample_indices = list(range(global_start, global_end))

                process_batch_multihost(
                    jax_model, params, batch_sequences, batch_sample_indices,
                    prompts_data, storage, cfg.layers_to_extract,
                    tokenizer.pad_token_id or 0,
                    cfg.batch_size,
                    cfg.max_seq_length,
                    host_info,
                    mesh=mesh,
                    sharding_specs=sharding_specs,
                    actual_lengths=None,
                )
                _save_checkpoint_if_needed(global_end, pbar)
        else:
            # Dynamic batching for standard prompt pipeline
            pbar = tqdm(
                dynamic_batches,
                desc=f"Host {host_info['host_id']}",
                disable=not (host_info['is_primary'] and cfg.verbose)
            )
            for batch in pbar:
                global_start = global_sample_offset
                global_end = global_start + len(batch.sequences)
                global_sample_offset = global_end

                if global_end <= start_sample_idx:
                    continue

                batch_sample_indices = list(range(global_start, global_end))

                process_batch_multihost(
                    jax_model, params, batch.sequences, batch_sample_indices,
                    prompts_data, storage, cfg.layers_to_extract,
                    tokenizer.pad_token_id or 0,
                    batch.batch_size,
                    batch.bucket_size,
                    host_info,
                    mesh=mesh,
                    sharding_specs=sharding_specs,
                    actual_lengths=batch.actual_lengths,
                )
                _save_checkpoint_if_needed(global_end, pbar, batch.bucket_size)
    
    # =========================================================================
    # Step 9: Finalize
    # =========================================================================
    
    storage.finalize()
    
    # Save final checkpoint (each host independently)
    if cfg.enable_checkpointing:
        ckpt_gcs_bucket = cfg.checkpoint_gcs_bucket or cfg.gcs_bucket
        save_checkpoint_multihost(
            checkpoint_data={
                'topology': cfg.topology,
                'host_id': host_info['host_id'],
                'last_processed_sample_idx': len(sequences) - 1,
                'total_samples_processed': len(sequences),
                'total_shards': storage.shard_count,
                'total_activations': storage.total_activations,
                'dataset_path': cfg.dataset_path,
                'model_path': cfg.model_path,
                'status': 'completed'
            },
            checkpoint_dir=cfg.checkpoint_dir,
            topology=cfg.topology,
            host_id=host_info['host_id'],
            gcs_bucket=ckpt_gcs_bucket,
            gcs_prefix=cfg.checkpoint_gcs_prefix,
        )

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
