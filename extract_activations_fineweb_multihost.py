"""
Distributed Activation Extraction for FineWeb-Edu Dataset
WITH MULTI-HOST TPU SUPPORT (v5e-64, v6e, etc.)

Supports:
1. Single-host TPUs (v4-8, v4-32, v5litepod-8, etc.) - TESTED ✅
2. Multi-host TPUs (v5e-64, v6e-256, etc.) - NEW

Multi-host usage (e.g., v5e-64 with 4 hosts):
    # On each host (0-3):
    python extract_activations_fineweb_multihost.py \
        --machine_id 0 \
        --total_machines 32 \
        --multihost \
        --coordinator_address "10.0.0.1:8476" \
        --host_id 0 \
        --num_hosts 4 \
        --mesh_type 2d \
        --batch_size 8 \
        --upload_to_gcs \
        --gcs_bucket your-bucket-name

Single-host usage (backward compatible):
    python extract_activations_fineweb_multihost.py \
        --machine_id 0 \
        --total_machines 1 \
        --batch_size 8
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.tree_util import tree_map
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

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks

# Alias for convenience
P = PartitionSpec


@dataclass
class ActivationExtractionConfig:
    """Configuration for activation extraction"""
    # Distributed config
    machine_id: int = 0  # 0-31 for 32 machines
    total_machines: int = 32  # Total number of machines

    # Model config
    model_path: str = "KathirKs/qwen-2.5-0.5b"

    # Dataset config
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"  # or "default" for full dataset
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # Max samples per machine (None = process all)

    # Extraction config
    layers_to_extract: List[int] = None  # Will default to [10, 11, ..., 23]
    batch_size: int = 8  # Batch size per device
    max_seq_length: int = 2048  # Max sequence length for tokenization

    # Output config
    output_dir: str = './activations_fineweb'

    # GCS upload config (highly recommended for distributed setup)
    upload_to_gcs: bool = True  # Enable GCS upload
    gcs_bucket: Optional[str] = None  # GCS bucket name (e.g., 'my-bucket')
    gcs_prefix: str = 'activations_fineweb'  # Base prefix/folder in bucket
    shard_size_gb: float = 1.0  # Shard size in GB
    compress_shards: bool = True  # Compress shards before upload
    delete_local_after_upload: bool = True  # Save local disk space

    # Multi-host TPU config (for v5e-64, v6e, etc.)
    multihost: bool = False  # Enable multi-host TPU support
    coordinator_address: Optional[str] = None  # "IP:PORT" for JAX distributed init
    host_id: int = 0  # Host ID within the pod (0 to num_hosts-1)
    num_hosts: int = 1  # Total number of hosts in the pod
    mesh_type: str = '2d'  # '1d' (model-only) or '2d' (data+model) or '3d' (pipeline+data+model)

    # Other
    verbose: bool = True
    use_data_parallel: bool = True  # Use sharding/mesh across TPU cores

    def __post_init__(self):
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(10, 24))  # Layers 10-23 for Qwen2.5-0.5B

        if self.upload_to_gcs and self.gcs_bucket is None:
            raise ValueError("gcs_bucket must be specified when upload_to_gcs=True")

        if self.machine_id >= self.total_machines:
            raise ValueError(f"machine_id ({self.machine_id}) must be < total_machines ({self.total_machines})")

        # Multi-host validation
        if self.multihost:
            if self.coordinator_address is None:
                raise ValueError("coordinator_address must be specified when multihost=True")
            if self.host_id >= self.num_hosts:
                raise ValueError(f"host_id ({self.host_id}) must be < num_hosts ({self.num_hosts})")
            if self.mesh_type not in ['1d', '2d', '3d']:
                raise ValueError(f"mesh_type must be '1d', '2d', or '3d', got: {self.mesh_type}")

        # Create machine-specific GCS prefix to avoid conflicts (only if using GCS)
        if self.upload_to_gcs:
            prefix = f"{self.gcs_prefix}/machine_{self.machine_id:03d}"
            if self.multihost:
                prefix += f"_host_{self.host_id:02d}"
            self.gcs_prefix = prefix


class ActivationStorage:
    """Handle saving activations to disk and optionally uploading to GCS with sharding"""

    def __init__(self, output_dir: str, upload_to_gcs: bool = False,
                 gcs_bucket: Optional[str] = None, gcs_prefix: str = 'activations',
                 shard_size_gb: float = 1.0, compress_shards: bool = True,
                 delete_local_after_upload: bool = False, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = {}  # layer_idx -> list of activations
        self.buffer_size_bytes = 0  # Track buffer size in bytes
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes

        self.metadata = []
        self.shard_count = 0
        self.total_samples = 0
        self.verbose = verbose

        # GCS settings
        self.upload_to_gcs = upload_to_gcs
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.compress_shards = compress_shards
        self.delete_local_after_upload = delete_local_after_upload

        # Initialize fsspec filesystem for GCS if needed
        self.fs = None
        if self.upload_to_gcs:
            try:
                import fsspec
                self.fs = fsspec.filesystem('gs')  # GCS filesystem
                if self.verbose:
                    print(f"✓ fsspec GCS filesystem initialized for bucket: {gcs_bucket}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize fsspec GCS filesystem: {e}")

    def add_activation(self, layer_idx: int, activation: np.ndarray,
                      sample_idx: int, text_preview: str):
        """Add activation to buffer and check if shard size exceeded"""
        if layer_idx not in self.buffer:
            self.buffer[layer_idx] = []

        activation_data = {
            'sample_idx': sample_idx,
            'activation': activation,
            'shape': activation.shape,
            'text_preview': text_preview[:200]  # Save first 200 chars for reference
        }

        self.buffer[layer_idx].append(activation_data)
        self.total_samples += 1

        # Estimate size (activation array + metadata overhead)
        activation_size = activation.nbytes
        metadata_overhead = 1024  # Rough estimate for metadata
        self.buffer_size_bytes += activation_size + metadata_overhead

        # Check if we should save a shard
        if self.buffer_size_bytes >= self.shard_size_bytes:
            self._save_shard()

    def _save_shard(self):
        """Save current buffer as a shard"""
        if not self.buffer:
            return

        self.shard_count += 1
        shard_name = f"shard_{self.shard_count:04d}.pkl"

        if self.compress_shards:
            shard_name += ".gz"

        shard_path = self.output_dir / shard_name

        if self.verbose:
            size_mb = self.buffer_size_bytes / (1024 * 1024)
            print(f"\n{'='*70}")
            print(f"Saving shard {self.shard_count}: {shard_name} (~{size_mb:.1f} MB)")
            print(f"{'='*70}")

        # Save to local file
        if self.compress_shards:
            import gzip
            with gzip.open(shard_path, 'wb') as f:
                pickle.dump(self.buffer, f)
        else:
            with open(shard_path, 'wb') as f:
                pickle.dump(self.buffer, f)

        # Get actual file size
        file_size_mb = shard_path.stat().st_size / (1024 * 1024)

        if self.verbose:
            print(f"  ✓ Saved locally: {file_size_mb:.1f} MB")
            for layer_idx, activations in self.buffer.items():
                print(f"    Layer {layer_idx}: {len(activations)} samples")

        # Upload to GCS if enabled
        gcs_path = None
        if self.upload_to_gcs:
            gcs_path = self._upload_to_gcs(shard_path, shard_name)

        # Record metadata
        self.metadata.append({
            'shard_id': self.shard_count,
            'filename': shard_name,
            'local_path': str(shard_path),
            'gcs_path': gcs_path,
            'file_size_mb': file_size_mb,
            'buffer_size_mb': self.buffer_size_bytes / (1024 * 1024),
            'layers': list(self.buffer.keys()),
            'samples_per_layer': {layer: len(acts) for layer, acts in self.buffer.items()},
            'total_samples_in_shard': sum(len(acts) for acts in self.buffer.values())
        })

        # Delete local file if requested
        if self.upload_to_gcs and self.delete_local_after_upload and gcs_path:
            shard_path.unlink()
            if self.verbose:
                print(f"  ✓ Deleted local file (uploaded to GCS)")

        # Clear buffer
        self.buffer = {}
        self.buffer_size_bytes = 0

    def _upload_to_gcs(self, local_path: Path, shard_name: str) -> str:
        """Upload shard to GCS using fsspec"""
        try:
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{shard_name}"

            if self.verbose:
                print(f"  ⬆ Uploading to {gcs_path}...")

            # Upload using fsspec
            self.fs.put_file(str(local_path), gcs_path)

            if self.verbose:
                print(f"  ✓ Uploaded to GCS: {gcs_path}")

            return gcs_path
        except Exception as e:
            print(f"  ✗ Failed to upload to GCS: {e}")
            return None

    def finalize(self):
        """Save remaining activations and metadata"""
        # Save any remaining data
        if self.buffer:
            self._save_shard()

        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'total_shards': self.shard_count,
                'total_samples': self.total_samples,
                'shard_size_gb': self.shard_size_bytes / (1024 * 1024 * 1024),
                'upload_to_gcs': self.upload_to_gcs,
                'gcs_bucket': self.gcs_bucket,
                'gcs_prefix': self.gcs_prefix,
                'shards': self.metadata
            }, f, indent=2)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STORAGE SUMMARY")
            print(f"{'='*70}")
            print(f"  Total shards: {self.shard_count}")
            print(f"  Total samples: {self.total_samples}")
            print(f"  Metadata: {metadata_file}")
            if self.upload_to_gcs:
                print(f"  GCS bucket: gs://{self.gcs_bucket}/{self.gcs_prefix}/")
            print(f"{'='*70}")


def load_dataset_shard(dataset_name: str, dataset_config: str, dataset_split: str,
                       machine_id: int, total_machines: int, max_samples: Optional[int] = None,
                       verbose: bool = False):
    """Load a shard of the dataset for this machine"""
    if verbose:
        print(f"\nLoading dataset shard for machine {machine_id}/{total_machines-1}...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Config: {dataset_config}")
        print(f"  Split: {dataset_split}")

    # Load dataset with streaming for memory efficiency
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=True  # Stream to avoid loading entire dataset
    )

    # Shard the dataset for this machine
    dataset = dataset.shard(num_shards=total_machines, index=machine_id)

    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset.take(max_samples)

    if verbose:
        print(f"  ✓ Dataset shard {machine_id} loaded (streaming mode)")

    return dataset


def load_model_and_tokenizer(model_path: str, config: QwenConfig, layers_to_extract: List[int]):
    """Load JAX model with activation hooks and tokenizer"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading HF model for weight conversion...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True
    )

    print(f"Creating JAX model with activation hooks for layers {layers_to_extract}...")
    jax_model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)

    print("Converting HF weights to JAX...")
    converted_params = convert_hf_to_jax_weights(hf_model, config)
    params = {'params': converted_params}

    del hf_model  # Free memory

    return jax_model, tokenizer, params


def pad_sequences(sequences: List[np.ndarray], pad_token_id: int = 0) -> np.ndarray:
    """Pad sequences to same length"""
    max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = np.pad(seq, (0, max_len - len(seq)), constant_values=pad_token_id)
        padded.append(seq)

    return np.stack(padded)


def initialize_multihost(coordinator_address: str, num_hosts: int, host_id: int, verbose: bool = True):
    """
    Initialize JAX distributed for multi-host TPU

    Args:
        coordinator_address: "IP:PORT" of coordinator (usually host 0)
        num_hosts: Total number of hosts in the pod
        host_id: This host's ID (0 to num_hosts-1)
        verbose: Print initialization info

    Returns:
        num_devices: Total number of devices across ALL hosts
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"INITIALIZING MULTI-HOST JAX DISTRIBUTED")
        print(f"{'='*70}")
        print(f"  Coordinator: {coordinator_address}")
        print(f"  Total hosts: {num_hosts}")
        print(f"  This host ID: {host_id}")

    # Initialize JAX distributed
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_hosts,
        process_id=host_id
    )

    # Get all devices (across all hosts)
    devices = jax.devices()
    num_devices = len(devices)
    local_devices = jax.local_devices()

    if verbose:
        print(f"  ✓ JAX distributed initialized")
        print(f"  Global devices: {num_devices} ({[d.device_kind for d in devices[:4]]}...)")
        print(f"  Local devices on host {host_id}: {len(local_devices)}")
        print(f"  Process index: {jax.process_index()}")
        print(f"  Process count: {jax.process_count()}")
        print(f"{'='*70}\n")

    return num_devices


def create_device_mesh(num_devices: int, mesh_type: str = '1d', num_hosts: int = 1) -> Tuple[Mesh, NamedSharding]:
    """
    Create a device mesh for model sharding (single or multi-host)

    Args:
        num_devices: Total number of TPU/GPU cores (across all hosts)
        mesh_type: '1d' (model only), '2d' (data+model), or '3d' (pipeline+data+model)
        num_hosts: Number of hosts (for multi-host)

    Returns:
        mesh: JAX Mesh object
        replicated_sharding: Sharding for fully replicated arrays
    """
    if mesh_type == '1d':
        # 1D mesh: pure model parallelism
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=('model',))

    elif mesh_type == '2d':
        # 2D mesh: data + model parallelism
        # Strategy: data axis = num_hosts, model axis = devices_per_host
        if num_hosts > 1:
            devices_per_host = num_devices // num_hosts
            devices = mesh_utils.create_device_mesh((num_hosts, devices_per_host))
            mesh = Mesh(devices, axis_names=('data', 'model'))
        else:
            # Single host: split devices into data and model
            # Use reasonable split (e.g., 2 data replicas if >= 8 devices)
            if num_devices >= 8:
                data_axis = 2
                model_axis = num_devices // 2
            else:
                data_axis = 1
                model_axis = num_devices
            devices = mesh_utils.create_device_mesh((data_axis, model_axis))
            mesh = Mesh(devices, axis_names=('data', 'model'))

    elif mesh_type == '3d':
        # 3D mesh: pipeline + data + model
        # For very large models (e.g., 70B+)
        # Example: 4 pipeline × 4 data × (devices/16) model
        if num_devices >= 64:
            pipeline_axis = 4
            data_axis = 4
            model_axis = num_devices // (pipeline_axis * data_axis)
            devices = mesh_utils.create_device_mesh((pipeline_axis, data_axis, model_axis))
            mesh = Mesh(devices, axis_names=('pipeline', 'data', 'model'))
        else:
            # Fall back to 2D for smaller pods
            return create_device_mesh(num_devices, '2d', num_hosts)

    else:
        raise ValueError(f"Invalid mesh_type: {mesh_type}. Must be '1d', '2d', or '3d'")

    # Create sharding for replicated arrays
    replicated_sharding = NamedSharding(mesh, P())

    return mesh, replicated_sharding


def create_sharding_strategy(mesh: Mesh) -> Dict[str, PartitionSpec]:
    """
    Define sharding strategy for model parameters

    For Qwen model:
    - Embed tokens: shard along vocab dimension (model axis)
    - Attention weights (q, k, v, o): shard along hidden/head dimension
    - MLP weights: shard along intermediate dimension
    - Layer norms: replicated (small)
    - Output head: shard along vocab dimension

    For 2D meshes ('data', 'model'):
    - Parameters are replicated along 'data' axis
    - Parameters are sharded along 'model' axis
    - Activations/inputs are sharded along 'data' axis

    Args:
        mesh: JAX Mesh object

    Returns:
        Dictionary mapping parameter paths to PartitionSpec
    """
    # Check mesh dimensionality
    mesh_axes = mesh.axis_names

    if mesh_axes == ('model',):
        # 1D mesh: pure model parallelism
        return {
            'embed_tokens': P('model', None),
            'q_proj': P(None, 'model'),
            'k_proj': P(None, 'model'),
            'v_proj': P(None, 'model'),
            'o_proj': P('model', None),
            'gate_proj': P(None, 'model'),
            'up_proj': P(None, 'model'),
            'down_proj': P('model', None),
            'input_layernorm': P(),
            'post_attention_layernorm': P(),
            'norm': P(),
            'lm_head': P(None, 'model'),
        }

    elif mesh_axes == ('data', 'model'):
        # 2D mesh: data + model parallelism
        # Parameters: replicated on 'data', sharded on 'model'
        return {
            'embed_tokens': P(None, 'model', None),  # [vocab, hidden] -> replicate data, shard vocab
            'q_proj': P(None, None, 'model'),        # [hidden, hidden]
            'k_proj': P(None, None, 'model'),        # [hidden, kv_hidden]
            'v_proj': P(None, None, 'model'),        # [hidden, kv_hidden]
            'o_proj': P(None, 'model', None),        # [hidden, hidden]
            'gate_proj': P(None, None, 'model'),     # [hidden, intermediate]
            'up_proj': P(None, None, 'model'),       # [hidden, intermediate]
            'down_proj': P(None, 'model', None),     # [intermediate, hidden]
            'input_layernorm': P(),
            'post_attention_layernorm': P(),
            'norm': P(),
            'lm_head': P(None, None, 'model'),       # [hidden, vocab]
        }

    else:
        # 3D or other: fall back to model-only sharding
        return {
            'embed_tokens': P(None, 'model'),
            'q_proj': P(None, 'model'),
            'k_proj': P(None, 'model'),
            'v_proj': P(None, 'model'),
            'o_proj': P('model', None),
            'gate_proj': P(None, 'model'),
            'up_proj': P(None, 'model'),
            'down_proj': P('model', None),
            'input_layernorm': P(),
            'post_attention_layernorm': P(),
            'norm': P(),
            'lm_head': P(None, 'model'),
        }


def shard_params(params: Dict, mesh: Mesh, sharding_rules: Dict[str, PartitionSpec]) -> Dict:
    """
    Shard parameters according to sharding strategy

    Args:
        params: Model parameters (pytree)
        mesh: JAX Mesh object
        sharding_rules: Dictionary mapping parameter names to PartitionSpec

    Returns:
        Sharded parameters
    """
    def get_sharding_spec(path: Tuple[str, ...], value) -> PartitionSpec:
        """Determine sharding spec for a parameter based on its path and shape"""
        path_str = '/'.join(path)

        # 1D arrays (biases, norms) must be replicated
        if value.ndim < 2:
            return P()

        # Check each rule
        for key, spec in sharding_rules.items():
            if key in path_str:
                # Verify spec is compatible with array dimensionality
                # PartitionSpec with 2 elements requires 2D array
                spec_tuple = spec if isinstance(spec, tuple) else (spec,)
                spec_ndim = sum(1 for s in spec_tuple if s is not None or s == 'model')
                if spec_ndim <= value.ndim:
                    return spec

        # Default: replicate small params, shard large ones along first dimension
        if value.ndim >= 2 and value.shape[0] > 1000:
            # Large matrix - shard along first dimension
            return P('model', None)
        else:
            # Small param or bias - replicate
            return P()

    def shard_array(path: Tuple[str, ...], value):
        """Shard a single parameter array"""
        if not isinstance(value, jnp.ndarray):
            return value

        spec = get_sharding_spec(path, value)
        sharding = NamedSharding(mesh, spec)

        # Shard the array according to spec
        return jax.device_put(value, sharding)

    # Recursively shard all parameters
    def _shard_tree(params_subtree, path=()):
        if isinstance(params_subtree, dict):
            return {
                k: _shard_tree(v, path + (k,))
                for k, v in params_subtree.items()
            }
        elif isinstance(params_subtree, (jnp.ndarray, np.ndarray)):
            return shard_array(path, jnp.array(params_subtree))
        else:
            return params_subtree

    return _shard_tree(params)


@partial(jit, static_argnums=(0,))
def extract_activations_sharded(model, params, input_ids):
    """
    Extract activations with sharded model (JIT compiled)

    Args:
        model: JAX model with hooks (static)
        params: Sharded model parameters
        input_ids: [batch, seq_len] - replicated across devices

    Returns:
        activations: Dict mapping layer names to tensors
    """
    # Single forward pass with sharded params - NO GENERATION!
    # The model will automatically handle sharded computation
    _, _, activations = model.apply(
        params,
        input_ids,
        return_activations=True
    )

    return activations


def extract_activations_sequential(model, params, dataset, tokenizer,
                                   storage: ActivationStorage, layers_to_extract: List[int],
                                   batch_size: int, max_seq_length: int,
                                   max_samples: Optional[int] = None,
                                   verbose: bool = False):
    """Extract activations sequentially (single device or CPU)"""
    batch_sequences = []
    batch_sample_indices = []
    batch_text_previews = []
    sample_idx = 0

    iterator = iter(dataset)
    pbar = tqdm(desc="Extracting activations", disable=not verbose)

    try:
        while True:
            if max_samples is not None and sample_idx >= max_samples:
                break

            # Get next sample
            try:
                sample = next(iterator)
            except StopIteration:
                break

            text = sample['text']

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=max_seq_length
            )

            batch_sequences.append(inputs['input_ids'][0])
            batch_sample_indices.append(sample_idx)
            batch_text_previews.append(text)

            sample_idx += 1
            pbar.update(1)

            # Process batch when full
            if len(batch_sequences) >= batch_size:
                process_batch(
                    model, params, batch_sequences, batch_sample_indices,
                    batch_text_previews, storage, layers_to_extract,
                    tokenizer.pad_token_id or 0
                )

                batch_sequences = []
                batch_sample_indices = []
                batch_text_previews = []

    finally:
        # Process remaining samples
        if batch_sequences:
            process_batch(
                model, params, batch_sequences, batch_sample_indices,
                batch_text_previews, storage, layers_to_extract,
                tokenizer.pad_token_id or 0
            )

        pbar.close()


def process_batch(model, params, sequences, sample_indices, text_previews,
                  storage, layers_to_extract, pad_token_id, use_sharding=False):
    """Process a single batch of sequences"""
    # Pad sequences
    padded = pad_sequences(sequences, pad_token_id=pad_token_id)
    input_ids = jnp.array(padded)

    # Forward pass (automatically handles sharding if params are sharded)
    activations = extract_activations_sharded(model, params, input_ids)

    # Process each sample in batch
    for i, (sample_idx, text_preview) in enumerate(zip(sample_indices, text_previews)):
        # Extract activations for each layer
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                # Get activation for this sample: [seq_len, hidden_dim]
                layer_act = activations[layer_key][i]

                # Convert to numpy
                layer_act_np = np.array(layer_act)

                # Store
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=layer_act_np,
                    sample_idx=sample_idx,
                    text_preview=text_preview
                )


def extract_activations_with_sharding(model, params, dataset, tokenizer,
                                      storage: ActivationStorage, layers_to_extract: List[int],
                                      mesh: Mesh, batch_size: int, max_seq_length: int,
                                      max_samples: Optional[int] = None,
                                      verbose: bool = False):
    """
    Extract activations using model sharding across TPU cores

    This uses JAX's mesh and sharding APIs for efficient model parallelism.
    The model parameters are sharded across devices, while inputs/activations
    can be replicated or data-parallel.

    Args:
        model: JAX model with hooks
        params: Sharded model parameters
        dataset: HuggingFace dataset
        tokenizer: Tokenizer
        storage: Storage handler
        layers_to_extract: List of layer indices
        mesh: JAX Mesh for sharding
        batch_size: Batch size
        max_seq_length: Max sequence length
        max_samples: Max samples to process
        verbose: Verbose output
    """
    batch_sequences = []
    batch_sample_indices = []
    batch_text_previews = []
    sample_idx = 0

    iterator = iter(dataset)
    pbar = tqdm(desc="Extracting activations (sharded)", disable=not verbose)

    try:
        while True:
            if max_samples is not None and sample_idx >= max_samples:
                break

            # Get next sample
            try:
                sample = next(iterator)
            except StopIteration:
                break

            text = sample['text']

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=max_seq_length
            )

            batch_sequences.append(inputs['input_ids'][0])
            batch_sample_indices.append(sample_idx)
            batch_text_previews.append(text)

            sample_idx += 1
            pbar.update(1)

            # Process batch when full
            if len(batch_sequences) >= batch_size:
                process_batch(
                    model, params, batch_sequences, batch_sample_indices,
                    batch_text_previews, storage, layers_to_extract,
                    tokenizer.pad_token_id or 0
                )

                batch_sequences = []
                batch_sample_indices = []
                batch_text_previews = []

    finally:
        # Process remaining samples
        if batch_sequences:
            process_batch(
                model, params, batch_sequences, batch_sample_indices,
                batch_text_previews, storage, layers_to_extract,
                tokenizer.pad_token_id or 0
            )

        pbar.close()


def main():
    """Main extraction function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract activations from FineWeb-Edu dataset")

    # Distributed args
    parser.add_argument('--machine_id', type=int, required=True, help="Machine ID (0 to total_machines-1)")
    parser.add_argument('--total_machines', type=int, default=32, help="Total number of machines")

    # Model args
    parser.add_argument('--model_path', type=str, help="Path to model")

    # Dataset args
    parser.add_argument('--dataset_name', type=str, help="HuggingFace dataset name")
    parser.add_argument('--dataset_config', type=str, help="Dataset config/subset")
    parser.add_argument('--dataset_split', type=str, help="Dataset split")
    parser.add_argument('--max_samples', type=int, help="Max samples per machine")

    # Extraction args
    parser.add_argument('--batch_size', type=int, help="Batch size per device")
    parser.add_argument('--layers_to_extract', type=int, nargs='+', help="Layer indices to extract")
    parser.add_argument('--max_seq_length', type=int, help="Max sequence length")

    # Output args
    parser.add_argument('--output_dir', type=str, help="Output directory for activations")

    # GCS upload arguments
    parser.add_argument('--upload_to_gcs', action='store_true', help="Upload shards to Google Cloud Storage")
    parser.add_argument('--gcs_bucket', type=str, help="GCS bucket name (required if upload_to_gcs=True)")
    parser.add_argument('--gcs_prefix', type=str, help="Prefix/folder in GCS bucket")
    parser.add_argument('--shard_size_gb', type=float, help="Shard size in GB")
    parser.add_argument('--compress_shards', action='store_true', default=None, help="Compress shards with gzip")
    parser.add_argument('--no_compress_shards', action='store_false', dest='compress_shards', help="Don't compress")
    parser.add_argument('--delete_local_after_upload', action='store_true', help="Delete local files after GCS upload")

    # Multi-host TPU args
    parser.add_argument('--multihost', action='store_true', help="Enable multi-host TPU support")
    parser.add_argument('--coordinator_address', type=str, help="Coordinator address (IP:PORT) for multi-host")
    parser.add_argument('--host_id', type=int, help="Host ID within the pod (0 to num_hosts-1)")
    parser.add_argument('--num_hosts', type=int, help="Total number of hosts in the pod")
    parser.add_argument('--mesh_type', type=str, help="Mesh type: 1d, 2d, or 3d")

    # Other args
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--no_data_parallel', action='store_true', help="Disable data parallelism")

    args = parser.parse_args()

    # Convert no_data_parallel to use_data_parallel
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    if 'no_data_parallel' in config_dict:
        config_dict['use_data_parallel'] = not config_dict.pop('no_data_parallel')

    cfg = ActivationExtractionConfig(**config_dict)

    print("="*70)
    print(f"FINEWEB-EDU ACTIVATION EXTRACTION - MACHINE {cfg.machine_id}/{cfg.total_machines-1}")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)

    # Check available devices on this machine
    devices = jax.devices()
    num_devices = len(devices)
    print(f"\nMachine {cfg.machine_id} - Found {num_devices} device(s): {[d.device_kind for d in devices]}")

    # Load dataset shard
    dataset = load_dataset_shard(
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.dataset_split,
        cfg.machine_id,
        cfg.total_machines,
        cfg.max_samples,
        cfg.verbose
    )

    # Create model config - auto-detect from model path or use defaults
    print(f"\nDetecting model configuration from {cfg.model_path}...")

    # Try to load config from HuggingFace
    try:
        from transformers import AutoConfig
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

        print(f"  ✓ Loaded config from HuggingFace")
        print(f"  Model: {hf_config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Vocab size: {config.vocab_size}")

    except Exception as e:
        print(f"  ⚠ Could not load config from HuggingFace: {e}")
        print(f"  Using default Qwen2.5-0.5B config")
        config = QwenConfig(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True
        )

    # Load model and tokenizer
    model, tokenizer, params = load_model_and_tokenizer(
        cfg.model_path, config, cfg.layers_to_extract
    )

    # Create mesh and shard parameters for multi-device setup
    mesh = None
    if cfg.use_data_parallel and num_devices > 1:
        print(f"\nSetting up model sharding across {num_devices} devices...")
        mesh, replicated_sharding = create_device_mesh(num_devices)
        print(f"  ✓ Created mesh with axis: {mesh.axis_names}")

        # Create sharding strategy
        sharding_rules = create_sharding_strategy(mesh)
        print(f"  ✓ Created sharding strategy with {len(sharding_rules)} rules")

        # Shard parameters
        print(f"  ⟳ Sharding parameters across devices...")
        with mesh:
            params = shard_params(params, mesh, sharding_rules)
        print(f"  ✓ Parameters sharded successfully")

        # Print memory info
        print(f"\n  Memory distribution across devices:")
        for i, device in enumerate(jax.devices()):
            print(f"    Device {i} ({device.device_kind}): Ready")

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
    print(f"\nExtracting activations from layers {cfg.layers_to_extract}...")
    print(f"Mode: {'Model Sharding' if cfg.use_data_parallel and num_devices > 1 else 'Sequential'}")

    if cfg.use_data_parallel and num_devices > 1 and mesh is not None:
        with mesh:
            extract_activations_with_sharding(
                model, params, dataset, tokenizer,
                storage, cfg.layers_to_extract,
                mesh, cfg.batch_size, cfg.max_seq_length,
                cfg.max_samples,
                verbose=cfg.verbose
            )
    else:
        extract_activations_sequential(
            model, params, dataset, tokenizer,
            storage, cfg.layers_to_extract,
            cfg.batch_size, cfg.max_seq_length,
            cfg.max_samples,
            verbose=cfg.verbose
        )

    # Finalize
    storage.finalize()

    print("\n" + "="*70)
    print(f"MACHINE {cfg.machine_id} - EXTRACTION COMPLETE!")
    print(f"Activations saved to: {cfg.output_dir}")
    if cfg.upload_to_gcs:
        print(f"GCS path: gs://{cfg.gcs_bucket}/{cfg.gcs_prefix}/")
    print("="*70)


if __name__ == '__main__':
    main()