"""
Simple Activation Extraction for ARC-AGI Dataset
No autoregressive generation - just forward passes!

This is 10-100x faster than generation-based extraction because:
- Single forward pass per sequence (not token-by-token)
- No KV caching needed
- True batching across sequences
- Parallel processing across TPU cores
"""

import jax
import jax.numpy as jnp
from jax import pmap, jit
import json
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import argparse
from tqdm.auto import tqdm
from itertools import islice, product
import os
from pathlib import Path
import pickle
from functools import partial
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks

from arc24.data_augmentation import (
    apply_data_augmentation, get_random_color_map, set_random_seed)
from arc24.prompting import create_prompts_from_task
from arc24.encoders import create_grid_encoder


@dataclass
class ActivationExtractionConfig:
    """Configuration for activation extraction"""
    # Model config
    model_path: str = "KathirKs/qwen-2.5-0.5b"

    # Dataset config
    dataset_path: str = 'test_data_small.json'
    n_tasks: Optional[int] = None

    # Prompt config
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'
    predictions_per_task: int = 8

    # Extraction config
    layers_to_extract: List[int] = None  # Will default to [10, 11, ..., 23]
    batch_size: int = 8  # Batch size per device
    max_seq_length: int = 2048  # Max sequence length for tokenization

    # Output config
    output_dir: str = './activations_arc'
    save_every_n_batches: int = 10

    # GCS upload config
    upload_to_gcs: bool = False  # Enable GCS upload
    gcs_bucket: Optional[str] = None  # GCS bucket name (e.g., 'my-bucket')
    gcs_prefix: str = 'activations'  # Prefix/folder in bucket
    shard_size_gb: float = 1.0  # Shard size in GB
    compress_shards: bool = True  # Compress shards before upload
    delete_local_after_upload: bool = False  # Delete local files after GCS upload

    # Other
    random_seed: Optional[int] = 42
    verbose: bool = True
    use_data_parallel: bool = True  # Use pmap across devices

    def __post_init__(self):
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(10, 24))  # Layers 10-23 for Qwen2.5-0.5B

        if self.upload_to_gcs and self.gcs_bucket is None:
            raise ValueError("gcs_bucket must be specified when upload_to_gcs=True")


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
                      task_id: str, sample_idx: int, prompt: str):
        """Add activation to buffer and check if shard size exceeded"""
        if layer_idx not in self.buffer:
            self.buffer[layer_idx] = []

        activation_data = {
            'task_id': task_id,
            'sample_idx': sample_idx,
            'activation': activation,
            'shape': activation.shape,
            'prompt': prompt[:200]  # Save first 200 chars of prompt for reference
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

    def save_batch(self, force=False, save_every_n=10):
        """Legacy method for compatibility - now triggers shard check"""
        # This is now mainly used for periodic checks
        # The actual saving is triggered by buffer size
        pass

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


def create_prompts(data: Dict, grid_encoder, tokenizer, prompt_version: str,
                   predictions_per_task: int, verbose: bool = False):
    """Create prompts for all tasks with data augmentation"""
    prompts = []

    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts', disable=not verbose):
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
                    is_train_prompt=False, prompt_version=prompt_version)

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


@partial(jit, static_argnums=(0,))
def extract_activations_batch_single_device(model, params, input_ids):
    """
    Extract activations for a single batch on one device (JIT compiled)

    Args:
        model: JAX model with hooks
        params: Model parameters
        input_ids: [batch, seq_len]

    Returns:
        activations: Dict mapping layer names to tensors
    """
    # Single forward pass - NO GENERATION!
    _, _, activations = model.apply(
        params,
        input_ids,
        return_activations=True
    )

    return activations



def replicate_params(params: Dict, num_devices: int) -> Dict:
    """Replicate parameters across devices"""
    from jax.tree_util import tree_map
    return tree_map(lambda x: jnp.array([x] * num_devices), params)


@partial(pmap, static_broadcasted_argnums=(0,))
def extract_activations_pmap(model, params, input_ids):
    """
    Pmapped activation extraction - runs on each device in parallel

    Args:
        model: JAX model (static)
        params: Replicated parameters
        input_ids: [batch_per_device, seq_len]

    Returns:
        activations: Dict of activation tensors per device
    """
    return extract_activations_batch_single_device(model, params, input_ids)


def extract_activations_sequential(model, params, batches, prompts_data,
                                   storage: ActivationStorage, layers_to_extract: List[int],
                                   batch_size: int, verbose: bool = False):
    """Extract activations sequentially (single device or CPU)"""
    for batch_idx, batch in enumerate(tqdm(batches, desc="Extracting activations", disable=not verbose)):
        input_ids = batch['input_ids']

        # Forward pass
        activations = extract_activations_batch_single_device(model, params, input_ids)

        # Process each sample in batch
        for sample_idx in range(input_ids.shape[0]):
            # Calculate global sample index using fixed batch_size
            global_sample_idx = batch_idx * batch_size + sample_idx
            if global_sample_idx >= len(prompts_data):
                break

            prompt_data = prompts_data[global_sample_idx]

            # Extract activations for each layer
            for layer_idx in layers_to_extract:
                layer_key = f'layer_{layer_idx}'
                if layer_key in activations:
                    # Get activation for this sample: [seq_len, hidden_dim]
                    layer_act = activations[layer_key][sample_idx]

                    # Convert to numpy
                    layer_act_np = np.array(layer_act)

                    # Store
                    storage.add_activation(
                        layer_idx=layer_idx,
                        activation=layer_act_np,
                        task_id=prompt_data['task_id'],
                        sample_idx=global_sample_idx,
                        prompt=prompt_data['prompt']
                    )

        # Save periodically
        storage.save_batch(save_every_n=10)


def extract_activations_parallel(model, params, batches, prompts_data,
                                 storage: ActivationStorage, layers_to_extract: List[int],
                                 num_devices: int, batch_size: int, verbose: bool = False):
    """Extract activations in parallel across devices using pmap"""
    # Replicate parameters
    replicated_params = replicate_params(params, num_devices)

    for batch_idx, batch in enumerate(tqdm(batches, desc="Extracting activations (parallel)",
                                           disable=not verbose)):
        input_ids = batch['input_ids']

        # Reshape for pmap: [num_devices, batch_per_device, seq_len]
        total_batch = input_ids.shape[0]

        # Pad batch to be divisible by num_devices
        if total_batch % num_devices != 0:
            padding_needed = num_devices - (total_batch % num_devices)
            # Pad with last sample
            padding = jnp.repeat(input_ids[-1:], padding_needed, axis=0)
            input_ids = jnp.concatenate([input_ids, padding], axis=0)

        # Reshape for pmap
        batch_per_device = input_ids.shape[0] // num_devices
        input_ids_reshaped = input_ids.reshape(num_devices, batch_per_device, -1)

        # Parallel forward pass across devices
        activations = extract_activations_pmap(model, replicated_params, input_ids_reshaped)

        # activations is now: {layer_key: [num_devices, batch_per_device, seq_len, hidden_dim]}
        # Reshape back to [total_batch, seq_len, hidden_dim]
        activations_merged = {}
        for layer_key, layer_act in activations.items():
            # Reshape from [num_devices, batch_per_device, ...] to [total_batch, ...]
            shape = layer_act.shape
            merged = layer_act.reshape(-1, *shape[2:])
            # Trim padding
            merged = merged[:total_batch]
            activations_merged[layer_key] = merged

        # Process each sample
        for sample_idx in range(total_batch):
            # Calculate global sample index using fixed batch_size
            global_sample_idx = batch_idx * batch_size + sample_idx
            if global_sample_idx >= len(prompts_data):
                break

            prompt_data = prompts_data[global_sample_idx]

            # Extract activations for each layer
            for layer_idx in layers_to_extract:
                layer_key = f'layer_{layer_idx}'
                if layer_key in activations_merged:
                    # Get activation for this sample: [seq_len, hidden_dim]
                    layer_act = activations_merged[layer_key][sample_idx]

                    # Convert to numpy
                    layer_act_np = np.array(layer_act)

                    # Store
                    storage.add_activation(
                        layer_idx=layer_idx,
                        activation=layer_act_np,
                        task_id=prompt_data['task_id'],
                        sample_idx=global_sample_idx,
                        prompt=prompt_data['prompt']
                    )

        # Save periodically
        storage.save_batch(save_every_n=10)


def main():
    """Main extraction function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract activations from ARC-AGI dataset")
    parser.add_argument('--model_path', type=str, help="Path to model")
    parser.add_argument('--dataset_path', type=str, help="Path to ARC dataset")
    parser.add_argument('--output_dir', type=str, help="Output directory for activations")
    parser.add_argument('--n_tasks', type=int, help="Number of tasks to process")
    parser.add_argument('--batch_size', type=int, help="Batch size per device")
    parser.add_argument('--layers_to_extract', type=int, nargs='+', help="Layer indices to extract")
    parser.add_argument('--predictions_per_task', type=int, help="Predictions per task")
    parser.add_argument('--random_seed', type=int, help="Random seed")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--no_data_parallel', action='store_true', help="Disable data parallelism")

    # GCS upload arguments
    parser.add_argument('--upload_to_gcs', action='store_true', help="Upload shards to Google Cloud Storage")
    parser.add_argument('--gcs_bucket', type=str, help="GCS bucket name (required if upload_to_gcs=True)")
    parser.add_argument('--gcs_prefix', type=str, help="Prefix/folder in GCS bucket (default: activations)")
    parser.add_argument('--shard_size_gb', type=float, help="Shard size in GB (default: 1.0)")
    parser.add_argument('--compress_shards', action='store_true', default=None, help="Compress shards with gzip")
    parser.add_argument('--no_compress_shards', action='store_false', dest='compress_shards', help="Don't compress shards")
    parser.add_argument('--delete_local_after_upload', action='store_true', help="Delete local files after GCS upload")

    args = parser.parse_args()

    # Convert no_data_parallel to use_data_parallel
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    if 'no_data_parallel' in config_dict:
        config_dict['use_data_parallel'] = not config_dict.pop('no_data_parallel')

    cfg = ActivationExtractionConfig(**config_dict)

    print("="*70)
    print("ARC-AGI ACTIVATION EXTRACTION (FORWARD PASS ONLY)")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)

    # Check available devices
    devices = jax.devices()
    num_devices = len(devices)
    print(f"\nFound {num_devices} device(s): {[d.device_kind for d in devices]}")

    # Set random seed
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)

    # Load dataset
    print(f"\nLoading dataset from {cfg.dataset_path}...")
    with open(cfg.dataset_path) as f:
        data = json.load(f)

    if cfg.n_tasks is not None and cfg.n_tasks > 0:
        data = dict(islice(data.items(), cfg.n_tasks))

    print(f"Loaded {len(data)} tasks")

    # Create model config
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

    # Create grid encoder
    print(f"\nCreating grid encoder: {cfg.grid_encoder}")
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    # Create prompts
    print(f"\nCreating prompts...")
    prompts_data = create_prompts(
        data, grid_encoder, tokenizer, cfg.prompt_version,
        cfg.predictions_per_task, verbose=cfg.verbose
    )
    print(f"Created {len(prompts_data)} prompts")

    # Tokenize prompts
    print("\nTokenizing prompts...")
    tokenized_sequences = []
    for prompt_data in tqdm(prompts_data, desc="Tokenizing", disable=not cfg.verbose):
        inputs = tokenizer(
            prompt_data['prompt'],
            return_tensors="np",
            truncation=True,
            max_length=cfg.max_seq_length
        )
        tokenized_sequences.append(inputs['input_ids'][0])

    # Create batches
    print(f"\nCreating batches (batch_size={cfg.batch_size})...")
    batches = []
    for i in range(0, len(tokenized_sequences), cfg.batch_size):
        batch_sequences = tokenized_sequences[i:i + cfg.batch_size]

        # Pad to same length
        padded = pad_sequences(batch_sequences, pad_token_id=tokenizer.pad_token_id or 0)

        batches.append({
            'input_ids': jnp.array(padded)
        })

    print(f"Created {len(batches)} batches")

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
    print(f"Mode: {'Data Parallel' if cfg.use_data_parallel and num_devices > 1 else 'Sequential'}")

    if cfg.use_data_parallel and num_devices > 1:
        extract_activations_parallel(
            model, params, batches, prompts_data,
            storage, cfg.layers_to_extract,
            num_devices, cfg.batch_size,
            verbose=cfg.verbose
        )
    else:
        extract_activations_sequential(
            model, params, batches, prompts_data,
            storage, cfg.layers_to_extract,
            cfg.batch_size,
            verbose=cfg.verbose
        )

    # Finalize
    storage.finalize()

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print(f"Activations saved to: {cfg.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
