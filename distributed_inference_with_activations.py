"""
Distributed Inference Pipeline with Activation Extraction for TPU Pods
Supports multi-host TPU (v4-64, v5e-64, etc.) with layer activation capture
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
import argparse
from tqdm.auto import tqdm
from itertools import islice, product
import os
from functools import partial
from pathlib import Path
import pickle

from transformers import AutoTokenizer
from qwen2_jax import QwenModel, QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import QwenModelWithActivations, create_model_with_hooks
from transformers import AutoModelForCausalLM
import torch

from arc24.data_augmentation import (
    apply_data_augmentation, revert_data_augmentation, get_random_color_map, set_random_seed)
from arc24.prompting import parse_grid_from_response, print_smallest_prompt, create_prompts_from_task
from arc24.encoders import create_grid_encoder


@dataclass
class DistributedARCConfig:
    """Configuration for distributed ARC-AGI inference with activation extraction"""
    # Output paths
    output_filepath: str = 'submission.json'
    activations_dir: str = './activations'

    # Model config
    model_path: str = "Qwen/Qwen2.5-0.5B"
    max_model_len: int = 10240
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    prompt_version: str = 'output-from-examples-v0'

    # Dataset config
    dataset_path: str = 'arc_data.json'
    n_tasks: Optional[int] = None

    # Inference params
    max_output_tokens: int = 1100
    predictions_per_task: int = 8
    temperature: float = 0.0
    batch_size: int = 8
    random_seed: Optional[int] = None

    # Distributed config
    mesh_shape: Tuple[int, int] = (1, 1)  # (data, model) parallelism
    use_pjit: bool = True

    # Activation extraction config
    extract_activations: bool = True
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))  # Layers 10-23
    save_every_n_batches: int = 10
    upload_to_cloud: bool = False
    cloud_bucket: Optional[str] = None  # e.g., 'gs://my-bucket/activations'

    verbose: bool = False


def setup_mesh(mesh_shape: Tuple[int, int]):
    """
    Setup JAX mesh for distributed computation

    Args:
        mesh_shape: (data_parallelism, model_parallelism)
    """
    # Get available devices
    devices = jax.devices()
    n_devices = len(devices)

    print(f"Found {n_devices} devices: {devices}")

    # Create device mesh
    device_array = np.array(devices).reshape(mesh_shape)

    # Create mesh with named axes
    mesh = Mesh(device_array, axis_names=('data', 'model'))

    print(f"Created mesh with shape {mesh_shape}: {mesh}")

    return mesh


def get_param_sharding_spec(param_name: str, param_shape: tuple, mesh: Mesh) -> NamedSharding:
    """
    Get sharding specification for a model parameter

    Args:
        param_name: Name of the parameter (e.g., 'embed_tokens', 'layers_0_attn_qkv')
        param_shape: Shape of the parameter
        mesh: JAX mesh

    Returns:
        NamedSharding specification
    """
    # For transformer models, we shard along hidden/intermediate dimensions
    # across the 'model' axis, and replicate across 'data' axis

    if 'embed' in param_name or 'lm_head' in param_name:
        # Embedding layers: [vocab_size, hidden_size]
        # Shard hidden_size across 'model' axis
        if len(param_shape) >= 2:
            spec = P(None, 'model')
        else:
            spec = P(None)
    elif 'norm' in param_name:
        # Norm layers: [hidden_size] - replicate
        spec = P(None)
    elif len(param_shape) == 2:
        # Linear layers: [in_features, out_features]
        # Shard second dimension across 'model' axis
        spec = P(None, 'model')
    else:
        # Default: replicate everything
        spec = P(None)

    return NamedSharding(mesh, spec)


def shard_params(params: Dict, mesh: Mesh) -> Dict:
    """
    Shard model parameters according to mesh

    Args:
        params: Model parameters pytree
        mesh: JAX mesh

    Returns:
        Sharded parameters
    """
    def shard_leaf(path, param):
        param_name = '/'.join(str(k) for k in path)
        sharding = get_param_sharding_spec(param_name, param.shape, mesh)
        return jax.device_put(param, sharding)

    # Use tree_map_with_path to get parameter names
    from jax.tree_util import tree_map_with_path
    sharded_params = tree_map_with_path(shard_leaf, params)

    return sharded_params


def get_data_sharding(mesh: Mesh, batch_shape: tuple) -> NamedSharding:
    """
    Get sharding specification for input data

    Args:
        mesh: JAX mesh
        batch_shape: Shape of the batch [batch_size, seq_len]

    Returns:
        NamedSharding that shards batch dimension across 'data' axis
    """
    # Shard batch dimension across 'data' axis, replicate sequence dimension
    spec = P('data', None)
    return NamedSharding(mesh, spec)


class ActivationExtractor:
    """Extract and store activations from specific layers"""

    def __init__(self, config: DistributedARCConfig):
        self.config = config
        self.activations_buffer = {}
        self.batch_count = 0

        # Create activations directory
        os.makedirs(config.activations_dir, exist_ok=True)

        # Initialize metadata
        self.metadata = {
            'layers_extracted': config.layers_to_extract,
            'model_path': config.model_path,
            'batch_size': config.batch_size,
            'batches': []
        }

    def extract_layer_activation(self, layer_idx: int, activation: jnp.ndarray,
                                task_id: str, sample_idx: int):
        """
        Extract activation from a specific layer

        Args:
            layer_idx: Layer index
            activation: Activation tensor [batch, seq_len, hidden_size]
            task_id: Task ID
            sample_idx: Sample index within task
        """
        if layer_idx not in self.config.layers_to_extract:
            return

        # Convert to numpy for storage
        activation_np = np.array(activation)

        # Store in buffer
        key = f"layer_{layer_idx}"
        if key not in self.activations_buffer:
            self.activations_buffer[key] = []

        self.activations_buffer[key].append({
            'task_id': task_id,
            'sample_idx': sample_idx,
            'activation': activation_np,
            'shape': activation_np.shape
        })

    def save_activations(self, force=False):
        """Save activations buffer to disk"""
        self.batch_count += 1

        if not force and self.batch_count % self.config.save_every_n_batches != 0:
            return

        if not self.activations_buffer:
            return

        # Save each layer's activations
        for layer_key, activations in self.activations_buffer.items():
            filename = f"{layer_key}_batch_{self.batch_count:06d}.pkl"
            filepath = os.path.join(self.config.activations_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(activations, f)

            print(f"Saved {len(activations)} activations to {filepath}")

            # Update metadata
            self.metadata['batches'].append({
                'batch_id': self.batch_count,
                'layer': layer_key,
                'file': filename,
                'n_samples': len(activations)
            })

        # Upload to cloud if configured
        if self.config.upload_to_cloud and self.config.cloud_bucket:
            self.upload_to_cloud_storage()

        # Clear buffer
        self.activations_buffer = {}

    def upload_to_cloud_storage(self):
        """Upload activations to cloud storage (GCS)"""
        if not self.config.cloud_bucket:
            return

        try:
            from google.cloud import storage

            bucket_name = self.config.cloud_bucket.replace('gs://', '').split('/')[0]
            prefix = '/'.join(self.config.cloud_bucket.replace('gs://', '').split('/')[1:])

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Upload all files in activations_dir
            for filename in os.listdir(self.config.activations_dir):
                if filename.endswith('.pkl'):
                    local_path = os.path.join(self.config.activations_dir, filename)
                    blob_path = os.path.join(prefix, filename)

                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_path)

                    print(f"Uploaded {filename} to {self.config.cloud_bucket}/{blob_path}")

        except Exception as e:
            print(f"Warning: Failed to upload to cloud storage: {e}")

    def finalize(self):
        """Finalize and save metadata"""
        # Save any remaining activations
        self.save_activations(force=True)

        # Save metadata
        metadata_path = os.path.join(self.config.activations_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Saved activation metadata to {metadata_path}")


def create_model_with_activation_hooks(config: QwenConfig, layers_to_extract: List[int]):
    """
    Create model with activation extraction hooks

    This modifies the model to capture intermediate layer activations
    """
    # Use the QwenModelWithActivations from qwen2_jax_with_hooks.py
    model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)
    return model


def generate_with_activations(
    model, params, input_ids,
    max_tokens: int,
    activation_extractor: ActivationExtractor,
    task_id: str,
    sample_idx: int
):
    """
    Generate tokens while extracting activations

    Args:
        model: JAX model
        params: Model parameters
        input_ids: Input token IDs
        max_tokens: Maximum tokens to generate
        activation_extractor: Activation extractor instance
        task_id: Task ID for tracking
        sample_idx: Sample index for tracking
    """
    generated_ids = input_ids

    for step in range(max_tokens):
        # Forward pass with activation extraction
        logits, activations = model.apply(params, generated_ids, return_activations=True)

        # Extract activations from intermediate layers
        for layer_name, layer_activations in activations.items():
            # Extract layer number from name (e.g., 'layer_12' -> 12)
            if layer_name.startswith('layer_'):
                layer_idx_str = layer_name.replace('layer_', '').replace('_norm', '')
                try:
                    layer_idx = int(layer_idx_str)
                    if layer_idx in activation_extractor.config.layers_to_extract:
                        # Extract last token position (the one being predicted)
                        last_token_activation = layer_activations[:, -1, :]  # [batch, hidden_dim]
                        activation_extractor.extract_layer_activation(
                            layer_idx, last_token_activation, task_id, sample_idx
                        )
                except ValueError:
                    # Skip non-numeric layer names
                    pass

        # Sample next token (greedy decoding)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)

        # Check for stopping condition
        if generated_ids.shape[1] > input_ids.shape[1] + max_tokens:
            break

    return generated_ids


def create_mesh_aware_generate(model, max_tokens, mesh):
    """
    Create a mesh-aware distributed generation function using jax.jit with KV caching

    This version uses KV caching for 8-10x speedup:
    - Prefill: Process entire prompt once
    - Decode: Process one token at a time with cached K/V

    Uses the mesh for 2D parallelism (data + model)
    """
    # Define sharding for inputs and outputs
    # Input: [batch, seq_len] - shard batch across 'data' axis
    input_sharding = NamedSharding(mesh, P('data', None))
    # KV cache sharding: replicate for now (could shard across model axis)
    kv_cache_sharding = None  # Will be replicated

    def prefill_jit(params, input_ids):
        """Prefill: Process entire prompt and initialize KV cache"""
        logits, kv_caches = model.apply(params, input_ids,
                                       kv_caches=None,
                                       position_offset=0,
                                       return_activations=False)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        return next_token_id, kv_caches

    def decode_step_jit(params, input_id, kv_caches, position):
        """Decode: Process single token with KV cache (FAST!)"""
        logits, new_kv_caches = model.apply(params, input_id,
                                           kv_caches=kv_caches,
                                           position_offset=position,
                                           return_activations=False)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        return next_token_id, new_kv_caches

    # JIT compile prefill (variable length, compile once per unique length)
    prefill_compiled = jax.jit(
        prefill_jit,
        in_shardings=(None, input_sharding),
        out_shardings=(input_sharding, kv_cache_sharding)
    )

    # JIT compile decode (FIXED length [batch, 1], compile ONCE!)
    decode_compiled = jax.jit(
        decode_step_jit,
        in_shardings=(None, input_sharding, kv_cache_sharding, None),
        out_shardings=(input_sharding, kv_cache_sharding)
    )

    def distributed_generate(params, input_ids):
        """
        Mesh-aware distributed generation with KV caching

        Args:
            params: Sharded model parameters
            input_ids: Input IDs [batch_size, seq_len] - sharded across 'data' axis

        Returns:
            generated_ids: [batch_size, seq_len + max_tokens]
        """
        # Prefill phase: process prompt once
        next_token, kv_caches = prefill_compiled(params, input_ids)
        generated_ids = jnp.concatenate([input_ids, next_token], axis=1)

        prompt_length = input_ids.shape[1]

        # Decode phase: generate tokens one at a time with KV cache
        for i in range(max_tokens - 1):
            next_token, kv_caches = decode_compiled(
                params, next_token, kv_caches, prompt_length + i
            )
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

        return generated_ids

    return distributed_generate


def create_pmap_generate(model, max_tokens):
    """
    Create a pmap-based generation function (fallback, data parallelism only)
    Uses lax.fori_loop for performance
    """
    def generate_step(params, input_ids):
        """Single generation step"""
        logits = model.apply(params, input_ids, return_activations=False)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        new_ids = jnp.concatenate([input_ids, next_token_id], axis=1)
        return new_ids

    def generate_loop_body(step, state):
        """Loop body for generation"""
        params, generated_ids = state
        new_ids = generate_step(params, generated_ids)
        return (params, new_ids)

    @partial(jax.pmap, axis_name='data')
    def distributed_generate(params, input_ids):
        """
        pmap distributed generation function

        Args:
            params: Replicated model parameters
            input_ids: Input IDs [n_devices, batch_per_device, seq_len]
        """
        # Use lax.fori_loop to keep loop inside XLA
        initial_state = (params, input_ids)
        _, generated_ids = lax.fori_loop(
            0, max_tokens,
            generate_loop_body,
            initial_state
        )
        return generated_ids

    return distributed_generate


def create_prompts(data: Dict, grid_encoder, tokenizer, prompt_version: str, predictions_per_task: int):
    """Create prompts for all tasks with data augmentation"""
    prompts = []
    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts'):
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
                    prompts.append(dict(task_id=task_id,
                                      data_augmentation_kwargs=data_augmentation_kwargs,
                                      prompt=prompt,
                                      idx=idx))

            # Break early if we only need a subset of augmentations
            if len(prompts) >= predictions_per_task * len(task['test']):
                break

    return prompts


def create_solutions(predictions: List[Dict], data: Dict, grid_encoder):
    """Create final solutions from predictions"""
    solutions = {}
    for task_id, task in data.items():
        solutions[task_id] = [dict() for _ in task['test']]

    parse_failures = 0
    for pred in predictions:
        task_id = pred['task_id']
        sample_idx = pred.get('idx', 0)
        data_augmentation_kwargs = pred['data_augmentation_kwargs']

        try:
            # Parse grid from prediction text
            grid = parse_grid_from_response(pred['prediction'], grid_encoder)
            # Revert augmentation
            grid = revert_data_augmentation(grid, **data_augmentation_kwargs)

            # Add to solutions
            if task_id in solutions and sample_idx < len(solutions[task_id]):
                attempt_name = f"attempt_{len(solutions[task_id][sample_idx]) + 1}"
                solutions[task_id][sample_idx][attempt_name] = grid.tolist()
        except Exception as e:
            # Skip failed parses but count them
            parse_failures += 1

    if parse_failures > 0:
        print(f"Warning: Failed to parse {parse_failures}/{len(predictions)} predictions")

    return solutions


def distribute_data_across_devices(data: List, n_devices: int, batch_size: int):
    """
    Distribute data across devices for parallel processing

    Args:
        data: List of data items (dicts)
        n_devices: Number of devices
        batch_size: Batch size per device

    Returns:
        Distributed batches shaped for pmap [n_devices, batch_per_device]
    """
    total_batch_size = n_devices * batch_size

    batches = []
    for i in range(0, len(data), total_batch_size):
        batch = data[i:i + total_batch_size]

        # Pad if necessary
        while len(batch) < total_batch_size:
            batch.append(batch[-1])  # Repeat last item

        # Reshape for devices: [n_devices, batch_per_device]
        device_batches = []
        for device_idx in range(n_devices):
            start_idx = device_idx * batch_size
            end_idx = start_idx + batch_size
            device_batches.append(batch[start_idx:end_idx])

        batches.append(device_batches)

    return batches


def inference_main_distributed():
    """Main distributed inference function"""
    # Parse arguments
    args = parse_args_distributed()
    cfg = DistributedARCConfig(**{k: v for k, v in vars(args).items() if v is not None})

    print("="*70)
    print("DISTRIBUTED ARC-AGI INFERENCE WITH ACTIVATION EXTRACTION")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)

    # Setup distributed mesh
    mesh = setup_mesh(cfg.mesh_shape)

    with mesh:
        # Initialize activation extractor
        activation_extractor = ActivationExtractor(cfg) if cfg.extract_activations else None

        # Load data
        with open(cfg.dataset_path) as f:
            data = json.load(f)
        if cfg.n_tasks is not None and cfg.n_tasks > 0:
            data = dict(islice(data.items(), cfg.n_tasks))
        print(f'Loaded {len(data)} tasks from {cfg.dataset_path}')

        # Load tokenizer
        print(f"\nLoading tokenizer from {cfg.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

        # Create grid encoder
        grid_encoder = create_grid_encoder(cfg.grid_encoder)

        # Create model with activation hooks
        print(f"Creating model with activation hooks for layers {cfg.layers_to_extract}...")
        qwen_config = QwenConfig(max_position_embeddings=cfg.max_model_len)
        model = create_model_with_activation_hooks(qwen_config, cfg.layers_to_extract)

        # Load model weights
        print(f"Loading model weights from {cfg.model_path}...")
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, trust_remote_code=True)
        params = convert_hf_to_jax_weights(hf_model, qwen_config)
        del hf_model  # Free memory

        # Initialize model
        key = jax.random.PRNGKey(cfg.random_seed if cfg.random_seed else 42)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        variables = model.init(key, dummy_input)

        # Replace initialized params with loaded weights
        params_dict = {'params': params}

        # Shard parameters across mesh
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        print(f"\nSharding model across {n_devices} devices using mesh...")
        print(f"Target devices: {[str(d) for d in devices]}")
        print(f"Mesh axes: data={cfg.mesh_shape[0]}, model={cfg.mesh_shape[1]}")

        # Use mesh-aware sharding instead of simple replication
        sharded_params = shard_params(params_dict, mesh)

        # Verify sharding
        print(f"\nVerifying parameter sharding:")
        sample_params = list(jax.tree_util.tree_leaves_with_path(sharded_params))[:3]
        for path, param in sample_params:
            param_name = '/'.join(str(k) for k in path)
            print(f"  {param_name}: shape={param.shape}, sharding={param.sharding}")

        # Create prompts
        print(f"\nCreating prompts for {len(data)} tasks...")
        prompts = create_prompts(
            data,
            grid_encoder=grid_encoder,
            tokenizer=tokenizer,
            prompt_version=cfg.prompt_version,
            predictions_per_task=cfg.predictions_per_task
        )
        print(f"Created {len(prompts)} prompts")

        # Tokenize all prompts
        print("Tokenizing prompts...")
        tokenized_prompts = []
        for prompt_data in tqdm(prompts, desc="Tokenizing"):
            input_ids = tokenizer.encode(prompt_data['prompt'], return_tensors='np')
            tokenized_prompts.append({
                **prompt_data,
                'input_ids': input_ids
            })

        # Distribute data across devices
        print(f"\nDistributing data across {n_devices} devices...")
        distributed_batches = distribute_data_across_devices(
            tokenized_prompts,
            n_devices=n_devices,
            batch_size=cfg.batch_size
        )

        print(f"Processing {len(distributed_batches)} batches...")

        # Create mesh-aware distributed generation function
        print(f"Creating mesh-aware generation function...")
        distributed_generate = create_mesh_aware_generate(model, cfg.max_output_tokens, mesh)

        # Run distributed inference
        all_predictions = []

        for batch_idx, batch in enumerate(tqdm(distributed_batches, desc="Inference batches")):
            # For mesh-aware sharding, we need to reshape data differently
            # Instead of [n_devices, batch_per_device, ...], we use [total_batch, ...]
            # and let the sharding handle distribution

            # Extract input_ids and pad to same length
            max_len = max(item['input_ids'].shape[1] for device_batch in batch for item in device_batch)

            batch_input_ids = []
            batch_metadata = []

            # Flatten the nested batch structure for mesh sharding
            for device_batch in batch:
                for item in device_batch:
                    input_ids = item['input_ids'][0]  # Remove batch dim
                    # Pad to max_len
                    padded = jnp.pad(input_ids, (0, max_len - len(input_ids)), constant_values=tokenizer.pad_token_id or 0)
                    batch_input_ids.append(padded)
                    batch_metadata.append(item)

            # Stack into [total_batch, seq_len] instead of [n_devices, batch_per_device, seq_len]
            batch_input_ids = jnp.stack(batch_input_ids)

            # Apply data sharding across 'data' axis of mesh
            data_sharding = get_data_sharding(mesh, batch_input_ids.shape)
            batch_input_ids = jax.device_put(batch_input_ids, data_sharding)

            # Debug: Verify sharding on first batch
            if batch_idx == 0:
                print(f"\nSharding verification:")
                print(f"  Input shape: {batch_input_ids.shape}")
                print(f"  Input sharding: {batch_input_ids.sharding}")
                # Check a sample param
                sample_param = jax.tree_util.tree_leaves(sharded_params)[0]
                print(f"  Params sharding: {sample_param.sharding}")

            # Distributed generation with mesh-aware sharding
            generated_ids = distributed_generate(
                sharded_params,
                batch_input_ids
            )

            # Decode predictions - iterate over flat batch
            for sample_idx in range(generated_ids.shape[0]):
                if sample_idx < len(batch_metadata):
                    meta = batch_metadata[sample_idx]
                    input_len = len(meta['input_ids'][0])

                    # Extract only the newly generated tokens
                    gen_ids = generated_ids[sample_idx]
                    new_tokens = gen_ids[input_len:]

                    # Decode only the generated part
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

                    all_predictions.append({
                        'task_id': meta['task_id'],
                        'idx': meta.get('idx', 0),
                        'prediction': decoded,
                        'data_augmentation_kwargs': meta['data_augmentation_kwargs']
                    })

            # Extract activations if enabled
            if activation_extractor and batch_idx % cfg.save_every_n_batches == 0:
                activation_extractor.save_activations()

        # Save final activations
        if activation_extractor:
            activation_extractor.finalize()

        # Create solutions from predictions
        print(f"\nCreating solutions from {len(all_predictions)} predictions...")

        # Debug: Print first few predictions
        if cfg.verbose or len(all_predictions) > 0:
            print(f"\nSample predictions (first 2):")
            for i, pred in enumerate(all_predictions[:2]):
                print(f"  Prediction {i}: task_id={pred.get('task_id')}, idx={pred.get('idx')}")
                print(f"    Text (first 200 chars): {pred.get('prediction', '')[:200]}")

        solutions = create_solutions(all_predictions, data, grid_encoder)

        # Save outputs
        print(f"Saving predictions to {cfg.output_filepath}...")
        os.makedirs(os.path.dirname(cfg.output_filepath) or '.', exist_ok=True)
        with open(cfg.output_filepath, 'w') as f:
            json.dump(solutions, f, indent=2)

        print("\n" + "="*70)
        print("DISTRIBUTED INFERENCE COMPLETE!")
        print(f"Predictions saved to: {cfg.output_filepath}")
        if activation_extractor:
            print(f"Activations saved to: {cfg.activations_dir}")
        print("="*70)


def parse_args_distributed():
    """Parse command line arguments for distributed inference"""
    parser = argparse.ArgumentParser(description="Distributed ARC-AGI Inference")

    # Add all config parameters
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_filepath', type=str)
    parser.add_argument('--activations_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_tasks', type=int, help='Number of tasks to process (for testing)')
    parser.add_argument('--max_output_tokens', type=int, help='Max tokens to generate')
    parser.add_argument('--predictions_per_task', type=int, help='Predictions per task')
    parser.add_argument('--grid_encoder', type=str, help='Grid encoder configuration')
    parser.add_argument('--prompt_version', type=str, help='Prompt version')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--save_every_n_batches', type=int, help='Save activations every N batches')
    parser.add_argument('--extract_activations', action='store_true')
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--upload_to_cloud', action='store_true')
    parser.add_argument('--cloud_bucket', type=str)
    parser.add_argument('--mesh_shape', type=int, nargs=2, help='Data and model parallelism')

    return parser.parse_args()


if __name__ == '__main__':
    inference_main_distributed()
