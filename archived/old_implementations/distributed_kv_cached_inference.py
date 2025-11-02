"""
Optimized Distributed Inference Pipeline with KV Cache and RoPE Cache
Supports multi-host TPU (v4-64, v5e-64, etc.) with layer activation capture
Includes KV caching, RoPE caching, and performance optimizations
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


# Note: KV caching is handled natively by the model (qwen2_jax.py)
# Each layer returns (k, v) tuples that are concatenated across generation steps


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

    # Cache config
    use_kv_cache: bool = True
    use_rope_cache: bool = True
    kv_cache_dtype: str = 'float32'  # or 'bfloat16' for memory savings
    
    # Performance optimizations
    compile_prefill: bool = True  # Compile prefill separately from decode
    use_scan_loop: bool = True  # Use lax.scan for generation loop
    prefill_chunk_size: Optional[int] = 512  # Chunk size for long prefills
    
    # Activation extraction config
    extract_activations: bool = True
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))
    save_every_n_batches: int = 10
    upload_to_cloud: bool = False
    cloud_bucket: Optional[str] = None

    verbose: bool = False


def setup_mesh(mesh_shape: Tuple[int, int]):
    """Setup JAX mesh for distributed computation"""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Found {n_devices} devices: {devices}")
    
    device_array = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(device_array, axis_names=('data', 'model'))
    print(f"Created mesh with shape {mesh_shape}: {mesh}")
    
    return mesh


def get_param_sharding_spec(param_name: str, param_shape: tuple, mesh: Mesh) -> NamedSharding:
    """Get sharding specification for a model parameter"""
    if 'embed' in param_name or 'lm_head' in param_name:
        if len(param_shape) >= 2:
            spec = P(None, 'model')
        else:
            spec = P(None)
    elif 'norm' in param_name:
        spec = P(None)
    elif len(param_shape) == 2:
        spec = P(None, 'model')
    else:
        spec = P(None)
    
    return NamedSharding(mesh, spec)


def shard_params(params: Dict, mesh: Mesh) -> Dict:
    """Shard model parameters according to mesh"""
    def shard_leaf(path, param):
        param_name = '/'.join(str(k) for k in path)
        sharding = get_param_sharding_spec(param_name, param.shape, mesh)
        return jax.device_put(param, sharding)
    
    from jax.tree_util import tree_map_with_path
    sharded_params = tree_map_with_path(shard_leaf, params)
    return sharded_params


# KV cache sharding is handled per-layer in the model
# Each layer's cache is a tuple (k, v) where k and v are [batch, n_heads, seq_len, head_dim]


def get_data_sharding(mesh: Mesh, batch_shape: tuple) -> NamedSharding:
    """Get sharding specification for input data"""
    spec = P('data', None)
    return NamedSharding(mesh, spec)


class ActivationExtractor:
    """Extract and store activations from specific layers"""

    def __init__(self, config: DistributedARCConfig):
        self.config = config
        self.activations_buffer = {}
        self.batch_count = 0
        os.makedirs(config.activations_dir, exist_ok=True)
        
        self.metadata = {
            'layers_extracted': config.layers_to_extract,
            'model_path': config.model_path,
            'batch_size': config.batch_size,
            'batches': []
        }

    def extract_layer_activation(self, layer_idx: int, activation: jnp.ndarray,
                                task_id: str, sample_idx: int):
        """Extract activation from a specific layer"""
        if layer_idx not in self.config.layers_to_extract:
            return
        
        activation_np = np.array(activation)
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
        
        for layer_key, activations in self.activations_buffer.items():
            filename = f"{layer_key}_batch_{self.batch_count:06d}.pkl"
            filepath = os.path.join(self.config.activations_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(activations, f)
            
            print(f"Saved {len(activations)} activations to {filepath}")
            
            self.metadata['batches'].append({
                'batch_id': self.batch_count,
                'layer': layer_key,
                'file': filename,
                'n_samples': len(activations)
            })
        
        if self.config.upload_to_cloud and self.config.cloud_bucket:
            self.upload_to_cloud_storage()
        
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
        self.save_activations(force=True)
        
        metadata_path = os.path.join(self.config.activations_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved activation metadata to {metadata_path}")


def create_model_with_activation_hooks(config: QwenConfig, layers_to_extract: List[int]):
    """Create model with activation extraction hooks"""
    model = create_model_with_hooks(config, layers_to_extract=layers_to_extract)
    return model


# RoPE is applied natively in qwen2_jax.py using apply_rotary_pos_emb()


def create_optimized_generate_fn(model, config: DistributedARCConfig, mesh: Mesh,
                                 model_config: QwenConfig,
                                 extract_activations: bool = False,
                                 activation_extractor = None):
    """
    Create optimized generation function with KV cache and optional activation extraction
    Separates prefill and decode phases for better performance
    Note: RoPE cache is built into the model (qwen2_jax.py QwenAttention.setup())

    Args:
        model: Model instance (QwenModel or QwenModelWithActivations)
        config: Distributed inference config
        mesh: JAX mesh for sharding
        model_config: Qwen model config
        extract_activations: Whether to extract activations during generation
        activation_extractor: ActivationExtractor instance (if extract_activations=True)
    """

    # Define sharding specs
    input_sharding = NamedSharding(mesh, P('data', None))
    output_sharding = NamedSharding(mesh, P('data', None))

    # Prefill function: process input prompt and initialize KV cache
    def prefill_fn(params, input_ids, kv_caches):
        """
        Prefill phase: process entire input sequence
        Returns: logits, updated kv_caches (list of per-layer caches)
        """
        batch_size, seq_len = input_ids.shape

        # Forward pass - model returns logits and new_kv_caches
        # Always use return_activations=False during prefill for speed
        # Activations are typically only needed during decode (for generated tokens)
        if hasattr(model, 'layers_to_extract'):
            # QwenModelWithActivations
            logits, new_kv_caches = model.apply(
                params, input_ids,
                kv_caches=kv_caches,
                position_offset=0,
                return_activations=False
            )
        else:
            # QwenModel - no return_activations parameter
            logits, new_kv_caches = model.apply(
                params, input_ids,
                kv_caches=kv_caches,
                position_offset=0
            )

        return logits, new_kv_caches

    # Decode function: generate one token using KV cache
    def decode_step_fn(params, input_ids, kv_caches, position, extract_acts=False):
        """
        Decode step: generate one token using cached KV
        input_ids: [batch, 1] - single new token
        Returns: next_token_id, updated kv_caches, activations (if extract_acts=True)
        """
        # Forward pass with cache (only processes new token)
        if hasattr(model, 'layers_to_extract') and extract_acts:
            # QwenModelWithActivations - extract activations
            logits, new_kv_caches, activations = model.apply(
                params, input_ids,
                kv_caches=kv_caches,
                position_offset=position,
                return_activations=True
            )
        elif hasattr(model, 'layers_to_extract'):
            # QwenModelWithActivations - no activation extraction
            logits, new_kv_caches = model.apply(
                params, input_ids,
                kv_caches=kv_caches,
                position_offset=position,
                return_activations=False
            )
            activations = None
        else:
            # QwenModel - no return_activations parameter
            logits, new_kv_caches = model.apply(
                params, input_ids,
                kv_caches=kv_caches,
                position_offset=position
            )
            activations = None

        # Sample next token (greedy or sampling based on temperature)
        next_token_logits = logits[:, -1, :]

        if config.temperature == 0.0:
            # Greedy decoding
            next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            # Temperature sampling
            scaled_logits = next_token_logits / config.temperature
            next_token = jax.random.categorical(
                jax.random.PRNGKey(0), scaled_logits, axis=-1
            ).reshape(-1, 1)

        return next_token, new_kv_caches, activations
    
    # Compile prefill if enabled
    if config.compile_prefill:
        prefill_fn = jax.jit(
            prefill_fn,
            in_shardings=(None, input_sharding, None),
            out_shardings=(output_sharding, None),
            donate_argnums=(2,)  # Donate cache
        )
        print("Compiled prefill function")
    
    # Note: We don't JIT compile decode_step_fn separately here because
    # it will be traced inside the scan loop with traced position values
    # The entire generate_with_cache function (including the scan) will be JIT compiled
    print("Decode function ready (will be JIT compiled as part of generation)")
    
    def generate_with_cache(params, input_ids, batch_metadata=None):
        """
        Full generation with KV cache optimization and optional activation extraction

        Args:
            params: Model parameters (sharded)
            input_ids: Input tokens [batch, seq_len]
            batch_metadata: Optional metadata for activation tracking

        Returns:
            generated_ids: [batch, seq_len + max_output_tokens]
        """
        batch_size, input_len = input_ids.shape

        # Initialize KV caches (None for prefill - model will initialize per-layer caches)
        kv_caches = None

        # Prefill phase: process input prompt
        if config.prefill_chunk_size and input_len > config.prefill_chunk_size:
            # Chunk prefill for very long sequences
            for chunk_start in range(0, input_len, config.prefill_chunk_size):
                chunk_end = min(chunk_start + config.prefill_chunk_size, input_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                _, kv_caches = prefill_fn(params, chunk_ids, kv_caches)
        else:
            # Single prefill pass
            _, kv_caches = prefill_fn(params, input_ids, kv_caches)

        # Decode phase: generate tokens one by one
        generated_ids = input_ids

        if config.use_scan_loop:
            # NOTE: lax.scan currently doesn't work with growing KV caches
            # because the cache shape changes on each iteration (growing from [batch, heads, seq_len, dim])
            # This violates scan's requirement for constant carry shapes
            # TODO: Implement with pre-allocated fixed-size caches for scan compatibility
            print("Warning: scan_loop disabled - growing KV caches incompatible with lax.scan")
            print("         Falling back to simple loop")

            # Fall back to simple loop with activation extraction
            for step in range(config.max_output_tokens):
                last_token = generated_ids[:, -1:]
                position = input_len + step

                # Extract activations if configured
                should_extract = (extract_activations and activation_extractor is not None
                                and step % config.save_every_n_batches == 0)

                next_token, kv_caches, activations = decode_step_fn(
                    params, last_token, kv_caches, position, extract_acts=should_extract
                )

                # Store activations if extracted
                if should_extract and activations and batch_metadata:
                    for sample_idx in range(batch_size):
                        if sample_idx < len(batch_metadata):
                            meta = batch_metadata[sample_idx]
                            for layer_name, layer_acts in activations.items():
                                if layer_name.startswith('layer_'):
                                    layer_idx = int(layer_name.split('_')[1])
                                    # Extract this sample's activation
                                    sample_act = layer_acts[sample_idx:sample_idx+1, -1, :]
                                    activation_extractor.extract_layer_activation(
                                        layer_idx, sample_act,
                                        meta['task_id'], meta.get('idx', 0)
                                    )

                generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)
        else:
            # Simple loop with activation extraction
            for step in range(config.max_output_tokens):
                last_token = generated_ids[:, -1:]
                position = input_len + step

                # Extract activations if configured
                should_extract = (extract_activations and activation_extractor is not None
                                and step % config.save_every_n_batches == 0)

                next_token, kv_caches, activations = decode_step_fn(
                    params, last_token, kv_caches, position, extract_acts=should_extract
                )

                # Store activations if extracted
                if should_extract and activations and batch_metadata:
                    for sample_idx in range(batch_size):
                        if sample_idx < len(batch_metadata):
                            meta = batch_metadata[sample_idx]
                            for layer_name, layer_acts in activations.items():
                                if layer_name.startswith('layer_'):
                                    layer_idx = int(layer_name.split('_')[1])
                                    # Extract this sample's activation
                                    sample_act = layer_acts[sample_idx:sample_idx+1, -1, :]
                                    activation_extractor.extract_layer_activation(
                                        layer_idx, sample_act,
                                        meta['task_id'], meta.get('idx', 0)
                                    )

                generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

        return generated_ids
    
    return generate_with_cache


def create_prompts(data: Dict, grid_encoder, tokenizer, prompt_version: str, predictions_per_task: int):
    """Create prompts for all tasks with data augmentation"""
    prompts = []
    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts'):
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        num_augmentations = len(data_augmentation_params)
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
                    prompts.append(dict(
                        task_id=task_id,
                        data_augmentation_kwargs=data_augmentation_kwargs,
                        prompt=prompt,
                        idx=idx
                    ))
            
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
            grid = parse_grid_from_response(pred['prediction'], grid_encoder)
            grid = revert_data_augmentation(grid, **data_augmentation_kwargs)
            
            if task_id in solutions and sample_idx < len(solutions[task_id]):
                attempt_name = f"attempt_{len(solutions[task_id][sample_idx]) + 1}"
                solutions[task_id][sample_idx][attempt_name] = grid.tolist()
        except Exception as e:
            parse_failures += 1
    
    if parse_failures > 0:
        print(f"Warning: Failed to parse {parse_failures}/{len(predictions)} predictions")
    
    return solutions


def distribute_and_pad_batch(batch_data: List[Dict], tokenizer, max_len: int):
    """
    Distribute and pad a batch for efficient processing
    
    Returns:
        batch_input_ids: [batch_size, max_len]
        batch_metadata: List of metadata dicts
    """
    batch_input_ids = []
    batch_metadata = []
    
    for item in batch_data:
        input_ids = item['input_ids'][0]  # Remove batch dim
        padded = jnp.pad(
            input_ids,
            (0, max_len - len(input_ids)),
            constant_values=tokenizer.pad_token_id or 0
        )
        batch_input_ids.append(padded)
        batch_metadata.append(item)
    
    return jnp.stack(batch_input_ids), batch_metadata


def inference_main_distributed():
    """Main distributed inference function with optimizations"""
    args = parse_args_distributed()
    cfg = DistributedARCConfig(**{k: v for k, v in vars(args).items() if v is not None})
    
    print("="*70)
    print("OPTIMIZED DISTRIBUTED INFERENCE WITH KV CACHE")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)
    
    # Setup mesh
    mesh = setup_mesh(cfg.mesh_shape)
    
    with mesh:
        # Initialize components
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
        
        # Create model
        print(f"Creating model...")
        qwen_config = QwenConfig(max_position_embeddings=cfg.max_model_len)
        model = create_model_with_activation_hooks(qwen_config, cfg.layers_to_extract)
        
        # Load weights
        print(f"Loading model weights from {cfg.model_path}...")
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, trust_remote_code=True)
        params = convert_hf_to_jax_weights(hf_model, qwen_config)
        del hf_model
        
        # Initialize and shard parameters
        key = jax.random.PRNGKey(cfg.random_seed if cfg.random_seed else 42)
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        variables = model.init(key, dummy_input)
        params_dict = {'params': params}
        
        n_devices = jax.local_device_count()
        print(f"\nSharding model across {n_devices} devices...")
        sharded_params = shard_params(params_dict, mesh)
        
        # Create prompts
        print(f"\nCreating prompts...")
        prompts = create_prompts(
            data, grid_encoder, tokenizer,
            cfg.prompt_version, cfg.predictions_per_task
        )
        print(f"Created {len(prompts)} prompts")
        
        # Tokenize
        print("Tokenizing...")
        tokenized_prompts = []
        for prompt_data in tqdm(prompts, desc="Tokenizing"):
            input_ids = tokenizer.encode(prompt_data['prompt'], return_tensors='np')
            tokenized_prompts.append({**prompt_data, 'input_ids': input_ids})
        
        # Create optimized generation function
        print(f"\nCreating optimized generation function...")
        print(f"  KV Cache: {cfg.use_kv_cache}")
        print(f"  RoPE Cache: {cfg.use_rope_cache}")
        print(f"  Scan Loop: {cfg.use_scan_loop}")
        print(f"  Extract Activations: {cfg.extract_activations}")

        generate_fn = create_optimized_generate_fn(
            model, cfg, mesh, qwen_config,
            extract_activations=cfg.extract_activations,
            activation_extractor=activation_extractor
        )
        
        # Process in batches
        all_predictions = []
        
        for batch_idx in tqdm(range(0, len(tokenized_prompts), cfg.batch_size), 
                             desc="Processing batches"):
            batch_end = min(batch_idx + cfg.batch_size, len(tokenized_prompts))
            batch_data = tokenized_prompts[batch_idx:batch_end]
            
            # Pad batch
            max_len = max(item['input_ids'].shape[1] for item in batch_data)
            batch_input_ids, batch_metadata = distribute_and_pad_batch(
                batch_data, tokenizer, max_len
            )
            
            # Shard input
            data_sharding = get_data_sharding(mesh, batch_input_ids.shape)
            batch_input_ids = jax.device_put(batch_input_ids, data_sharding)

            # Generate with optimizations (pass metadata for activation extraction)
            generated_ids = generate_fn(sharded_params, batch_input_ids, batch_metadata)
            
            # Decode predictions
            for sample_idx in range(len(batch_metadata)):
                meta = batch_metadata[sample_idx]
                input_len = len(meta['input_ids'][0])
                
                gen_ids = generated_ids[sample_idx]
                new_tokens = gen_ids[input_len:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                all_predictions.append({
                    'task_id': meta['task_id'],
                    'idx': meta.get('idx', 0),
                    'prediction': decoded,
                    'data_augmentation_kwargs': meta['data_augmentation_kwargs']
                })
            
            # Save activations periodically
            if activation_extractor and batch_idx % cfg.save_every_n_batches == 0:
                activation_extractor.save_activations()
        
        # Finalize
        if activation_extractor:
            activation_extractor.finalize()
        
        # Create solutions
        print(f"\nCreating solutions from {len(all_predictions)} predictions...")
        solutions = create_solutions(all_predictions, data, grid_encoder)
        
        # Save
        print(f"Saving to {cfg.output_filepath}...")
        os.makedirs(os.path.dirname(cfg.output_filepath) or '.', exist_ok=True)
        with open(cfg.output_filepath, 'w') as f:
            json.dump(solutions, f, indent=2)
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETE!")
        print(f"Predictions: {cfg.output_filepath}")
        if activation_extractor:
            print(f"Activations: {cfg.activations_dir}")
        print("="*70)


def parse_args_distributed():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimized Distributed Inference")
    
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_filepath', type=str)
    parser.add_argument('--activations_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_tasks', type=int)
    parser.add_argument('--max_output_tokens', type=int)
    parser.add_argument('--predictions_per_task', type=int)
    parser.add_argument('--grid_encoder', type=str)
    parser.add_argument('--prompt_version', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--save_every_n_batches', type=int)
    parser.add_argument('--extract_activations', action='store_true')
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--upload_to_cloud', action='store_true')
    parser.add_argument('--cloud_bucket', type=str)
    parser.add_argument('--mesh_shape', type=int, nargs=2)
    
    # Cache and optimization flags
    parser.add_argument('--use_kv_cache', action='store_true', default=True)
    parser.add_argument('--no_kv_cache', dest='use_kv_cache', action='store_false')
    parser.add_argument('--use_rope_cache', action='store_true', default=True)
    parser.add_argument('--no_rope_cache', dest='use_rope_cache', action='store_false')
    parser.add_argument('--kv_cache_dtype', type=str, choices=['float32', 'bfloat16'])
    parser.add_argument('--compile_prefill', action='store_true', default=True)
    parser.add_argument('--use_scan_loop', action='store_true', default=True)
    parser.add_argument('--prefill_chunk_size', type=int)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    inference_main_distributed()