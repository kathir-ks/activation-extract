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


@dataclass
class KVCache:
    """
    Key-Value cache for transformer attention
    Stores keys and values for all layers to avoid recomputation
    """
    keys: jnp.ndarray  # [batch, n_layers, n_heads, seq_len, head_dim]
    values: jnp.ndarray  # [batch, n_layers, n_heads, seq_len, head_dim]
    cache_position: int  # Current position in cache
    
    @classmethod
    def create_empty(cls, batch_size: int, n_layers: int, n_heads: int, 
                     head_dim: int, max_seq_len: int, dtype=jnp.float32):
        """Create an empty KV cache"""
        keys = jnp.zeros((batch_size, n_layers, n_heads, max_seq_len, head_dim), dtype=dtype)
        values = jnp.zeros((batch_size, n_layers, n_heads, max_seq_len, head_dim), dtype=dtype)
        return cls(keys=keys, values=values, cache_position=0)
    
    def update(self, layer_idx: int, new_keys: jnp.ndarray, new_values: jnp.ndarray):
        """
        Update cache with new keys/values for a specific layer
        
        Args:
            layer_idx: Layer index to update
            new_keys: [batch, n_heads, new_seq_len, head_dim]
            new_values: [batch, n_heads, new_seq_len, head_dim]
        """
        batch_size, n_heads, new_seq_len, head_dim = new_keys.shape
        start_pos = self.cache_position
        end_pos = start_pos + new_seq_len
        
        # Update keys and values at current position
        keys = self.keys.at[:, layer_idx, :, start_pos:end_pos, :].set(new_keys)
        values = self.values.at[:, layer_idx, :, start_pos:end_pos, :].set(new_values)
        
        return KVCache(keys=keys, values=values, cache_position=end_pos)
    
    def get_layer_cache(self, layer_idx: int):
        """Get cached keys and values for a specific layer up to current position"""
        keys = self.keys[:, layer_idx, :, :self.cache_position, :]
        values = self.values[:, layer_idx, :, :self.cache_position, :]
        return keys, values


@dataclass
class RoPECache:
    """
    Precomputed Rotary Position Embeddings cache
    Stores cos/sin values for all positions to avoid recomputation
    """
    cos: jnp.ndarray  # [max_seq_len, head_dim]
    sin: jnp.ndarray  # [max_seq_len, head_dim]
    
    @classmethod
    def create(cls, max_seq_len: int, head_dim: int, base: float = 10000.0, dtype=jnp.float32):
        """Create RoPE cache with precomputed values"""
        # Compute frequency for each dimension pair
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        
        # Compute position indices
        positions = jnp.arange(max_seq_len, dtype=jnp.float32)
        
        # Compute outer product: [seq_len, head_dim/2]
        freqs = jnp.outer(positions, inv_freq)
        
        # Compute cos and sin, then interleave to match head_dim
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        # Interleave: [seq_len, head_dim/2] -> [seq_len, head_dim]
        cos = jnp.repeat(cos, 2, axis=-1)
        sin = jnp.repeat(sin, 2, axis=-1)
        
        return cls(cos=cos.astype(dtype), sin=sin.astype(dtype))
    
    def get_rope_values(self, start_pos: int, seq_len: int):
        """Get RoPE cos/sin values for a sequence starting at start_pos"""
        cos = self.cos[start_pos:start_pos + seq_len]
        sin = self.sin[start_pos:start_pos + seq_len]
        return cos, sin


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


def get_kv_cache_sharding(mesh: Mesh, cache_shape: tuple) -> NamedSharding:
    """
    Get sharding specification for KV cache
    Shape: [batch, n_layers, n_heads, seq_len, head_dim]
    Shard: batch across 'data', n_heads across 'model'
    """
    spec = P('data', None, 'model', None, None)
    return NamedSharding(mesh, spec)


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


def apply_rope(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    """
    Apply Rotary Position Embeddings using cached cos/sin values
    
    Args:
        q, k: Query and key tensors [batch, n_heads, seq_len, head_dim]
        cos, sin: Cached RoPE values [seq_len, head_dim]
    """
    # Expand cos/sin to match q/k shape
    cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
    sin = sin[None, None, :, :]
    
    # Rotate q and k
    # Split into even and odd dimensions for rotation
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]
    
    # Apply rotation
    q_rotated_even = q_even * cos[..., ::2] - q_odd * sin[..., ::2]
    q_rotated_odd = q_even * sin[..., 1::2] + q_odd * cos[..., 1::2]
    k_rotated_even = k_even * cos[..., ::2] - k_odd * sin[..., ::2]
    k_rotated_odd = k_even * sin[..., 1::2] + k_odd * cos[..., 1::2]
    
    # Interleave back
    q_rotated = jnp.stack([q_rotated_even, q_rotated_odd], axis=-1).reshape(q.shape)
    k_rotated = jnp.stack([k_rotated_even, k_rotated_odd], axis=-1).reshape(k.shape)
    
    return q_rotated, k_rotated


def create_optimized_generate_fn(model, config: DistributedARCConfig, mesh: Mesh, 
                                 model_config: QwenConfig):
    """
    Create optimized generation function with KV cache and RoPE cache
    Separates prefill and decode phases for better performance
    """
    # Get model dimensions
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // n_heads
    max_seq_len = config.max_model_len
    
    # Create RoPE cache if enabled
    rope_cache = None
    if config.use_rope_cache:
        rope_cache = RoPECache.create(
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            base=model_config.rope_theta if hasattr(model_config, 'rope_theta') else 10000.0,
            dtype=jnp.float32
        )
        print(f"Created RoPE cache for {max_seq_len} positions")
    
    # Define sharding specs
    input_sharding = NamedSharding(mesh, P('data', None))
    output_sharding = NamedSharding(mesh, P('data', None))
    
    # Prefill function: process input prompt and initialize KV cache
    def prefill_fn(params, input_ids, kv_cache):
        """
        Prefill phase: process entire input sequence
        Returns: logits, updated kv_cache
        """
        batch_size, seq_len = input_ids.shape
        
        # Forward pass with KV cache
        if config.use_kv_cache:
            logits = model.apply(
                params, input_ids,
                cache=kv_cache,
                use_cache=True,
                return_activations=False
            )
        else:
            logits = model.apply(params, input_ids, return_activations=False)
        
        return logits, kv_cache
    
    # Decode function: generate one token using KV cache
    def decode_step_fn(params, input_ids, kv_cache, rope_cache_obj):
        """
        Decode step: generate one token using cached KV
        input_ids: [batch, 1] - single new token
        Returns: next_token_id, updated kv_cache
        """
        # Forward pass with cache (only processes new token)
        if config.use_kv_cache:
            logits = model.apply(
                params, input_ids,
                cache=kv_cache,
                cache_position=kv_cache.cache_position if kv_cache else 0,
                rope_cache=rope_cache_obj,
                use_cache=True,
                return_activations=False
            )
        else:
            logits = model.apply(params, input_ids, return_activations=False)
        
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
        
        return next_token, kv_cache
    
    # Compile prefill if enabled
    if config.compile_prefill:
        prefill_fn = jax.jit(
            prefill_fn,
            in_shardings=(None, input_sharding, None),
            out_shardings=(output_sharding, None),
            donate_argnums=(2,)  # Donate cache
        )
        print("Compiled prefill function")
    
    # Compile decode step
    decode_step_fn = jax.jit(
        decode_step_fn,
        in_shardings=(None, input_sharding, None, None),
        out_shardings=(input_sharding, None),
        donate_argnums=(2,)  # Donate cache
    )
    print("Compiled decode function")
    
    def generate_with_cache(params, input_ids):
        """
        Full generation with KV cache optimization
        
        Args:
            params: Model parameters (sharded)
            input_ids: Input tokens [batch, seq_len]
        
        Returns:
            generated_ids: [batch, seq_len + max_output_tokens]
        """
        batch_size, input_len = input_ids.shape
        
        # Initialize KV cache
        kv_cache = None
        if config.use_kv_cache:
            cache_dtype = jnp.bfloat16 if config.kv_cache_dtype == 'bfloat16' else jnp.float32
            kv_cache = KVCache.create_empty(
                batch_size=batch_size,
                n_layers=n_layers,
                n_heads=n_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                dtype=cache_dtype
            )
            
            # Shard KV cache across mesh
            cache_sharding = get_kv_cache_sharding(mesh, kv_cache.keys.shape)
            kv_cache = KVCache(
                keys=jax.device_put(kv_cache.keys, cache_sharding),
                values=jax.device_put(kv_cache.values, cache_sharding),
                cache_position=0
            )
        
        # Prefill phase: process input prompt
        if config.prefill_chunk_size and input_len > config.prefill_chunk_size:
            # Chunk prefill for very long sequences
            for chunk_start in range(0, input_len, config.prefill_chunk_size):
                chunk_end = min(chunk_start + config.prefill_chunk_size, input_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                _, kv_cache = prefill_fn(params, chunk_ids, kv_cache)
        else:
            # Single prefill pass
            _, kv_cache = prefill_fn(params, input_ids, kv_cache)
        
        # Decode phase: generate tokens one by one
        generated_ids = input_ids
        
        if config.use_scan_loop:
            # Use lax.scan for efficient loop
            def scan_body(carry, _):
                gen_ids, cache = carry
                last_token = gen_ids[:, -1:]
                next_token, new_cache = decode_step_fn(params, last_token, cache, rope_cache)
                new_gen_ids = jnp.concatenate([gen_ids, next_token], axis=1)
                return (new_gen_ids, new_cache), None
            
            (generated_ids, kv_cache), _ = lax.scan(
                scan_body,
                (generated_ids, kv_cache),
                None,
                length=config.max_output_tokens
            )
        else:
            # Simple loop
            for step in range(config.max_output_tokens):
                last_token = generated_ids[:, -1:]
                next_token, kv_cache = decode_step_fn(params, last_token, kv_cache, rope_cache)
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
        
        generate_fn = create_optimized_generate_fn(model, cfg, mesh, qwen_config)
        
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
            
            # Generate with optimizations
            generated_ids = generate_fn(sharded_params, batch_input_ids)
            
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