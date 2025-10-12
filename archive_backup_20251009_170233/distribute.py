"""
Fixed Distributed Inference Pipeline with Activation Extraction for TPU Pods
Supports multi-host TPU (v4-64, v5e-64, etc.) with layer activation capture
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.tree_util import tree_map_with_path
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
import warnings

from transformers import AutoTokenizer
from qwen2 import QwenConfig, convert_hf_to_jax_weights
from qwen2 import QwenModelWithActivations, create_model_with_hooks
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
    eos_token_id: Optional[int] = None  # Added for proper stopping

    # Distributed config
    mesh_shape: Tuple[int, int] = (1, 1)  # (data, model) parallelism
    use_pjit: bool = True

    # Activation extraction config
    extract_activations: bool = True
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))
    save_every_n_batches: int = 10
    upload_to_cloud: bool = False
    cloud_bucket: Optional[str] = None

    verbose: bool = False

    def validate(self):
        """Validate configuration parameters"""
        n_devices = jax.device_count()
        mesh_size = self.mesh_shape[0] * self.mesh_shape[1]
        if mesh_size != n_devices:
            raise ValueError(f"Mesh shape {self.mesh_shape} requires {mesh_size} devices, but found {n_devices}")
        
        if self.batch_size % self.mesh_shape[0] != 0:
            raise ValueError(f"Batch size {self.batch_size} must be divisible by data parallelism {self.mesh_shape[0]}")
        
        os.makedirs(os.path.dirname(self.output_filepath) or '.', exist_ok=True)
        os.makedirs(self.activations_dir, exist_ok=True)


def setup_mesh(mesh_shape: Tuple[int, int]):
    """Setup JAX mesh for distributed computation"""
    devices = jax.devices()
    n_devices = len(devices)
    
    print(f"Found {n_devices} devices: {devices}")
    
    # Validate mesh shape
    if mesh_shape[0] * mesh_shape[1] != n_devices:
        raise ValueError(f"Mesh shape {mesh_shape} incompatible with {n_devices} devices")
    
    # Create device mesh
    device_array = np.array(devices).reshape(mesh_shape)
    
    # Create mesh with named axes
    mesh = Mesh(device_array, axis_names=('data', 'model'))
    
    print(f"Created mesh with shape {mesh_shape}: {mesh}")
    
    return mesh


def get_param_sharding_spec(param_name: str, param_shape: tuple, mesh: Mesh) -> NamedSharding:
    """
    Get sharding specification for a model parameter
    
    Fixed to handle all parameter shapes correctly
    """
    # Handle different parameter types and shapes
    if 'embed' in param_name or 'lm_head' in param_name:
        # Embedding layers: [vocab_size, hidden_size]
        if len(param_shape) == 2:
            spec = P(None, 'model')  # Shard hidden dimension
        else:
            spec = P(None)
    elif 'norm' in param_name or 'bias' in param_name:
        # Norm layers and biases: replicate
        spec = P(*([None] * len(param_shape)))
    elif 'attention' in param_name or 'attn' in param_name:
        # Attention weights: handle various shapes
        if len(param_shape) == 4:  # [batch, heads, seq, dim]
            spec = P(None, 'model', None, None)
        elif len(param_shape) == 3:  # [heads, seq, dim] or similar
            spec = P('model', None, None)
        elif len(param_shape) == 2:  # [in_dim, out_dim]
            spec = P(None, 'model')
        else:
            spec = P(*([None] * len(param_shape)))
    elif len(param_shape) == 2:
        # Linear layers: [in_features, out_features]
        spec = P(None, 'model')
    else:
        # Default: replicate
        spec = P(*([None] * len(param_shape)))
    
    return NamedSharding(mesh, spec)


def shard_params(params: Dict, mesh: Mesh) -> Dict:
    """Shard model parameters according to mesh"""
    def shard_leaf(path, param):
        param_name = '/'.join(str(k) for k in path)
        sharding = get_param_sharding_spec(param_name, param.shape, mesh)
        return jax.device_put(param, sharding)
    
    # Use tree_map_with_path to get parameter names
    sharded_params = tree_map_with_path(shard_leaf, params)
    
    return sharded_params


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
    
    def extract_batch_activations(self, activations_dict: Dict, batch_metadata: List[Dict]):
        """Extract activations for a batch of samples"""
        for layer_idx in self.config.layers_to_extract:
            layer_key = f"layer_{layer_idx}"
            if layer_key in activations_dict:
                activation = activations_dict[layer_key]
                
                # Convert to numpy and store with metadata
                activation_np = np.array(activation)
                
                if layer_key not in self.activations_buffer:
                    self.activations_buffer[layer_key] = []
                
                for i, meta in enumerate(batch_metadata):
                    if i < activation_np.shape[0]:
                        self.activations_buffer[layer_key].append({
                            'task_id': meta.get('task_id'),
                            'sample_idx': meta.get('idx', 0),
                            'activation': activation_np[i],
                            'shape': activation_np[i].shape
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
            self._upload_to_cloud_storage()
        
        # Clear buffer
        self.activations_buffer = {}
    
    def _upload_to_cloud_storage(self):
        """Upload activations to cloud storage (GCS)"""
        try:
            from google.cloud import storage
            
            bucket_name = self.config.cloud_bucket.replace('gs://', '').split('/')[0]
            prefix = '/'.join(self.config.cloud_bucket.replace('gs://', '').split('/')[1:])
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Upload all pkl files
            for filename in os.listdir(self.config.activations_dir):
                if filename.endswith('.pkl'):
                    local_path = os.path.join(self.config.activations_dir, filename)
                    blob_path = os.path.join(prefix, filename) if prefix else filename
                    
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_path)
                    
                    if self.config.verbose:
                        print(f"Uploaded {filename} to {self.config.cloud_bucket}")
        
        except ImportError:
            warnings.warn("google-cloud-storage not installed, skipping cloud upload")
        except Exception as e:
            warnings.warn(f"Failed to upload to cloud storage: {e}")
    
    def finalize(self):
        """Finalize and save metadata"""
        # Save any remaining activations
        self.save_activations(force=True)
        
        # Save metadata
        metadata_path = os.path.join(self.config.activations_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved activation metadata to {metadata_path}")


def create_mesh_aware_generate_with_activations(model, cfg: DistributedARCConfig, mesh):
    """
    Create a mesh-aware distributed generation function with activation extraction
    
    Fixed to properly handle activations and stopping conditions
    """
    # Define sharding
    input_sharding = NamedSharding(mesh, P('data', None))
    output_sharding = NamedSharding(mesh, P('data', None))
    
    def generate_step(params, generated_ids, return_activations=False):
        """Single generation step"""
        if return_activations:
            logits, activations = model.apply(params, generated_ids, return_activations=True)
            return logits, activations
        else:
            logits = model.apply(params, generated_ids, return_activations=False)
            return logits, None
    
    def distributed_generate(params, input_ids):
        """
        Mesh-aware distributed generation function
        
        Returns: (generated_ids, activations_list)
        """
        batch_size = input_ids.shape[0]
        init_len = input_ids.shape[1]
        max_len = init_len + cfg.max_output_tokens
        
        # Pre-allocate output array
        generated_ids = jnp.zeros((batch_size, max_len), dtype=input_ids.dtype)
        generated_ids = generated_ids.at[:, :init_len].set(input_ids)
        
        # Storage for activations if needed
        all_activations = [] if cfg.extract_activations else None
        
        # Generation loop
        for step in range(cfg.max_output_tokens):
            current_len = init_len + step
            current_sequence = generated_ids[:, :current_len]
            
            # Forward pass
            logits, activations = generate_step(
                params, current_sequence, 
                return_activations=cfg.extract_activations
            )
            
            # Store activations
            if cfg.extract_activations and activations is not None:
                all_activations.append(activations)
            
            # Sample next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token_id = jnp.argmax(next_token_logits, axis=-1)
            
            # Update sequence
            generated_ids = generated_ids.at[:, current_len].set(next_token_id)
            
            # Check for EOS token if configured
            if cfg.eos_token_id is not None:
                # Create mask for sequences that have hit EOS
                eos_mask = jnp.any(generated_ids[:, :current_len+1] == cfg.eos_token_id, axis=1)
                if jnp.all(eos_mask):
                    break
        
        # Return generated sequences and activations
        return generated_ids[:, :init_len + step + 1], all_activations
    
    # JIT compile with sharding
    distributed_generate_jit = jax.jit(
        distributed_generate,
        in_shardings=(None, input_sharding),
        out_shardings=(output_sharding, None)  # Don't shard activations
    )
    
    return distributed_generate_jit


def create_prompts(data: Dict, grid_encoder, tokenizer, prompt_version: str, predictions_per_task: int):
    """Create prompts for all tasks with data augmentation"""
    prompts = []
    
    for task_id, task in tqdm(data.items(), total=len(data), desc='Creating prompts'):
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        num_augmentations = len(data_augmentation_params)
        
        # Calculate how many times to repeat each augmentation
        repeats_per_aug = max(1, predictions_per_task // num_augmentations)
        
        prompt_count = 0
        for hflip, n_rot90 in data_augmentation_params:
            if prompt_count >= predictions_per_task * len(task['test']):
                break
                
            for _ in range(repeats_per_aug):
                if prompt_count >= predictions_per_task * len(task['test']):
                    break
                    
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
                    prompt_count += 1
    
    return prompts


def create_solutions(predictions: List[Dict], data: Dict, grid_encoder):
    """Create final solutions from predictions with better error handling"""
    solutions = {}
    for task_id, task in data.items():
        solutions[task_id] = [dict() for _ in task['test']]
    
    parse_failures = []
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
            # Track failed parses with details
            parse_failures.append({
                'task_id': task_id,
                'idx': sample_idx,
                'error': str(e),
                'prediction_preview': pred['prediction'][:100] if pred['prediction'] else None
            })
    
    if parse_failures:
        print(f"Warning: Failed to parse {len(parse_failures)}/{len(predictions)} predictions")
        if len(parse_failures) <= 5:  # Show details for small number of failures
            for failure in parse_failures:
                print(f"  Task {failure['task_id']}-{failure['idx']}: {failure['error']}")
    
    return solutions


def prepare_batches(tokenized_prompts: List[Dict], cfg: DistributedARCConfig):
    """
    Prepare batches for distributed processing with proper alignment
    
    Fixed to ensure metadata stays aligned with data
    """
    total_batch_size = cfg.batch_size
    batches = []
    
    for i in range(0, len(tokenized_prompts), total_batch_size):
        batch_items = tokenized_prompts[i:i + total_batch_size]
        
        # Pad batch if needed (but track which are padding)
        actual_size = len(batch_items)
        while len(batch_items) < total_batch_size:
            # Create dummy padding item
            batch_items.append({
                'task_id': '__padding__',
                'idx': -1,
                'input_ids': batch_items[-1]['input_ids'],  # Reuse last item's input
                'data_augmentation_kwargs': {},
                'is_padding': True
            })
        
        batches.append((batch_items, actual_size))
    
    return batches


def inference_main_distributed():
    """Main distributed inference function - Fixed version"""
    # Parse arguments
    args = parse_args_distributed()
    cfg = DistributedARCConfig(**{k: v for k, v in vars(args).items() if v is not None})
    
    # Validate configuration
    try:
        cfg.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    print("="*70)
    print("DISTRIBUTED ARC-AGI INFERENCE WITH ACTIVATION EXTRACTION")
    print("="*70)
    print(json.dumps(asdict(cfg), indent=2))
    print("="*70)
    
    # Set random seed if provided
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)
        key = jax.random.PRNGKey(cfg.random_seed)
    else:
        key = jax.random.PRNGKey(42)
    
    # Setup distributed mesh
    mesh = setup_mesh(cfg.mesh_shape)
    
    with mesh:
        # Initialize activation extractor
        activation_extractor = ActivationExtractor(cfg) if cfg.extract_activations else None
        
        # Load data
        try:
            with open(cfg.dataset_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file {cfg.dataset_path} not found")
            return
        
        if cfg.n_tasks is not None and cfg.n_tasks > 0:
            data = dict(islice(data.items(), cfg.n_tasks))
        print(f'Loaded {len(data)} tasks from {cfg.dataset_path}')
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {cfg.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        
        # Set padding token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
        # Store EOS token ID in config
        cfg.eos_token_id = tokenizer.eos_token_id
        
        # Create grid encoder
        grid_encoder = create_grid_encoder(cfg.grid_encoder)
        
        # Create model with activation hooks if needed
        print(f"Creating model...")
        qwen_config = QwenConfig(max_position_embeddings=cfg.max_model_len)
        
        # if cfg.extract_activations:
        #     print(f"  with activation hooks for layers {cfg.layers_to_extract}")
        #     model = create_model_with_hooks(qwen_config, cfg.layers_to_extract)
        # else:
        #     model = QwenModel(qwen_config)
        
        model = create_model_with_hooks(qwen_config, cfg.layers_to_extract)
        
        # Load model weights
        print(f"Loading model weights from {cfg.model_path}...")
        try:
            # Load in streaming mode to save memory
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True
            )
            params = convert_hf_to_jax_weights(hf_model, qwen_config)
            del hf_model  # Free memory immediately
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Initialize model
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        variables = model.init(key, dummy_input)
        
        # Replace initialized params with loaded weights
        params_dict = {'params': params}
        
        # Shard parameters across mesh
        n_devices = jax.local_device_count()
        print(f"\nSharding model across {n_devices} devices using mesh...")
        print(f"Mesh axes: data={cfg.mesh_shape[0]}, model={cfg.mesh_shape[1]}")
        
        sharded_params = shard_params(params_dict, mesh)
        
        # Verify sharding
        if cfg.verbose:
            print(f"\nParameter sharding verification:")
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
            # Ensure we get numpy array without batch dimension
            encoded = tokenizer.encode(prompt_data['prompt'])
            if isinstance(encoded, list):
                input_ids = np.array(encoded)
            else:
                input_ids = np.array(encoded).squeeze()
            
            tokenized_prompts.append({
                **prompt_data,
                'input_ids': input_ids
            })
        
        # Prepare batches with proper alignment
        print(f"\nPreparing batches (batch_size={cfg.batch_size})...")
        batches = prepare_batches(tokenized_prompts, cfg)
        print(f"Created {len(batches)} batches")
        
        # Create mesh-aware distributed generation function
        print(f"Creating distributed generation function...")
        distributed_generate = create_mesh_aware_generate_with_activations(
            model, cfg, mesh
        )
        
        # Run distributed inference
        all_predictions = []
        
        for batch_idx, (batch_items, actual_size) in enumerate(tqdm(batches, desc="Inference")):
            # Prepare batch input
            max_len = max(len(item['input_ids']) for item in batch_items)
            
            batch_input_ids = []
            batch_metadata = []
            
            for item in batch_items:
                # Skip padding items in metadata
                if not item.get('is_padding', False):
                    batch_metadata.append(item)
                
                # Pad input_ids to max_len
                input_ids = item['input_ids']
                padded = np.pad(
                    input_ids, 
                    (0, max_len - len(input_ids)), 
                    constant_values=tokenizer.pad_token_id
                )
                batch_input_ids.append(padded)
            
            # Stack into batch tensor
            batch_input_ids = jnp.stack(batch_input_ids)
            
            # Apply data sharding
            data_sharding = NamedSharding(mesh, P('data', None))
            batch_input_ids = jax.device_put(batch_input_ids, data_sharding)
            
            # Run generation with activation extraction
            generated_ids, batch_activations = distributed_generate(
                sharded_params,
                batch_input_ids
            )
            
            # Extract activations if enabled
            if activation_extractor and batch_activations:
                # Aggregate activations from all steps
                aggregated_activations = {}
                for step_activations in batch_activations:
                    for layer_name, layer_act in step_activations.items():
                        if layer_name not in aggregated_activations:
                            aggregated_activations[layer_name] = []
                        aggregated_activations[layer_name].append(layer_act)
                
                # Average or take last activation per layer
                for layer_name in aggregated_activations:
                    # Take last token's activation
                    aggregated_activations[layer_name] = aggregated_activations[layer_name][-1]
                
                activation_extractor.extract_batch_activations(
                    aggregated_activations, 
                    batch_metadata
                )
            
            # Decode predictions (only for non-padding items)
            for i, meta in enumerate(batch_metadata):
                if i < generated_ids.shape[0]:
                    input_len = len(meta['input_ids'])
                    gen_ids = generated_ids[i]
                    
                    # Extract only the newly generated tokens
                    new_tokens = gen_ids[input_len:]
                    
                    # Find EOS position if exists
                    if cfg.eos_token_id is not None:
                        eos_positions = jnp.where(new_tokens == cfg.eos_token_id)[0]
                        if len(eos_positions) > 0:
                            new_tokens = new_tokens[:eos_positions[0]]
                    
                    # Decode
                    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    all_predictions.append({
                        'task_id': meta['task_id'],
                        'idx': meta.get('idx', 0),
                        'prediction': decoded,
                        'data_augmentation_kwargs': meta['data_augmentation_kwargs']
                    })
            
            # Save activations periodically
            if activation_extractor and (batch_idx + 1) % cfg.save_every_n_batches == 0:
                activation_extractor.save_activations()
        
        # Finalize activations
        if activation_extractor:
            activation_extractor.finalize()
        
        # Create solutions from predictions
        print(f"\nCreating solutions from {len(all_predictions)} predictions...")
        
        # Debug output if verbose
        if cfg.verbose and all_predictions:
            print(f"\nSample predictions (first 2):")
            for i, pred in enumerate(all_predictions[:2]):
                print(f"  Prediction {i}: task_id={pred['task_id']}, idx={pred['idx']}")
                print(f"    Text preview: {pred['prediction'][:200]}...")
        
        solutions = create_solutions(all_predictions, data, grid_encoder)
        
        # Save outputs
        print(f"Saving predictions to {cfg.output_filepath}...")
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
    parser.add_argument('--no_extract_activations', dest='extract_activations', action='store_false')
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    parser.add_argument('--upload_to_cloud', action='store_true')
    parser.add_argument('--cloud_bucket', type=str)
    parser.add_argument('--mesh_shape', type=int, nargs=2, help='Data and model parallelism')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


if __name__ == '__main__':
    inference_main_distributed()