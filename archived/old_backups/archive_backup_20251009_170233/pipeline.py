"""
Complete Distributed Inference Pipeline for JAX/TPU with Activation Extraction
Clean architecture with proper JAX patterns and modular design
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Union
import argparse
from functools import partial
from itertools import islice, product

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.tree_util import tree_map_with_path
import flax.linen as nn
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Qwen model configuration"""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    dtype: Any = jnp.float32


@dataclass
class InferenceConfig:
    """Inference pipeline configuration"""
    # Model paths
    model_path: str = "KathirKs/qwen-2.5-0.5b"
    
    # Data paths
    dataset_path: str = "arc_data.json"
    output_path: str = "submission.json"
    activations_dir: str = "./activations"
    
    # Generation settings
    max_output_tokens: int = 1100
    temperature: float = 0.0
    batch_size: int = 8
    
    # Task settings
    n_tasks: Optional[int] = None
    predictions_per_task: int = 8
    prompt_version: str = "output-from-examples-v0"
    grid_encoder: str = "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))"
    
    # Distributed settings
    mesh_shape: Tuple[int, int] = (1, 1)  # (data_parallel, model_parallel)
    
    # Activation extraction
    extract_activations: bool = True
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))
    save_frequency: int = 10
    
    # Misc
    seed: int = 42
    verbose: bool = False


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings"""
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0
    
    def setup(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = inv_freq
    
    def __call__(self, seq_len: int, dtype=jnp.float32):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def rotate_half(x):
    """Rotate half the hidden dims"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    dim: int
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.eps)
        scale = self.param('scale', nn.initializers.ones, (self.dim,))
        return x * scale


class Attention(nn.Module):
    """Multi-head attention with grouped query attention"""
    config: ModelConfig
    
    def setup(self):
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.rotary = RotaryEmbedding(
            self.head_dim,
            self.config.max_position_embeddings,
            self.config.rope_theta
        )
    
    @nn.compact
    def __call__(self, hidden_states, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        
        # QKV projections
        q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=True, name='q_proj')
        k_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, name='k_proj')
        v_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, name='v_proj')
        o_proj = nn.Dense(self.config.hidden_size, use_bias=False, name='o_proj')
        
        # Compute Q, K, V
        q = q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k_proj(hidden_states).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v_proj(hidden_states).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(seq_len, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos[None, :, None, :], sin[None, :, None, :])
        
        # Repeat KV heads if using GQA
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)
        
        # Reshape for attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Compute attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return o_proj(attn_output)


class MLP(nn.Module):
    """Feed-forward network"""
    config: ModelConfig
    
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.config.intermediate_size, use_bias=False, name='gate_proj')(x)
        gate = nn.silu(gate)
        up = nn.Dense(self.config.intermediate_size, use_bias=False, name='up_proj')(x)
        down = nn.Dense(self.config.hidden_size, use_bias=False, name='down_proj')(gate * up)
        return down


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    config: ModelConfig
    layer_idx: int
    
    @nn.compact
    def __call__(self, hidden_states, attention_mask=None):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps, 
                                name='input_layernorm')(hidden_states)
        hidden_states = Attention(self.config, name='self_attn')(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='post_attention_layernorm')(hidden_states)
        hidden_states = MLP(self.config, name='mlp')(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class QwenModel(nn.Module):
    """Complete Qwen model with optional activation extraction"""
    config: ModelConfig
    extract_layers: Optional[List[int]] = None
    
    @nn.compact
    def __call__(self, input_ids, return_activations=False):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        embed = nn.Embed(self.config.vocab_size, self.config.hidden_size, 
                        name='embed_tokens')
        hidden_states = embed(input_ids)
        
        # Causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attention_mask = jnp.where(causal_mask[None, None, :, :], 0.0, -1e9)
        
        # Store activations if requested
        activations = {} if return_activations else None
        
        # Transformer blocks
        for i in range(self.config.num_hidden_layers):
            if return_activations and (self.extract_layers is None or i in self.extract_layers):
                activations[f'layer_{i}_input'] = hidden_states
            
            hidden_states = TransformerBlock(
                self.config, i, name=f'layers_{i}'
            )(hidden_states, attention_mask)
            
            if return_activations and (self.extract_layers is None or i in self.extract_layers):
                activations[f'layer_{i}_output'] = hidden_states
        
        # Final norm
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='norm')(hidden_states)
        
        # Output projection
        if self.config.tie_word_embeddings:
            logits = hidden_states @ embed.embedding.T
        else:
            lm_head = nn.Dense(self.config.vocab_size, use_bias=False, name='lm_head')
            logits = lm_head(hidden_states)
        
        if return_activations:
            return logits, activations
        return logits


# ============================================================================
# GENERATION ENGINE
# ============================================================================

class GenerationEngine:
    """Handles text generation with proper JAX patterns"""
    
    def __init__(self, model, config: InferenceConfig):
        self.model = model
        self.config = config
    
    def create_generate_fn(self, mesh: Optional[Mesh] = None):
        """Create JIT-compiled generation function"""
        
        def generate_tokens(params, input_ids, max_new_tokens):
            """Generate tokens using scan for efficiency"""
            batch_size = input_ids.shape[0]
            
            def scan_fn(carry, _):
                generated_ids, position = carry
                
                # Get sequence up to current position
                current_seq = lax.dynamic_slice(
                    generated_ids, (0, 0), (batch_size, position)
                )
                
                # Forward pass
                logits = self.model.apply(params, current_seq)
                
                # Greedy sampling (temp=0)
                next_tokens = jnp.argmax(logits[:, -1, :], axis=-1)
                
                # Update sequence
                generated_ids = lax.dynamic_update_slice(
                    generated_ids, next_tokens[:, None], (0, position)
                )
                
                return (generated_ids, position + 1), None
            
            # Prepare for scan
            init_len = input_ids.shape[1]
            max_len = init_len + max_new_tokens
            
            # Initialize with padding
            generated = jnp.zeros((batch_size, max_len), dtype=jnp.int32)
            generated = generated.at[:, :init_len].set(input_ids)
            
            # Run generation
            (final_ids, final_pos), _ = lax.scan(
                scan_fn, (generated, init_len), None, length=max_new_tokens
            )
            
            return final_ids[:, :final_pos]
        
        # Apply sharding if mesh is provided
        if mesh is not None:
            in_shardings = (
                None,  # params - replicated or custom sharding
                NamedSharding(mesh, P('data', None)),  # input_ids
                None   # max_new_tokens
            )
            out_sharding = NamedSharding(mesh, P('data', None))
            
            return jax.jit(
                generate_tokens,
                in_shardings=in_shardings,
                out_shardings=out_sharding,
                static_argnums=(2,)
            )
        else:
            return jax.jit(generate_tokens, static_argnums=(2,))
    
    def generate_with_activations(self, params, input_ids, max_new_tokens):
        """Generate with activation extraction (not JIT-compiled)"""
        batch_size = input_ids.shape[0]
        generated = input_ids
        all_activations = []
        
        for _ in range(max_new_tokens):
            logits, activations = self.model.apply(
                params, generated, return_activations=True
            )
            all_activations.append(activations)
            
            next_tokens = jnp.argmax(logits[:, -1, :], axis=-1)
            generated = jnp.concatenate([generated, next_tokens[:, None]], axis=1)
        
        return generated, all_activations


# ============================================================================
# ACTIVATION MANAGER
# ============================================================================

class ActivationManager:
    """Manages extraction and storage of activations"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.buffer = {}
        self.batch_count = 0
        os.makedirs(config.activations_dir, exist_ok=True)
        
        self.metadata = {
            'model_path': config.model_path,
            'layers': config.layers_to_extract,
            'batches': []
        }
    
    def add_batch(self, activations: List[Dict], batch_info: List[Dict]):
        """Add a batch of activations to buffer"""
        for step_idx, step_acts in enumerate(activations):
            for layer_name, layer_acts in step_acts.items():
                if layer_name not in self.buffer:
                    self.buffer[layer_name] = []
                
                # Convert to numpy and store with metadata
                acts_np = np.array(layer_acts)
                for i, info in enumerate(batch_info):
                    if i < acts_np.shape[0]:
                        self.buffer[layer_name].append({
                            'activation': acts_np[i],
                            'task_id': info.get('task_id'),
                            'sample_idx': info.get('idx'),
                            'step': step_idx
                        })
        
        self.batch_count += 1
        if self.batch_count % self.config.save_frequency == 0:
            self.save()
    
    def save(self):
        """Save buffer to disk"""
        if not self.buffer:
            return
        
        for layer_name, activations in self.buffer.items():
            filename = f"{layer_name}_batch_{self.batch_count:06d}.pkl"
            filepath = os.path.join(self.config.activations_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(activations, f)
            
            self.metadata['batches'].append({
                'batch': self.batch_count,
                'layer': layer_name,
                'file': filename,
                'samples': len(activations)
            })
        
        self.buffer = {}
        print(f"Saved activations for batch {self.batch_count}")
    
    def finalize(self):
        """Save remaining activations and metadata"""
        self.save()
        
        metadata_path = os.path.join(self.config.activations_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed(config: InferenceConfig) -> Tuple[Mesh, Dict]:
    """Setup JAX mesh and device assignment"""
    devices = jax.devices()
    n_devices = len(devices)
    
    # Validate mesh configuration
    mesh_size = config.mesh_shape[0] * config.mesh_shape[1]
    if mesh_size != n_devices:
        raise ValueError(f"Mesh size {mesh_size} != device count {n_devices}")
    
    # Create mesh
    device_mesh = np.array(devices).reshape(config.mesh_shape)
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))
    
    print(f"Created mesh: {mesh}")
    print(f"  Data parallel: {config.mesh_shape[0]}")
    print(f"  Model parallel: {config.mesh_shape[1]}")
    
    return mesh, {'n_devices': n_devices, 'devices': devices}


def shard_model_params(params: Dict, mesh: Mesh) -> Dict:
    """Apply sharding to model parameters"""
    
    def get_sharding(path, param):
        """Determine sharding for a parameter"""
        name = '/'.join(str(k) for k in path)
        shape = param.shape
        
        # Embedding and output layers
        if 'embed' in name or 'lm_head' in name:
            if len(shape) == 2:
                return NamedSharding(mesh, P(None, 'model'))
        
        # Linear layers
        elif len(shape) == 2:
            return NamedSharding(mesh, P(None, 'model'))
        
        # Default: replicate
        return NamedSharding(mesh, P(*([None] * len(shape))))
    
    return tree_map_with_path(
        lambda path, x: jax.device_put(x, get_sharding(path, x)),
        params
    )


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_prepare_data(config: InferenceConfig):
    """Load dataset and prepare for inference"""
    # This is a placeholder - integrate with your actual data loading
    with open(config.dataset_path) as f:
        data = json.load(f)
    
    if config.n_tasks:
        data = dict(islice(data.items(), config.n_tasks))
    
    return data


def create_prompts(data: Dict, config: InferenceConfig):
    """Create prompts from data"""
    # Placeholder - integrate with your prompt creation logic
    prompts = []
    for task_id, task in data.items():
        for i in range(config.predictions_per_task):
            prompts.append({
                'task_id': task_id,
                'idx': i,
                'text': f"Task {task_id} prompt {i}",  # Replace with actual prompt
                'augmentation': {}
            })
    return prompts


def batch_data(items: List, batch_size: int) -> List[List]:
    """Create batches from items"""
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Pad if necessary
        while len(batch) < batch_size:
            batch.append(batch[-1])
        batches.append(batch)
    return batches


# ============================================================================
# WEIGHT CONVERSION
# ============================================================================

def convert_hf_weights(hf_model, config: ModelConfig) -> Dict:
    """Convert HuggingFace weights to JAX format"""
    import torch
    
    params = {}
    
    # Embeddings
    if hasattr(hf_model.model, 'embed_tokens'):
        params['embed_tokens'] = {
            'embedding': jnp.array(hf_model.model.embed_tokens.weight.detach().cpu().numpy())
        }
    
    # Transformer layers
    for i in range(config.num_hidden_layers):
        layer = hf_model.model.layers[i]
        layer_params = {}
        
        # Attention
        layer_params['self_attn'] = {
            'q_proj': {
                'kernel': jnp.array(layer.self_attn.q_proj.weight.T.detach().cpu().numpy()),
                'bias': jnp.array(layer.self_attn.q_proj.bias.detach().cpu().numpy())
                        if layer.self_attn.q_proj.bias is not None else None
            },
            'k_proj': {
                'kernel': jnp.array(layer.self_attn.k_proj.weight.T.detach().cpu().numpy()),
                'bias': jnp.array(layer.self_attn.k_proj.bias.detach().cpu().numpy())
                        if layer.self_attn.k_proj.bias is not None else None
            },
            'v_proj': {
                'kernel': jnp.array(layer.self_attn.v_proj.weight.T.detach().cpu().numpy()),
                'bias': jnp.array(layer.self_attn.v_proj.bias.detach().cpu().numpy())
                        if layer.self_attn.v_proj.bias is not None else None
            },
            'o_proj': {
                'kernel': jnp.array(layer.self_attn.o_proj.weight.T.detach().cpu().numpy())
            }
        }
        
        # MLP
        layer_params['mlp'] = {
            'gate_proj': {'kernel': jnp.array(layer.mlp.gate_proj.weight.T.detach().cpu().numpy())},
            'up_proj': {'kernel': jnp.array(layer.mlp.up_proj.weight.T.detach().cpu().numpy())},
            'down_proj': {'kernel': jnp.array(layer.mlp.down_proj.weight.T.detach().cpu().numpy())}
        }
        
        # Norms
        layer_params['input_layernorm'] = {
            'scale': jnp.array(layer.input_layernorm.weight.detach().cpu().numpy())
        }
        layer_params['post_attention_layernorm'] = {
            'scale': jnp.array(layer.post_attention_layernorm.weight.detach().cpu().numpy())
        }
        
        params[f'layers_{i}'] = layer_params
    
    # Final norm
    params['norm'] = {
        'scale': jnp.array(hf_model.model.norm.weight.detach().cpu().numpy())
    }
    
    # LM head
    if not config.tie_word_embeddings:
        params['lm_head'] = {
            'kernel': jnp.array(hf_model.lm_head.weight.T.detach().cpu().numpy())
        }
    
    return {'params': params}


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Main inference pipeline"""
    # Load configuration
    config = InferenceConfig(**vars(args))
    
    print("="*70)
    print("DISTRIBUTED INFERENCE PIPELINE")
    print("="*70)
    print(json.dumps(asdict(config), indent=2))
    print("="*70)
    
    # Setup
    jax.config.update('jax_platform_name', 'tpu')
    key = jax.random.PRNGKey(config.seed)
    
    # Setup distributed
    mesh, device_info = setup_distributed(config)
    
    with mesh:
        # Load model
        print("\n1. Loading model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
        # Create JAX model
        model_config = ModelConfig()
        if config.extract_activations:
            model = QwenModel(model_config, extract_layers=config.layers_to_extract)
        else:
            model = QwenModel(model_config)
        
        # Load weights
        print("   Loading HF weights...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        params = convert_hf_weights(hf_model, model_config)
        del hf_model  # Free memory
        
        # Shard parameters
        print("   Sharding parameters...")
        params = shard_model_params(params, mesh)
        
        # Setup generation
        print("\n2. Setting up generation...")
        gen_engine = GenerationEngine(model, config)
        
        if config.extract_activations:
            activation_manager = ActivationManager(config)
            generate_fn = gen_engine.generate_with_activations
        else:
            generate_fn = gen_engine.create_generate_fn(mesh)
            activation_manager = None
        
        # Load data
        print("\n3. Loading data...")
        data = load_and_prepare_data(config)
        prompts = create_prompts(data, config)
        
        print(f"   Loaded {len(data)} tasks")
        print(f"   Created {len(prompts)} prompts")
        
        # Tokenize prompts
        print("\n4. Tokenizing prompts...")
        tokenized = []
        for prompt in tqdm(prompts, desc="Tokenizing"):
            ids = tokenizer.encode(prompt['text'], return_tensors='np')
            if ids.ndim == 2:
                ids = ids[0]
            tokenized.append({
                **prompt,
                'input_ids': ids
            })
        
        # Create batches
        batches = batch_data(tokenized, config.batch_size)
        print(f"   Created {len(batches)} batches")
        
        # Run inference
        print("\n5. Running inference...")
        all_predictions = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Inference")):
            # Prepare batch inputs
            max_len = max(len(item['input_ids']) for item in batch)
            batch_ids = []
            
            for item in batch:
                ids = item['input_ids']
                padded = np.pad(ids, (0, max_len - len(ids)), 
                              constant_values=tokenizer.pad_token_id)
                batch_ids.append(padded)
            
            batch_ids = jnp.stack(batch_ids)
            
            # Generate
            if config.extract_activations:
                generated, activations = generate_fn(
                    params, batch_ids, config.max_output_tokens
                )
                activation_manager.add_batch(activations, batch)
            else:
                generated = generate_fn(
                    params, batch_ids, config.max_output_tokens
                )
            
            # Decode
            for i, item in enumerate(batch):
                if i < generated.shape[0]:
                    input_len = len(item['input_ids'])
                    output_ids = generated[i, input_len:]
                    text = tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    all_predictions.append({
                        'task_id': item['task_id'],
                        'idx': item['idx'],
                        'prediction': text,
                        'augmentation': item.get('augmentation', {})
                    })
        
        # Finalize
        if activation_manager:
            activation_manager.finalize()
        
        # Save predictions
        print(f"\n6. Saving {len(all_predictions)} predictions...")
        os.makedirs(os.path.dirname(config.output_path) or '.', exist_ok=True)
        
        # Process predictions into submission format
        solutions = {}
        for pred in all_predictions:
            task_id = pred['task_id']
            if task_id not in solutions:
                solutions[task_id] = []
            solutions[task_id].append(pred['prediction'])
        
        with open(config.output_path, 'w') as f:
            json.dump(solutions, f, indent=2)
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETE")
        print(f"Predictions: {config.output_path}")
        if config.extract_activations:
            print(f"Activations: {config.activations_dir}")
        print("="*70)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Distributed Inference Pipeline")
    
    # Model
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-0.5B')
    
    # Data
    parser.add_argument('--dataset_path', type=str, default='arc_data.json')
    parser.add_argument('--output_path', type=str, default='submission.json')
    parser.add_argument('--n_tasks', type=int, help='Limit number of tasks')
    
    # Generation
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_output_tokens', type=int, default=1100)
    parser.add_argument('--temperature', type=float, default=0.0)
    
    # Distributed
    parser.add_argument('--mesh_shape', type=int, nargs=2, default=[1, 1])
    
    # Activations
    parser.add_argument('--extract_activations', action='store_true')
    parser.add_argument('--activations_dir', type=str, default='./activations')
    parser.add_argument('--layers_to_extract', type=int, nargs='+')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)