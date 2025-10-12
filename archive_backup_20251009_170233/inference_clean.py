"""
Clean JAX Inference Pipeline with JIT Activation Extraction and Shardmap
Production-ready implementation with proper error handling
"""

import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import flax.linen as nn
from flax.core import FrozenDict

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    dtype: Any = jnp.float32


@dataclass
class InferenceConfig:
    """Complete inference configuration"""
    # Model
    model_path: str = "KathirKs/qwen-2.5-0.5b"

    # Data
    dataset_path: str = "arc_data.json"
    output_path: str = "submission.json"

    # Generation
    max_new_tokens: int = 512
    batch_size: int = 8

    # Distributed
    mesh_shape: Tuple[int, int] = (1, 1)  # (data, model)

    # Activations
    extract_activations: bool = False
    activations_dir: str = "./activations"
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))

    # Misc
    seed: int = 42


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.eps)
        scale = self.param('scale', nn.initializers.ones, (self.dim,))
        return x * scale


def rotate_half(x):
    """Rotate half the hidden dimensions"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings"""
    dim: int
    max_position_embeddings: int = 32768
    base: float = 10000.0

    def setup(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int, dtype=jnp.float32):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


class Attention(nn.Module):
    """Multi-head attention with GQA"""
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
    def __call__(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        # QKV projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=True, name='q_proj')(hidden_states)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, name='k_proj')(hidden_states)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, name='v_proj')(hidden_states)

        # Reshape
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(seq_len, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos[None, :, None, :], sin[None, :, None, :])

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)

        # Transpose for attention
        q = q.transpose(0, 2, 1, 3)  # [B, H, S, D]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        # Attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        scores = jnp.where(mask[None, None, :, :], scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return nn.Dense(self.config.hidden_size, use_bias=False, name='o_proj')(attn_output)


class MLP(nn.Module):
    """Feed-forward network"""
    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.config.intermediate_size, use_bias=False, name='gate_proj')(x)
        up = nn.Dense(self.config.intermediate_size, use_bias=False, name='up_proj')(x)
        down = nn.Dense(self.config.hidden_size, use_bias=False, name='down_proj')(nn.silu(gate) * up)
        return down


class TransformerBlock(nn.Module):
    """Transformer decoder block"""
    config: ModelConfig
    layer_idx: int

    @nn.compact
    def __call__(self, hidden_states):
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='input_layernorm')(hidden_states)
        hidden_states = Attention(self.config, name='self_attn')(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='post_attention_layernorm')(hidden_states)
        hidden_states = MLP(self.config, name='mlp')(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QwenModel(nn.Module):
    """Complete Qwen model"""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids):
        # Embeddings
        hidden_states = nn.Embed(self.config.vocab_size, self.config.hidden_size,
                                 name='embed_tokens')(input_ids)

        # Transformer layers
        for i in range(self.config.num_hidden_layers):
            hidden_states = TransformerBlock(self.config, i, name=f'layers_{i}')(hidden_states)

        # Final norm
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='norm')(hidden_states)

        # Output projection
        if self.config.tie_word_embeddings:
            # Use embedding weights
            embed_kernel = self.variables['params']['embed_tokens']['embedding']
            logits = jnp.dot(hidden_states, embed_kernel.T)
        else:
            logits = nn.Dense(self.config.vocab_size, use_bias=False, name='lm_head')(hidden_states)

        return logits


# ============================================================================
# ACTIVATION EXTRACTION MODEL
# ============================================================================

class QwenModelWithActivations(nn.Module):
    """Qwen model that extracts intermediate activations"""
    config: ModelConfig
    extract_layers: List[int]

    @nn.compact
    def __call__(self, input_ids):
        activations = {}

        # Embeddings
        hidden_states = nn.Embed(self.config.vocab_size, self.config.hidden_size,
                                 name='embed_tokens')(input_ids)

        # Transformer layers
        for i in range(self.config.num_hidden_layers):
            if i in self.extract_layers:
                activations[f'layer_{i}_input'] = hidden_states

            hidden_states = TransformerBlock(self.config, i, name=f'layers_{i}')(hidden_states)

            if i in self.extract_layers:
                activations[f'layer_{i}_output'] = hidden_states

        # Final norm
        hidden_states = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps,
                                name='norm')(hidden_states)

        # Output projection
        if self.config.tie_word_embeddings:
            embed_kernel = self.variables['params']['embed_tokens']['embedding']
            logits = jnp.dot(hidden_states, embed_kernel.T)
        else:
            logits = nn.Dense(self.config.vocab_size, use_bias=False, name='lm_head')(hidden_states)

        return logits, activations


# ============================================================================
# GENERATION WITH JIT
# ============================================================================

def create_generation_step(model, extract_activations=False):
    """Create a single JIT-compiled generation step"""

    if extract_activations:
        @jax.jit
        def generation_step(params, input_ids):
            """Single step: forward pass + greedy sampling + activations"""
            logits, activations = model.apply(params, input_ids)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            return next_token, activations
    else:
        @jax.jit
        def generation_step(params, input_ids):
            """Single step: forward pass + greedy sampling"""
            logits = model.apply(params, input_ids)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            return next_token

    return generation_step


def generate_tokens(params, input_ids, max_new_tokens, generation_step, extract_activations=False):
    """Generate tokens with JIT-compiled steps"""
    generated = input_ids
    all_activations = [] if extract_activations else None

    for _ in range(max_new_tokens):
        if extract_activations:
            next_token, activations = generation_step(params, generated)
            all_activations.append(activations)
        else:
            next_token = generation_step(params, generated)

        generated = jnp.concatenate([generated, next_token], axis=1)

    if extract_activations:
        return generated, all_activations
    return generated


# ============================================================================
# SHARDMAP-BASED BATCH PROCESSING
# ============================================================================

def create_sharded_generation_fn(model, mesh, extract_activations=False):
    """Create sharded generation function using shard_map"""

    generation_step = create_generation_step(model, extract_activations)

    def per_device_generate(params, input_ids, max_new_tokens):
        """Generate on a single device (called via shard_map)"""
        return generate_tokens(params, input_ids, max_new_tokens, generation_step, extract_activations)

    # Shard across data dimension
    sharded_fn = shard_map(
        per_device_generate,
        mesh=mesh,
        in_specs=(P(), P('data', None), P()),  # params replicated, inputs sharded on batch
        out_specs=P('data', None) if not extract_activations else (P('data', None), P('data'))
    )

    return sharded_fn


# ============================================================================
# WEIGHT CONVERSION
# ============================================================================

def convert_hf_to_jax(hf_model_path: str, config: ModelConfig) -> Dict:
    """Convert HuggingFace model to JAX params"""
    from transformers import AutoModelForCausalLM
    import torch

    print(f"Loading HF model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    params = {}

    # Embeddings
    params['embed_tokens'] = {
        'embedding': jnp.array(hf_model.model.embed_tokens.weight.detach().cpu().numpy())
    }

    # Layers
    for i in range(config.num_hidden_layers):
        layer = hf_model.model.layers[i]

        params[f'layers_{i}'] = {
            'self_attn': {
                'q_proj': {
                    'kernel': jnp.array(layer.self_attn.q_proj.weight.T.detach().cpu().numpy()),
                    'bias': jnp.array(layer.self_attn.q_proj.bias.detach().cpu().numpy())
                },
                'k_proj': {
                    'kernel': jnp.array(layer.self_attn.k_proj.weight.T.detach().cpu().numpy()),
                    'bias': jnp.array(layer.self_attn.k_proj.bias.detach().cpu().numpy())
                },
                'v_proj': {
                    'kernel': jnp.array(layer.self_attn.v_proj.weight.T.detach().cpu().numpy()),
                    'bias': jnp.array(layer.self_attn.v_proj.bias.detach().cpu().numpy())
                },
                'o_proj': {
                    'kernel': jnp.array(layer.self_attn.o_proj.weight.T.detach().cpu().numpy())
                }
            },
            'mlp': {
                'gate_proj': {'kernel': jnp.array(layer.mlp.gate_proj.weight.T.detach().cpu().numpy())},
                'up_proj': {'kernel': jnp.array(layer.mlp.up_proj.weight.T.detach().cpu().numpy())},
                'down_proj': {'kernel': jnp.array(layer.mlp.down_proj.weight.T.detach().cpu().numpy())}
            },
            'input_layernorm': {
                'scale': jnp.array(layer.input_layernorm.weight.detach().cpu().numpy())
            },
            'post_attention_layernorm': {
                'scale': jnp.array(layer.post_attention_layernorm.weight.detach().cpu().numpy())
            }
        }

    # Final norm
    params['norm'] = {
        'scale': jnp.array(hf_model.model.norm.weight.detach().cpu().numpy())
    }

    # LM head (if not tied)
    if not config.tie_word_embeddings:
        params['lm_head'] = {
            'kernel': jnp.array(hf_model.lm_head.weight.T.detach().cpu().numpy())
        }

    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return FrozenDict({'params': params})


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_mesh(config: InferenceConfig) -> Mesh:
    """Create device mesh for distributed execution"""
    devices = jax.devices()
    n_devices = len(devices)

    mesh_size = config.mesh_shape[0] * config.mesh_shape[1]
    if mesh_size != n_devices:
        print(f"Warning: mesh size {mesh_size} != device count {n_devices}")
        # Auto-adjust: use all devices for data parallelism
        config.mesh_shape = (n_devices, 1)

    device_array = mesh_utils.create_device_mesh(config.mesh_shape)
    mesh = Mesh(device_array, axis_names=('data', 'model'))

    print(f"Mesh created: {config.mesh_shape} (data={config.mesh_shape[0]}, model={config.mesh_shape[1]})")
    return mesh


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """Complete inference pipeline with clean API"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model_config = ModelConfig()
        self.mesh = None
        self.params = None
        self.model = None
        self.tokenizer = None
        self.generate_fn = None

    def setup(self):
        """Initialize all components"""
        print("="*70)
        print("INITIALIZING INFERENCE PIPELINE")
        print("="*70)

        # Setup JAX
        jax.config.update('jax_platform_name', 'tpu')

        # Create mesh
        self.mesh = setup_mesh(self.config)

        # Load tokenizer
        print("\nLoading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        # Create model
        print("Creating model...")
        if self.config.extract_activations:
            self.model = QwenModelWithActivations(
                self.model_config,
                extract_layers=self.config.layers_to_extract
            )
        else:
            self.model = QwenModel(self.model_config)

        # Load weights
        print("Converting weights...")
        self.params = convert_hf_to_jax(self.config.model_path, self.model_config)

        # Create generation function
        print("Creating generation function...")
        with self.mesh:
            if self.config.mesh_shape[0] > 1:  # Use shard_map for multi-device
                self.generate_fn = create_sharded_generation_fn(
                    self.model, self.mesh, self.config.extract_activations
                )
            else:
                # Single device - just use regular JIT
                self.generate_fn = create_generation_step(self.model, self.config.extract_activations)

        print("\nSetup complete!")
        print("="*70)

    def generate(self, prompts: List[str]) -> List[Dict]:
        """Run inference on a list of prompts"""
        print(f"\nGenerating for {len(prompts)} prompts...")

        # Tokenize
        tokenized = [self.tokenizer.encode(p, return_tensors='np')[0] for p in prompts]

        # Batch
        batches = self._create_batches(tokenized)

        results = []
        activations_all = [] if self.config.extract_activations else None

        for batch_ids in batches:
            # Pad batch
            max_len = max(len(ids) for ids in batch_ids)
            padded = np.array([
                np.pad(ids, (0, max_len - len(ids)), constant_values=self.tokenizer.pad_token_id)
                for ids in batch_ids
            ])

            input_ids = jnp.array(padded)

            # Generate
            with self.mesh:
                if self.config.extract_activations:
                    if self.config.mesh_shape[0] > 1:
                        generated, activations = self.generate_fn(
                            self.params, input_ids, self.config.max_new_tokens
                        )
                    else:
                        generated, activations = generate_tokens(
                            self.params, input_ids, self.config.max_new_tokens,
                            self.generate_fn, extract_activations=True
                        )
                    activations_all.extend(activations)
                else:
                    if self.config.mesh_shape[0] > 1:
                        generated = self.generate_fn(
                            self.params, input_ids, self.config.max_new_tokens
                        )
                    else:
                        generated = generate_tokens(
                            self.params, input_ids, self.config.max_new_tokens,
                            self.generate_fn, extract_activations=False
                        )

            # Decode
            for i, orig_ids in enumerate(batch_ids):
                if i < generated.shape[0]:
                    output_ids = generated[i, len(orig_ids):]
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    results.append({'prompt_idx': len(results), 'output': text})

        # Save activations if extracted
        if self.config.extract_activations and activations_all:
            self._save_activations(activations_all)

        return results

    def _create_batches(self, tokenized: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Create batches from tokenized inputs"""
        batches = []
        for i in range(0, len(tokenized), self.config.batch_size):
            batch = tokenized[i:i + self.config.batch_size]
            # Pad last batch if needed
            while len(batch) < self.config.batch_size:
                batch.append(batch[-1])  # Duplicate last item
            batches.append(batch)
        return batches

    def _save_activations(self, activations: List[Dict]):
        """Save extracted activations to disk"""
        os.makedirs(self.config.activations_dir, exist_ok=True)

        filepath = os.path.join(self.config.activations_dir, 'activations.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(activations, f)

        print(f"Saved activations to {filepath}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Clean JAX Inference Pipeline")

    # Model
    parser.add_argument('--model_path', type=str, default='KathirKs/qwen-2.5-0.5b')

    # Generation
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=512)

    # Distributed
    parser.add_argument('--mesh_shape', type=int, nargs=2, default=[1, 1])

    # Activations
    parser.add_argument('--extract_activations', action='store_true')
    parser.add_argument('--activations_dir', type=str, default='./activations')
    parser.add_argument('--layers_to_extract', type=int, nargs='+', default=list(range(10, 24)))

    # Data
    parser.add_argument('--dataset_path', type=str, default='arc_data.json')
    parser.add_argument('--output_path', type=str, default='submission.json')

    args = parser.parse_args()

    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        mesh_shape=tuple(args.mesh_shape),
        extract_activations=args.extract_activations,
        activations_dir=args.activations_dir,
        layers_to_extract=args.layers_to_extract,
        dataset_path=args.dataset_path,
        output_path=args.output_path
    )

    # Create pipeline
    pipeline = InferencePipeline(config)
    pipeline.setup()

    # Example usage
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]

    results = pipeline.generate(test_prompts)

    for r in results:
        print(f"\nPrompt {r['prompt_idx']}: {test_prompts[r['prompt_idx']]}")
        print(f"Output: {r['output']}")

    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
