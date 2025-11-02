"""
Qwen 2.5 Model with Layer Activation Extraction Hooks
Complete implementation with all necessary components for activation capture
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import struct
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class QwenConfig:
    """Configuration for Qwen model"""
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
    attention_dropout: float = 0.0
    dtype: Any = jnp.float32
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 21


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0
    dtype: Any = jnp.float32

    def setup(self):
        # Create sinusoidal position embeddings
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int, dtype: Any = None):
        if dtype is None:
            dtype = self.dtype
            
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos[position_ids] if position_ids is not None else cos
    sin = sin[position_ids] if position_ids is not None else sin
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    dim: int
    eps: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Compute RMS
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.eps)
        
        # Apply learned scale
        scale = self.param(
            'scale',
            nn.initializers.ones,
            (self.dim,),
            self.dtype
        )
        return x * scale


class QwenMLP(nn.Module):
    """Qwen MLP (Feed-Forward Network)"""
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states):
        # Gate and up projection
        gate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='gate_proj'
        )
        up_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='up_proj'
        )
        down_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='down_proj'
        )
        
        # Apply projections with SiLU activation
        gate = gate_proj(hidden_states)
        gate = nn.silu(gate)  # SiLU activation
        up = up_proj(hidden_states)
        hidden_states = down_proj(gate * up)
        
        return hidden_states


class QwenAttention(nn.Module):
    """Multi-head attention with grouped query attention support"""
    config: QwenConfig
    layer_idx: int

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            dtype=self.config.dtype
        )

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections
        q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='q_proj'
        )
        k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='k_proj'
        )
        v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.config.dtype,
            name='v_proj'
        )
        o_proj = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.config.dtype,
            name='o_proj'
        )
        
        # Compute Q, K, V
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
        
        cos, sin = self.rotary_emb(seq_len, dtype=hidden_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos[None, :, None, :], sin[None, :, None, :]
        )
        
        # Repeat K, V for grouped query attention
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        # Compute attention scores
        attn_weights = jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2)))
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(self.config.dtype)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, value_states)
        
        # Transpose back: [batch, seq_len, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = o_proj(attn_output)
        
        return attn_output


class QwenDecoderLayer(nn.Module):
    """Transformer decoder layer"""
    config: QwenConfig
    layer_idx: int = 0

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None):
        # Store input for residual
        residual = hidden_states
        
        # Pre-norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='input_layernorm'
        )(hidden_states)
        
        # Self-attention
        hidden_states = QwenAttention(
            self.config,
            self.layer_idx,
            name='self_attn'
        )(hidden_states, attention_mask)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Store for next residual
        residual = hidden_states
        
        # Post-norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='post_attention_layernorm'
        )(hidden_states)
        
        # MLP
        hidden_states = QwenMLP(self.config, name='mlp')(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states


class QwenModelWithActivations(nn.Module):
    """
    Qwen model that returns intermediate layer activations
    
    Returns:
        - logits: Final output logits
        - activations: Dict mapping layer_idx -> hidden_states
    """
    config: QwenConfig
    layers_to_extract: Optional[List[int]] = None  # Which layers to extract (None = all)

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, return_activations=True):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.config.dtype,
            name='embed_tokens'
        )
        hidden_states = embed_tokens(input_ids)
        
        # Create causal mask if not provided
        if attention_mask is None:
            # Create proper causal mask
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attention_mask = jnp.where(causal_mask[None, None, :, :], 0.0, -1e9)
        
        # Store activations
        activations = {} if return_activations else None
        
        # Apply decoder layers
        for i in range(self.config.num_hidden_layers):
            # Store pre-layer activation if requested
            if return_activations and (self.layers_to_extract is None or i in self.layers_to_extract):
                activations[f'layer_{i}_input'] = hidden_states
            
            hidden_states = QwenDecoderLayer(
                self.config,
                layer_idx=i,
                name=f'layers_{i}'
            )(hidden_states, attention_mask)
            
            # Store post-layer activation if requested
            if return_activations and (self.layers_to_extract is None or i in self.layers_to_extract):
                activations[f'layer_{i}'] = hidden_states
        
        # Final norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='norm'
        )(hidden_states)
        
        # Store final layer activation
        if return_activations:
            if self.layers_to_extract is None or self.config.num_hidden_layers in self.layers_to_extract:
                activations[f'layer_{self.config.num_hidden_layers}_norm'] = hidden_states
        
        # LM head
        if self.config.tie_word_embeddings:
            # Reuse embedding weights for output
            lm_logits = hidden_states @ embed_tokens.embedding.T
        else:
            lm_head = nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                dtype=self.config.dtype,
                name='lm_head'
            )
            lm_logits = lm_head(hidden_states)
        
        if return_activations:
            return lm_logits, activations
        else:
            return lm_logits


def create_model_with_hooks(config: QwenConfig, layers_to_extract: Optional[List[int]] = None):
    """
    Create a Qwen model with activation extraction hooks
    
    Args:
        config: Model configuration
        layers_to_extract: List of layer indices to extract (None = all layers)
    
    Returns:
        QwenModelWithActivations instance
    """
    return QwenModelWithActivations(config=config, layers_to_extract=layers_to_extract)


def convert_hf_to_jax_weights(hf_model, config: QwenConfig) -> Dict[str, Any]:
    """
    Convert HuggingFace model weights to JAX format
    
    Args:
        hf_model: HuggingFace model instance
        config: QwenConfig
    
    Returns:
        Dict of JAX parameters
    """
    import torch
    
    params = {}
    
    # Convert embeddings
    if hasattr(hf_model.model, 'embed_tokens'):
        params['embed_tokens'] = {
            'embedding': jnp.array(hf_model.model.embed_tokens.weight.detach().numpy())
        }
    
    # Convert decoder layers
    for i in range(config.num_hidden_layers):
        layer_params = {}
        hf_layer = hf_model.model.layers[i]
        
        # Self-attention
        layer_params['self_attn'] = {
            'q_proj': {
                'kernel': jnp.array(hf_layer.self_attn.q_proj.weight.T.detach().numpy()),
                'bias': jnp.array(hf_layer.self_attn.q_proj.bias.detach().numpy())
                        if hf_layer.self_attn.q_proj.bias is not None else None
            },
            'k_proj': {
                'kernel': jnp.array(hf_layer.self_attn.k_proj.weight.T.detach().numpy()),
                'bias': jnp.array(hf_layer.self_attn.k_proj.bias.detach().numpy())
                        if hf_layer.self_attn.k_proj.bias is not None else None
            },
            'v_proj': {
                'kernel': jnp.array(hf_layer.self_attn.v_proj.weight.T.detach().numpy()),
                'bias': jnp.array(hf_layer.self_attn.v_proj.bias.detach().numpy())
                        if hf_layer.self_attn.v_proj.bias is not None else None
            },
            'o_proj': {
                'kernel': jnp.array(hf_layer.self_attn.o_proj.weight.T.detach().numpy())
            }
        }
        
        # MLP
        layer_params['mlp'] = {
            'gate_proj': {
                'kernel': jnp.array(hf_layer.mlp.gate_proj.weight.T.detach().numpy())
            },
            'up_proj': {
                'kernel': jnp.array(hf_layer.mlp.up_proj.weight.T.detach().numpy())
            },
            'down_proj': {
                'kernel': jnp.array(hf_layer.mlp.down_proj.weight.T.detach().numpy())
            }
        }
        
        # Layer norms
        layer_params['input_layernorm'] = {
            'scale': jnp.array(hf_layer.input_layernorm.weight.detach().numpy())
        }
        layer_params['post_attention_layernorm'] = {
            'scale': jnp.array(hf_layer.post_attention_layernorm.weight.detach().numpy())
        }
        
        params[f'layers_{i}'] = layer_params
    
    # Final norm
    params['norm'] = {
        'scale': jnp.array(hf_model.model.norm.weight.detach().numpy())
    }
    
    # LM head (if not tied)
    if not config.tie_word_embeddings:
        params['lm_head'] = {
            'kernel': jnp.array(hf_model.lm_head.weight.T.detach().numpy())
        }
    
    return params


def generate_with_layer_activations(
    model,
    params,
    input_ids,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    return_all_step_activations: bool = False
):
    """
    Generate tokens and extract layer activations at each generation step
    
    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        eos_token_id: End-of-sequence token ID
        return_all_step_activations: If True, return activations for every generation step
    
    Returns:
        generated_ids: Generated token IDs
        final_activations: Activations from final forward pass
        activations_per_step: List of dicts, one per generation step (if requested)
    """
    generated_ids = input_ids
    activations_per_step = [] if return_all_step_activations else None
    batch_size = input_ids.shape[0]
    
    for step in range(max_tokens):
        # Forward pass with activation extraction
        logits, activations = model.apply(
            params,
            generated_ids,
            return_activations=True
        )
        
        # Store activations for this step
        if return_all_step_activations:
            activations_per_step.append({
                'step': step,
                'activations': activations,
                'seq_len': generated_ids.shape[1]
            })
        
        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-p sampling if needed
        if top_p < 1.0:
            sorted_logits = jnp.sort(next_token_logits, axis=-1)[:, ::-1]
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff
            cutoff_idx = jnp.argmax(cumsum_probs > top_p, axis=-1)
            cutoff_logits = jnp.take_along_axis(sorted_logits, cutoff_idx[:, None], axis=-1)
            next_token_logits = jnp.where(
                next_token_logits < cutoff_logits,
                -1e9,
                next_token_logits
            )
        
        # Sample next token (greedy for temperature=0, otherwise sample)
        if temperature == 0:
            next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            key = jax.random.PRNGKey(step)
            next_token_id = jax.random.categorical(key, next_token_logits, axis=-1)[:, None]
        
        # Append to sequence
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        
        # Check for EOS token
        if eos_token_id is not None:
            eos_mask = (next_token_id == eos_token_id).squeeze(-1)
            if jnp.all(eos_mask):
                break
    
    # Final forward pass for activations
    _, final_activations = model.apply(params, generated_ids, return_activations=True)
    
    return generated_ids, final_activations, activations_per_step


def extract_specific_layer_positions(
    activations: Dict[str, jnp.ndarray],
    layer_name: str,
    token_positions: Optional[Union[List[int], slice]] = None
) -> jnp.ndarray:
    """
    Extract activations from specific token positions in a layer
    
    Args:
        activations: Dict from layer_name -> hidden_states [batch, seq_len, hidden_dim]
        layer_name: e.g., 'layer_12'
        token_positions: List of token positions, slice, or None (last token only)
    
    Returns:
        Extracted activations [batch, n_positions, hidden_dim]
    """
    if layer_name not in activations:
        available = list(activations.keys())
        raise ValueError(f"Layer {layer_name} not in activations. Available: {available}")
    
    layer_acts = activations[layer_name]  # [batch, seq_len, hidden_dim]
    
    if token_positions is None:
        # Extract last token only
        return layer_acts[:, -1:, :]
    elif isinstance(token_positions, slice):
        # Extract slice of positions
        return layer_acts[:, token_positions, :]
    else:
        # Extract specific positions
        return layer_acts[:, jnp.array(token_positions), :]


def analyze_activation_statistics(activations: Dict[str, jnp.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each layer's activations
    
    Args:
        activations: Dict of layer_name -> activations
    
    Returns:
        Dict of layer_name -> statistics dict
    """
    stats = {}
    
    for layer_name, acts in activations.items():
        # Flatten to [batch * seq_len, hidden_dim]
        flat_acts = acts.reshape(-1, acts.shape[-1])
        
        stats[layer_name] = {
            'mean': float(jnp.mean(flat_acts)),
            'std': float(jnp.std(flat_acts)),
            'min': float(jnp.min(flat_acts)),
            'max': float(jnp.max(flat_acts)),
            'sparsity': float(jnp.mean(jnp.abs(flat_acts) < 1e-6)),  # Fraction near zero
            'l2_norm': float(jnp.mean(jnp.linalg.norm(flat_acts, axis=-1))),
        }
    
    return stats


def get_activation_memory_usage(activations: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """
    Calculate memory usage of stored activations
    
    Args:
        activations: Dict of layer_name -> activations
    
    Returns:
        Dict of layer_name -> memory in MB
    """
    memory = {}
    for layer_name, acts in activations.items():
        memory[layer_name] = acts.nbytes / (1024 * 1024)  # Convert to MB
    
    memory['total'] = sum(memory.values())
    return memory


# Example usage configuration
DEFAULT_SAE_LAYERS = list(range(10, 24))  # Layers 10-23 for SAE training
DEFAULT_PROBE_LAYERS = [6, 12, 18, 23]  # Key layers for probing


if __name__ == "__main__":
    # Example: Create model with hooks
    config = QwenConfig(
        num_hidden_layers=24,
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2
    )
    model = create_model_with_hooks(config, layers_to_extract=DEFAULT_PROBE_LAYERS)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((2, 10), dtype=jnp.int32)
    variables = model.init(key, dummy_input)
    
    # Test forward pass
    logits, activations = model.apply(variables, dummy_input, return_activations=True)
    
    print("Model test results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Extracted layers: {list(activations.keys())}")
    for layer_name, acts in activations.items():
        print(f"    {layer_name}: shape={acts.shape}")
    
    # Test activation statistics
    stats = analyze_activation_statistics(activations)
    print("\nActivation statistics:")
    for layer_name, layer_stats in stats.items():
        print(f"  {layer_name}:")
        for stat_name, value in layer_stats.items():
            print(f"    {stat_name}: {value:.4f}")
    
    # Test memory usage
    memory = get_activation_memory_usage(activations)
    print(f"\nMemory usage:")
    for layer_name, mem in memory.items():
        print(f"  {layer_name}: {mem:.2f} MB")