"""
Qwen Model with Activation Extraction Hooks

Provides model variants that can capture intermediate activations
during forward passes, useful for:
- SAE (Sparse Autoencoder) training
- Interpretability research
- Activation analysis

Key Features:
- Supports KV cache for fast generation
- Optional activation extraction from specific layers
- Configurable activation types (mlp, attn, residual)
- JIT-compatible with proper static argument handling
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, List
import time

from .config import QwenConfig
from .qwen import RMSNorm, QwenMLP, QwenAttention


class QwenDecoderLayerWithHooks(nn.Module):
    """
    Qwen decoder layer with optional activation capture.
    
    Can return intermediate activations at various points:
    - attn: Attention output before residual
    - mlp: MLP output before residual
    - residual: Final output after both residuals
    
    Attributes:
        config: Model configuration
    """
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None,
                 position_offset=0, return_activations=False):
        residual = hidden_states

        # Self Attention with pre-attention norm
        hidden_states_norm = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='input_layernorm'
        )(hidden_states)

        hidden_states_attn, new_kv_cache = QwenAttention(
            self.config,
            name='self_attn'
        )(hidden_states_norm, attention_mask, kv_cache, position_offset)

        # Store attention output BEFORE residual if requested
        attn_activation = hidden_states_attn if return_activations else None

        hidden_states = residual + hidden_states_attn

        # Store post-attention activation if requested
        post_attn_activation = hidden_states if return_activations else None

        # MLP with post-attention norm
        residual = hidden_states
        hidden_states_norm = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='post_attention_layernorm'
        )(hidden_states)

        hidden_states_mlp = QwenMLP(self.config, name='mlp')(hidden_states_norm)
        
        # Store MLP output BEFORE residual if requested
        mlp_activation = hidden_states_mlp if return_activations else None
        
        hidden_states = residual + hidden_states_mlp

        # Store post-MLP activation if requested (this is the layer output / residual)
        layer_output_activation = hidden_states if return_activations else None

        # Return based on what's requested
        if return_activations:
            activations = {
                'attn': attn_activation,
                'mlp': mlp_activation,
                'residual': layer_output_activation,
                'post_attn': post_attn_activation,
                'layer_output': layer_output_activation
            }
            return hidden_states, new_kv_cache, activations
        else:
            return hidden_states, new_kv_cache, None


class QwenModelWithActivations(nn.Module):
    """
    Qwen model that supports both KV caching and activation extraction.
    
    Operating modes:
    1. Fast inference (return_activations=False, with KV cache)
    2. Normal inference (return_activations=False, without KV cache)
    3. Activation extraction (return_activations=True)
    
    Attributes:
        config: Model configuration
        layers_to_extract: Tuple of layer indices to extract (None = all)
        activation_type: Type of activation ('mlp', 'attn', or 'residual')
    """
    config: QwenConfig
    layers_to_extract: Optional[Tuple[int, ...]] = None
    activation_type: str = 'residual'

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, kv_caches=None,
                 position_offset=0, return_activations=False):
        """
        Forward pass with optional activation extraction.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            kv_caches: Optional list of (k, v) tuples
            position_offset: Position offset for generation
            return_activations: Whether to return layer activations
            
        Returns:
            When return_activations=False:
                logits, new_kv_caches
            When return_activations=True:
                logits, new_kv_caches, activations
        """
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
            if kv_caches is not None:
                attention_mask = None
            else:
                attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                attention_mask = jnp.where(attention_mask == 0, -1e9, 0.0)
                attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, 0), 0)

        # Apply decoder layers
        new_kv_caches = []
        activations = {} if return_activations else None

        for i in range(self.config.num_hidden_layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None

            # Determine if we should extract activations from this layer
            extract_this_layer = return_activations and (
                self.layers_to_extract is None or i in self.layers_to_extract
            )

            hidden_states, new_kv_cache, layer_activations = QwenDecoderLayerWithHooks(
                self.config,
                name=f'layers_{i}'
            )(hidden_states, attention_mask, layer_kv_cache, position_offset, extract_this_layer)

            new_kv_caches.append(new_kv_cache)

            # Store layer activation if extracted
            if extract_this_layer and layer_activations is not None:
                if self.activation_type == 'mlp':
                    activations[f'layer_{i}'] = layer_activations['mlp']
                elif self.activation_type == 'attn':
                    activations[f'layer_{i}'] = layer_activations['attn']
                elif self.activation_type == 'residual':
                    activations[f'layer_{i}'] = layer_activations['residual']
                else:
                    activations[f'layer_{i}'] = layer_activations.get('residual', layer_activations['layer_output'])

        # Final norm
        hidden_states = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            self.config.dtype,
            name='norm'
        )(hidden_states)

        # Store final normalized activation if requested
        if return_activations:
            activations['final_norm'] = hidden_states

        # LM head
        if self.config.tie_word_embeddings:
            lm_logits = embed_tokens.attend(hidden_states)
        else:
            lm_head = nn.Dense(
                self.config.vocab_size,
                use_bias=False,
                dtype=self.config.dtype,
                name='lm_head'
            )
            lm_logits = lm_head(hidden_states)

        if return_activations:
            return lm_logits, new_kv_caches, activations
        else:
            return lm_logits, new_kv_caches


def create_model_with_hooks(
    config: QwenConfig, 
    layers_to_extract: Optional[List[int]] = None,
    activation_type: str = 'residual'
) -> QwenModelWithActivations:
    """
    Create a Qwen model with activation extraction hooks.
    
    Args:
        config: Model configuration
        layers_to_extract: List of layer indices to extract (None = all layers)
        activation_type: Type of activation to extract:
            - 'mlp': MLP output before residual connection
            - 'attn': Attention output before residual connection
            - 'residual': Final layer output after both residuals (default)
            
    Returns:
        QwenModelWithActivations instance
        
    Note:
        layers_to_extract is converted to tuple for JIT compatibility
    """
    if activation_type not in ['mlp', 'attn', 'residual']:
        raise ValueError(f"activation_type must be 'mlp', 'attn', or 'residual', got: {activation_type}")
    
    layers_tuple = tuple(layers_to_extract) if layers_to_extract is not None else None
    return QwenModelWithActivations(
        config=config, 
        layers_to_extract=layers_tuple, 
        activation_type=activation_type
    )


def generate_with_kv_cache_and_activations(
    model,
    params,
    input_ids,
    max_tokens: int = 50,
    extract_activations: bool = False,
    extract_every_n_tokens: int = 1,
    tokenizer=None
) -> Tuple[jnp.ndarray, Optional[List[Dict]], Dict]:
    """
    Efficient generation with KV caching and optional activation extraction.
    
    Implements the prefill/decode pattern for fast autoregressive generation
    while supporting activation extraction at configurable intervals.
    
    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        max_tokens: Maximum tokens to generate
        extract_activations: Whether to extract activations
        extract_every_n_tokens: Extract activations every N tokens
        tokenizer: Optional tokenizer for EOS detection
        
    Returns:
        generated_ids: Generated token IDs [batch, seq_len + max_tokens]
        activations_per_step: List of activation dicts (if extract_activations=True)
        timing_info: Dict with prefill/decode timing
    """
    # JIT compile functions for performance
    @jax.jit
    def prefill(params, input_ids):
        return model.apply(
            params, input_ids,
            kv_caches=None,
            position_offset=0,
            return_activations=False
        )

    @jax.jit
    def decode_step(params, input_id, kv_caches, position):
        return model.apply(
            params, input_id,
            kv_caches=kv_caches,
            position_offset=position,
            return_activations=False
        )

    def decode_step_with_activations(params, input_id, kv_caches, position):
        return model.apply(
            params, input_id,
            kv_caches=kv_caches,
            position_offset=position,
            return_activations=True
        )

    timing_info = {}
    start_time = time.time()

    # Prefill phase
    logits, kv_caches = prefill(params, input_ids)
    next_token_logits = logits[:, -1, :]
    next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
    generated_ids = jnp.concatenate([input_ids, next_token_id], axis=1)

    prefill_time = time.time() - start_time
    timing_info['prefill_time'] = prefill_time

    # Decode phase
    activations_per_step = [] if extract_activations else None
    decode_start = time.time()
    tokens_generated = 1

    for i in range(1, max_tokens):
        should_extract = extract_activations and (i % extract_every_n_tokens == 0)

        if should_extract:
            logits, kv_caches, activations = decode_step_with_activations(
                params, next_token_id, kv_caches, input_ids.shape[1] + i - 1
            )
            activations_per_step.append({
                'step': i,
                'position': input_ids.shape[1] + i - 1,
                'activations': activations
            })
        else:
            logits, kv_caches = decode_step(
                params, next_token_id, kv_caches, input_ids.shape[1] + i - 1
            )

        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        tokens_generated += 1

        if tokenizer and tokenizer.eos_token_id and next_token_id[0, 0] == tokenizer.eos_token_id:
            break

    decode_time = time.time() - decode_start
    timing_info['decode_time'] = decode_time
    timing_info['total_time'] = prefill_time + decode_time
    timing_info['tokens_generated'] = tokens_generated
    timing_info['tokens_per_sec'] = tokens_generated / decode_time if decode_time > 0 else 0

    return generated_ids, activations_per_step, timing_info


def extract_activations_from_prompt(
    model,
    params,
    input_ids,
    layers_to_extract: Optional[List[int]] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Extract activations from a prompt (single forward pass, no generation).
    
    Useful for extracting activations from existing text.
    
    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        layers_to_extract: Which layers to extract (None = all)
        
    Returns:
        logits: Model logits [batch, seq_len, vocab_size]
        activations: Dict mapping 'layer_{i}' -> hidden states
    """
    logits, _, activations = model.apply(
        params, input_ids,
        kv_caches=None,
        position_offset=0,
        return_activations=True
    )
    return logits, activations


def extract_specific_token_activations(
    activations: Dict[str, jnp.ndarray],
    layer_indices: Optional[List[int]] = None,
    token_positions: Optional[List[int]] = None
) -> Dict[str, jnp.ndarray]:
    """
    Extract activations from specific layers and token positions.
    
    Args:
        activations: Dict from 'layer_{i}' -> hidden_states [batch, seq_len, hidden_dim]
        layer_indices: Which layers to extract (None = all)
        token_positions: Which token positions to extract (None = all, [-1] = last token)
        
    Returns:
        Filtered activations dict
    """
    filtered = {}

    for layer_name, acts in activations.items():
        # Filter by layer
        if layer_name.startswith('layer_'):
            try:
                layer_idx = int(layer_name.split('_')[1])
                if layer_indices is not None and layer_idx not in layer_indices:
                    continue
            except (ValueError, IndexError):
                if layer_indices is not None:
                    continue

        # Filter by token position
        if token_positions is not None:
            if token_positions == [-1]:
                filtered[layer_name] = acts[:, -1:, :]
            else:
                filtered[layer_name] = acts[:, token_positions, :]
        else:
            filtered[layer_name] = acts

    return filtered


# Default layers for SAE training (later layers typically more interesting)
DEFAULT_SAE_LAYERS = list(range(10, 24))
