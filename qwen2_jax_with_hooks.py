"""
Qwen 2.5 Model with Layer Activation Extraction Hooks
Fully compatible with KV caching and RoPE caching from qwen2_jax.py

Key Features:
- Supports KV cache for fast generation (8-10x speedup)
- RoPE cache pre-computed in model setup
- Optional activation extraction from intermediate layers
- JIT-compatible with proper static argument handling
- Can extract activations during both prefill and decode phases
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import time

from qwen2_jax import (
    QwenConfig, RMSNorm, QwenMLP, QwenAttention,
    rotate_half, apply_rotary_pos_emb, convert_hf_to_jax_weights,
    compute_rope_embeddings
)


class QwenDecoderLayerWithHooks(nn.Module):
    """Qwen decoder layer with optional activation capture"""
    config: QwenConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None, kv_cache=None,
                 position_offset=0, return_activations=False,
                 rope_cos=None, rope_sin=None):
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
        )(hidden_states_norm, attention_mask, kv_cache, position_offset,
          rope_cos=rope_cos, rope_sin=rope_sin)

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
                'attn': attn_activation,  # Attention output before residual
                'mlp': mlp_activation,    # MLP output before residual
                'residual': layer_output_activation,  # Final output after both residuals
                'post_attn': post_attn_activation,  # For backward compatibility
                'layer_output': layer_output_activation  # For backward compatibility
            }
            return hidden_states, new_kv_cache, activations
        else:
            return hidden_states, new_kv_cache, None


class QwenModelWithActivations(nn.Module):
    """
    Qwen model that supports both KV caching and activation extraction

    This model can operate in three modes:
    1. Fast inference (return_activations=False, with KV cache)
    2. Normal inference (return_activations=False, without KV cache)
    3. Activation extraction (return_activations=True, with or without KV cache)

    Returns:
        When return_activations=False:
            - logits: [batch, seq_len, vocab_size]
            - new_kv_caches: List of per-layer (k, v) tuples

        When return_activations=True:
            - logits: [batch, seq_len, vocab_size]
            - new_kv_caches: List of per-layer (k, v) tuples
            - activations: Dict mapping 'layer_{i}' -> layer hidden states
    """
    config: QwenConfig
    layers_to_extract: Optional[Tuple[int, ...]] = None  # Tuple for static arg compatibility
    activation_type: str = 'residual'  # 'mlp', 'attn', or 'residual'

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, kv_caches=None,
                 position_offset=0, return_activations=False):
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
                # Decode phase: no masking needed (attend to all cached tokens)
                attention_mask = None
            else:
                # Prefill phase: use causal mask
                attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                attention_mask = jnp.where(attention_mask == 0, -1e9, 0.0)
                attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, 0), 0)

        # Apply decoder layers
        new_kv_caches = []
        activations = {} if return_activations else None

        # Compute RoPE embeddings once, shared across all layers
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rope_cos, rope_sin = compute_rope_embeddings(
            head_dim, self.config.max_position_embeddings, self.config.rope_theta
        )

        for i in range(self.config.num_hidden_layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None

            # Determine if we should extract activations from this layer
            extract_this_layer = return_activations and (
                self.layers_to_extract is None or i in self.layers_to_extract
            )

            hidden_states, new_kv_cache, layer_activations = QwenDecoderLayerWithHooks(
                self.config,
                name=f'layers_{i}'
            )(hidden_states, attention_mask, layer_kv_cache, position_offset,
              extract_this_layer, rope_cos=rope_cos, rope_sin=rope_sin)

            new_kv_caches.append(new_kv_cache)

            # Store layer activation if extracted
            if extract_this_layer and layer_activations is not None:
                # Select activation based on activation_type configuration
                if self.activation_type == 'mlp':
                    # MLP output before residual
                    activations[f'layer_{i}'] = layer_activations['mlp']
                elif self.activation_type == 'attn':
                    # Attention output before residual
                    activations[f'layer_{i}'] = layer_activations['attn']
                elif self.activation_type == 'residual':
                    # Final output after both residuals (default)
                    activations[f'layer_{i}'] = layer_activations['residual']
                else:
                    # Fallback to residual for backward compatibility
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


def create_model_with_hooks(config: QwenConfig, layers_to_extract: Optional[List[int]] = None,
                            activation_type: str = 'residual'):
    """
    Create a Qwen model with activation extraction hooks

    Args:
        config: Model configuration
        layers_to_extract: List of layer indices to extract (None = all layers)
        activation_type: Type of activation to extract. One of:
            - 'mlp': MLP output before residual connection
            - 'attn': Attention output before residual connection
            - 'residual': Final layer output after both residual connections (default)

    Returns:
        QwenModelWithActivations instance

    Note: layers_to_extract is converted to tuple for JIT compatibility
    """
    # Validate activation_type
    if activation_type not in ['mlp', 'attn', 'residual']:
        raise ValueError(f"activation_type must be 'mlp', 'attn', or 'residual', got: {activation_type}")
    
    # Convert list to tuple for JIT static argument compatibility
    layers_tuple = tuple(layers_to_extract) if layers_to_extract is not None else None
    return QwenModelWithActivations(config=config, layers_to_extract=layers_tuple, activation_type=activation_type)


def generate_with_kv_cache_and_activations(
    model,
    params,
    input_ids,
    max_tokens: int = 50,
    extract_activations: bool = False,
    extract_every_n_tokens: int = 1,
    tokenizer=None
):
    """
    Efficient generation with KV caching and optional activation extraction

    This function implements the prefill/decode pattern from qwen2_jax.py
    while adding support for activation extraction.

    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        max_tokens: Maximum tokens to generate
        extract_activations: Whether to extract activations
        extract_every_n_tokens: Extract activations every N tokens (for efficiency)
        tokenizer: Optional tokenizer for EOS detection

    Returns:
        generated_ids: Generated token IDs [batch, seq_len + max_tokens]
        activations_per_step: List of activation dicts (if extract_activations=True)
        timing_info: Dict with prefill/decode timing
    """

    # JIT compile functions for performance
    @jax.jit
    def prefill(params, input_ids):
        """Prefill: Process entire prompt once"""
        return model.apply(
            params, input_ids,
            kv_caches=None,
            position_offset=0,
            return_activations=False
        )

    @jax.jit
    def decode_step(params, input_id, kv_caches, position):
        """Decode: Process one token with KV cache"""
        return model.apply(
            params, input_id,
            kv_caches=kv_caches,
            position_offset=position,
            return_activations=False
        )

    # For activation extraction, we use a non-JIT version
    def decode_step_with_activations(params, input_id, kv_caches, position):
        """Decode with activation extraction (slower but captures activations)"""
        return model.apply(
            params, input_id,
            kv_caches=kv_caches,
            position_offset=position,
            return_activations=True
        )

    timing_info = {}
    start_time = time.time()

    # Prefill phase: process the entire prompt
    logits, kv_caches = prefill(params, input_ids)
    next_token_logits = logits[:, -1, :]
    next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
    generated_ids = jnp.concatenate([input_ids, next_token_id], axis=1)

    prefill_time = time.time() - start_time
    timing_info['prefill_time'] = prefill_time

    # Decode phase: generate tokens one at a time
    activations_per_step = [] if extract_activations else None
    decode_start = time.time()
    tokens_generated = 1

    for i in range(1, max_tokens):
        # Decide whether to extract activations for this step
        should_extract = extract_activations and (i % extract_every_n_tokens == 0)

        if should_extract:
            # Use slower path with activation extraction
            logits, kv_caches, activations = decode_step_with_activations(
                params, next_token_id, kv_caches, input_ids.shape[1] + i - 1
            )
            activations_per_step.append({
                'step': i,
                'position': input_ids.shape[1] + i - 1,
                'activations': activations
            })
        else:
            # Use fast path without activation extraction
            logits, kv_caches = decode_step(
                params, next_token_id, kv_caches, input_ids.shape[1] + i - 1
            )

        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        tokens_generated += 1

        # Check for EOS token
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
):
    """
    Extract activations from a prompt (single forward pass, no generation)

    This is useful for extracting activations from existing text without generation.

    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        layers_to_extract: Which layers to extract (None = all)

    Returns:
        logits: Model logits [batch, seq_len, vocab_size]
        activations: Dict mapping 'layer_{i}' -> hidden states [batch, seq_len, hidden_dim]
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
    Extract activations from specific layers and token positions

    Args:
        activations: Dict from 'layer_{i}' -> hidden_states [batch, seq_len, hidden_dim]
        layer_indices: Which layers to extract (None = all)
        token_positions: Which token positions to extract (None = all, -1 = last token)

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
                # Skip non-numeric layer names like 'final_norm'
                if layer_indices is not None:
                    continue

        # Filter by token position
        if token_positions is not None:
            if token_positions == [-1]:
                # Last token only
                filtered[layer_name] = acts[:, -1:, :]
            else:
                # Specific positions
                filtered[layer_name] = acts[:, token_positions, :]
        else:
            # All positions
            filtered[layer_name] = acts

    return filtered


# Example usage and defaults
DEFAULT_SAE_LAYERS = list(range(10, 24))  # Layers 10-23 for SAE training


if __name__ == "__main__":
    # Example: Create model with hooks for specific layers
    print("Testing QwenModelWithActivations...")

    config = QwenConfig()

    # Test 1: Model with specific layers
    print("\n=== Test 1: Model with specific layer extraction ===")
    model = create_model_with_hooks(config, layers_to_extract=[6, 12, 18, 23])

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(key, dummy_input)

    # Test forward pass with activation extraction
    print("Testing forward pass with activations...")
    logits, kv_caches, activations = model.apply(
        variables, dummy_input, return_activations=True
    )

    print(f"Logits shape: {logits.shape}")
    print(f"Number of KV caches: {len(kv_caches)}")
    print(f"Extracted layers: {list(activations.keys())}")
    for layer_name, acts in activations.items():
        print(f"  {layer_name}: {acts.shape}")

    # Test 2: Fast inference without activations
    print("\n=== Test 2: Fast inference with KV cache (no activations) ===")

    # Prefill
    logits, kv_caches = model.apply(
        variables, dummy_input,
        kv_caches=None,
        position_offset=0,
        return_activations=False
    )
    print(f"Prefill - Logits: {logits.shape}, KV caches created: {len(kv_caches)}")

    # Decode step
    next_token = jnp.array([[42]], dtype=jnp.int32)
    logits, new_kv_caches = model.apply(
        variables, next_token,
        kv_caches=kv_caches,
        position_offset=10,
        return_activations=False
    )
    print(f"Decode - Logits: {logits.shape}, KV caches updated: {len(new_kv_caches)}")

    # Test 3: Generation with activation extraction
    print("\n=== Test 3: Generation with KV cache and activation extraction ===")
    generated_ids, activations_per_step, timing = generate_with_kv_cache_and_activations(
        model=model,
        params=variables,
        input_ids=dummy_input,
        max_tokens=10,
        extract_activations=True,
        extract_every_n_tokens=2  # Extract every 2 tokens for efficiency
    )

    print(f"Generated shape: {generated_ids.shape}")
    print(f"Activations extracted at {len(activations_per_step)} steps")
    print(f"Timing: prefill={timing['prefill_time']:.3f}s, decode={timing['decode_time']:.3f}s")
    print(f"Tokens/sec: {timing['tokens_per_sec']:.2f}")

    print("\n=== All tests passed! ===")
