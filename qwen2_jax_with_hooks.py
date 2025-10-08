"""
Qwen 2.5 Model with Layer Activation Extraction Hooks
Extended from qwen2_jax.py to support intermediate layer activation capture
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import struct
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from dataclasses import dataclass

from qwen2_jax import (
    QwenConfig, RMSNorm, QwenMLP, QwenAttention,
    QwenDecoderLayer, convert_hf_to_jax_weights
)


class QwenModelWithActivations(nn.Module):
    """
    Qwen model that returns intermediate layer activations

    Returns:
        - logits: Final output logits
        - activations: Dict mapping layer_idx -> hidden_states
    """
    config: QwenConfig
    layers_to_extract: List[int] = None  # Which layers to extract (None = all)

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, kv_caches=None, position_offset=0, return_activations=True):
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
            # For generation with KV cache, we only need to mask new tokens
            if kv_caches is not None and position_offset > 0:
                # Only mask for the current position against all previous positions
                total_len = position_offset + seq_len
                attention_mask = jnp.zeros((1, 1, seq_len, total_len))
            else:
                attention_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
                attention_mask = jnp.where(attention_mask == 0, -1e9, 0.0)
                attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, 0), 0)

        # Store activations
        activations = {} if return_activations else None

        # Apply decoder layers with KV caching
        new_kv_caches = []
        for i in range(self.config.num_hidden_layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv_cache = QwenDecoderLayer(self.config, name=f'layers_{i}')(
                hidden_states, attention_mask, layer_kv_cache, position_offset
            )
            new_kv_caches.append(new_kv_cache)

            # Extract activation if requested
            if return_activations:
                if self.layers_to_extract is None or i in self.layers_to_extract:
                    # Store a copy of hidden states
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
            return lm_logits, activations
        else:
            return lm_logits, new_kv_caches


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


def generate_with_layer_activations(
    model,
    params,
    input_ids,
    max_tokens: int = 50,
    layers_to_extract: Optional[List[int]] = None,
    return_all_step_activations: bool = False
):
    """
    Generate tokens and extract layer activations at each generation step

    Args:
        model: QwenModelWithActivations instance
        params: Model parameters
        input_ids: Input token IDs [batch, seq_len]
        max_tokens: Maximum tokens to generate
        layers_to_extract: Which layers to extract (None = all)
        return_all_step_activations: If True, return activations for every generation step

    Returns:
        generated_ids: Generated token IDs
        activations_per_step: List of dicts, one per generation step
    """
    generated_ids = input_ids
    activations_per_step = [] if return_all_step_activations else None

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

        # Sample next token (greedy for now)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)

        # Append to sequence
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)

        # Check for stopping condition (you can add EOS token check here)
        if generated_ids.shape[1] > input_ids.shape[1] + max_tokens:
            break

    # For final result, do one more forward pass to get final activations
    _, final_activations = model.apply(params, generated_ids, return_activations=True)

    return generated_ids, final_activations, activations_per_step


def extract_specific_layer_positions(
    activations: Dict[str, jnp.ndarray],
    layer_name: str,
    token_positions: Optional[List[int]] = None
) -> jnp.ndarray:
    """
    Extract activations from specific token positions in a layer

    Args:
        activations: Dict from layer_name -> hidden_states [batch, seq_len, hidden_dim]
        layer_name: e.g., 'layer_12'
        token_positions: List of token positions to extract (None = last token only)

    Returns:
        Extracted activations [batch, n_positions, hidden_dim]
    """
    if layer_name not in activations:
        raise ValueError(f"Layer {layer_name} not in activations. Available: {list(activations.keys())}")

    layer_acts = activations[layer_name]  # [batch, seq_len, hidden_dim]

    if token_positions is None:
        # Extract last token only
        return layer_acts[:, -1:, :]
    else:
        # Extract specific positions
        return layer_acts[:, token_positions, :]


# Example usage configuration
DEFAULT_SAE_LAYERS = list(range(10, 24))  # Layers 10-23 for SAE training


if __name__ == "__main__":
    # Example: Create model with hooks
    config = QwenConfig()
    model = create_model_with_hooks(config, layers_to_extract=[6, 12, 18, 23])

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(key, dummy_input)

    # Test forward pass
    logits, activations = model.apply(variables, dummy_input, return_activations=True)

    print("Logits shape:", logits.shape)
    print("Extracted layers:", list(activations.keys()))
    for layer_name, acts in activations.items():
        print(f"{layer_name}: {acts.shape}")
