"""
Qwen Model Configuration

Provides configuration dataclasses for different Qwen model sizes.
"""

import jax.numpy as jnp
from flax import struct
from typing import Any


@struct.dataclass
class QwenConfig:
    """
    Configuration for Qwen model.
    
    Attributes:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        intermediate_size: MLP intermediate dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        max_position_embeddings: Maximum sequence length
        rope_theta: RoPE base frequency
        rms_norm_eps: RMSNorm epsilon
        tie_word_embeddings: Whether to tie input/output embeddings
        use_sliding_window: Whether to use sliding window attention
        sliding_window: Sliding window size
        use_fixed_cache: Use MaxText-style fixed-size cache
        dtype: Data type for computations
    """
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 32768
    use_fixed_cache: bool = False
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


def get_default_config() -> QwenConfig:
    """Get default Qwen 2.5 0.5B configuration."""
    return QwenConfig()


def get_qwen_0_5b_config() -> QwenConfig:
    """Get Qwen 2.5 0.5B configuration."""
    return QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


def get_qwen_7b_config() -> QwenConfig:
    """Get Qwen 2.5 7B configuration."""
    return QwenConfig(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def config_from_hf(hf_config) -> QwenConfig:
    """
    Create QwenConfig from a HuggingFace config object.
    
    Args:
        hf_config: HuggingFace model configuration
        
    Returns:
        QwenConfig instance
    """
    return QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=getattr(hf_config, 'rope_theta', 1000000.0),
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings', False),
    )
