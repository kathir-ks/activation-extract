"""
Qwen Model Package

This module provides a clean, modular implementation of the Qwen model for JAX/TPU.

Components:
- config: Model configuration dataclass
- qwen: Core model implementation (RMSNorm, MLP, Attention, Decoder, Model)
- hooks: Activation extraction with layer hooks
- kv_cache: KV cache utilities for efficient generation
- sharding: Multi-host TPU sharding utilities

Example usage:
    from model import QwenConfig, QwenModel, create_model_with_hooks
    from model.sharding import initialize_multihost, create_device_mesh
"""

# Configuration
from .config import QwenConfig, get_default_config, get_qwen_0_5b_config, get_qwen_7b_config, config_from_hf

# Core model components
from .qwen import (
    RMSNorm,
    QwenMLP,
    QwenAttention,
    QwenAttentionFixed,
    QwenDecoderLayer,
    QwenModel,
    rotate_half,
    apply_rotary_pos_emb,
    convert_hf_to_jax_weights,
)

# Activation hooks
from .hooks import (
    QwenDecoderLayerWithHooks,
    QwenModelWithActivations,
    create_model_with_hooks,
    generate_with_kv_cache_and_activations,
    extract_activations_from_prompt,
    extract_specific_token_activations,
    DEFAULT_SAE_LAYERS,
)

# KV Cache utilities
from .kv_cache import (
    KVCacheConfig,
    create_kv_cache_buffers,
    write_prefill_cache,
    update_kv_cache_ar,
    get_attention_kv,
    create_activation_buffer,
    update_activation_buffer,
    get_cache_info,
    validate_cache_shapes,
)

# Sharding utilities
from .sharding import (
    initialize_multihost,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded,
    pad_sequences,
    get_device_memory_info,
    P,
)

__all__ = [
    # Config
    'QwenConfig',
    'get_default_config',
    'get_qwen_0_5b_config', 
    'get_qwen_7b_config',
    'config_from_hf',
    
    # Core model
    'RMSNorm',
    'QwenMLP',
    'QwenAttention',
    'QwenAttentionFixed',
    'QwenDecoderLayer',
    'QwenModel',
    'rotate_half',
    'apply_rotary_pos_emb',
    'convert_hf_to_jax_weights',
    
    # Hooks
    'QwenDecoderLayerWithHooks',
    'QwenModelWithActivations',
    'create_model_with_hooks',
    'generate_with_kv_cache_and_activations',
    'extract_activations_from_prompt',
    'extract_specific_token_activations',
    'DEFAULT_SAE_LAYERS',
    
    # KV Cache
    'KVCacheConfig',
    'create_kv_cache_buffers',
    'write_prefill_cache',
    'update_kv_cache_ar',
    'get_attention_kv',
    'create_activation_buffer',
    'update_activation_buffer',
    'get_cache_info',
    'validate_cache_shapes',
    
    # Sharding
    'initialize_multihost',
    'create_device_mesh',
    'create_sharding_strategy',
    'shard_params',
    'extract_activations_sharded',
    'pad_sequences',
    'get_device_memory_info',
    'P',
]
