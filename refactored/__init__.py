"""
Qwen Activation Extraction Framework

A refactored, modular codebase for:
- Qwen model inference on TPUs (single and multi-host)
- ARC-AGI task processing and prompting
- Activation extraction pipelines
- Dataset conversion and management

Modules:
- model: Qwen JAX implementation, hooks, KV cache, sharding
- arc: ARC-AGI encoders, prompting, data augmentation
- data: Dataset loading, conversion, and sharding
- extraction: Activation storage and extraction pipelines
"""

__version__ = "1.0.0"

# Re-export commonly used components for convenience
from .model import (
    # Configuration
    QwenConfig,
    get_default_config,
    get_qwen_0_5b_config,
    get_qwen_7b_config,
    config_from_hf,
    
    # Core model
    QwenModel,
    convert_hf_to_jax_weights,
    
    # Hooks
    QwenModelWithActivations,
    create_model_with_hooks,
    extract_activations_from_prompt,
    DEFAULT_SAE_LAYERS,
    
    # KV Cache
    KVCacheConfig,
    create_kv_cache_buffers,
    
    # Sharding
    initialize_multihost,
    create_device_mesh,
    shard_params,
)

from .arc import (
    # Encoders
    GridEncoder,
    create_grid_encoder,
    
    # Prompting
    create_prompts_from_task,
    parse_grid_from_response,
    get_prompt_templates,
    
    # Augmentation
    apply_data_augmentation,
    revert_data_augmentation,
    set_random_seed,
)

from .data import (
    # Loading
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    
    # Conversion
    convert_hf_dataset_to_arc_format,
    
    # Sharding
    ShardManager,
    create_sharded_dataset,
)

from .extraction import (
    # Storage
    ActivationStorage,
    load_activation_shard,
    
    # Extraction
    ExtractionConfig,
    run_extraction,
)


__all__ = [
    # Version
    '__version__',
    
    # Model
    'QwenConfig',
    'get_default_config',
    'get_qwen_0_5b_config', 
    'get_qwen_7b_config',
    'config_from_hf',
    'QwenModel',
    'convert_hf_to_jax_weights',
    'QwenModelWithActivations',
    'create_model_with_hooks',
    'extract_activations_from_prompt',
    'DEFAULT_SAE_LAYERS',
    'KVCacheConfig',
    'create_kv_cache_buffers',
    'initialize_multihost',
    'create_device_mesh',
    'shard_params',
    
    # ARC
    'GridEncoder',
    'create_grid_encoder',
    'create_prompts_from_task',
    'parse_grid_from_response',
    'get_prompt_templates',
    'apply_data_augmentation',
    'revert_data_augmentation',
    'set_random_seed',
    
    # Data
    'load_arc_dataset_jsonl',
    'load_arc_dataset_from_shard',
    'convert_hf_dataset_to_arc_format',
    'ShardManager',
    'create_sharded_dataset',
    
    # Extraction
    'ActivationStorage',
    'load_activation_shard',
    'ExtractionConfig',
    'run_extraction',
]
