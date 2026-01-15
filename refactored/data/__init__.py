"""
Data Module

Provides utilities for dataset loading, conversion, and sharding:
- Dataset conversion from HuggingFace to ARC format
- ARC dataset loading from JSONL and sharded formats
- Shard management for distributed processing
"""

from .converter import (
    convert_hf_dataset_to_arc_format,
    get_file_size_mb,
)

from .loader import (
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    create_prompts_from_dataset,
)

from .sharding import (
    ShardManager,
    load_shard_chunks,
    create_sharded_dataset,
)


__all__ = [
    # Converter
    'convert_hf_dataset_to_arc_format',
    'get_file_size_mb',
    
    # Loader
    'load_arc_dataset_jsonl',
    'load_arc_dataset_from_shard',
    'create_prompts_from_dataset',
    
    # Sharding
    'ShardManager',
    'load_shard_chunks',
    'create_sharded_dataset',
]
