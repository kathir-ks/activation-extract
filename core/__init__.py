"""
Core utilities for activation extraction

This package contains shared utilities used across extraction scripts.
"""

from .jax_utils import (
    initialize_multihost,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded,
    pad_sequences,
    get_device_memory_info,
    P  # PartitionSpec alias
)

from .dataset_utils import (
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    create_prompts_from_dataset
)

from .activation_storage import ActivationStorage, load_activation_shard

__all__ = [
    # JAX utilities
    'initialize_multihost',
    'create_device_mesh',
    'create_sharding_strategy',
    'shard_params',
    'extract_activations_sharded',
    'pad_sequences',
    'get_device_memory_info',
    'P',

    # Dataset utilities
    'load_arc_dataset_jsonl',
    'load_arc_dataset_from_shard',
    'create_prompts_from_dataset',

    # Storage
    'ActivationStorage',
    'load_activation_shard',
]
