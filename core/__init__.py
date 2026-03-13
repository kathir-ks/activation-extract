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
    P,  # PartitionSpec alias
    # Multihost utilities
    initialize_multihost_auto,
    get_host_info,
    distribute_data_across_hosts,
    gather_activations_to_primary,
    sync_hosts,
    is_primary_host,
    get_per_host_batch_indices,
)

from .dataset_utils import (
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    create_prompts_from_dataset
)

from .activation_storage import ActivationStorage, load_activation_shard

from .dynamic_batching import (
    create_dynamic_batches,
    pad_batch_to_bucket,
    DynamicBatch,
    DEFAULT_LENGTH_BUCKETS,
    DEFAULT_BATCH_SIZES,
)

from .grid_chunking import (
    create_grid_token_stream,
    chunk_token_stream,
    create_grid_chunks_from_dataset,
    ChunkMetadata,
    save_chunks_cache,
    load_chunks_cache,
    get_chunk_cache_path,
)

from .mesh_configs import (
    TopologyConfig,
    TPU_TOPOLOGIES,
    get_topology_config,
    detect_topology,
    create_mesh_for_topology,
    create_sharding_specs,
    get_per_host_batch_size,
    validate_batch_size,
)

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

    # Multihost utilities
    'initialize_multihost_auto',
    'get_host_info',
    'distribute_data_across_hosts',
    'gather_activations_to_primary',
    'sync_hosts',
    'is_primary_host',
    'get_per_host_batch_indices',

    # Mesh configuration
    'TopologyConfig',
    'TPU_TOPOLOGIES',
    'get_topology_config',
    'detect_topology',
    'create_mesh_for_topology',
    'create_sharding_specs',
    'get_per_host_batch_size',
    'validate_batch_size',

    # Dataset utilities
    'load_arc_dataset_jsonl',
    'load_arc_dataset_from_shard',
    'create_prompts_from_dataset',

    # Storage
    'ActivationStorage',
    'load_activation_shard',

    # Dynamic batching
    'create_dynamic_batches',
    'pad_batch_to_bucket',
    'DynamicBatch',
    'DEFAULT_LENGTH_BUCKETS',
    'DEFAULT_BATCH_SIZES',

    # Grid chunking
    'create_grid_token_stream',
    'chunk_token_stream',
    'create_grid_chunks_from_dataset',
    'ChunkMetadata',
    'save_chunks_cache',
    'load_chunks_cache',
    'get_chunk_cache_path',
]
