"""
JAX Utilities for Multi-Host TPU Processing

This module provides utilities for:
- Multi-host initialization
- Device mesh creation
- Model parameter sharding
- JIT-compiled activation extraction
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
from typing import Dict, Optional, Tuple

# Alias for convenience
P = PartitionSpec


def initialize_multihost(
    coordinator_address: str,
    num_hosts: int,
    host_id: int,
    verbose: bool = False
) -> int:
    """
    Initialize JAX distributed for multi-host TPU processing

    Args:
        coordinator_address: Coordinator IP:PORT (e.g., "10.0.0.1:8476")
        num_hosts: Total number of hosts
        host_id: This host's ID (0 to num_hosts-1)
        verbose: Print initialization info

    Returns:
        Number of devices across all hosts
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Initializing Multi-Host JAX")
        print(f"{'='*70}")
        print(f"  Host ID: {host_id}/{num_hosts-1}")
        print(f"  Coordinator: {coordinator_address}")

    # Initialize distributed JAX
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_hosts,
        process_id=host_id
    )

    devices = jax.devices()
    num_devices = len(devices)

    if verbose:
        print(f"  Local devices: {num_devices}")
        print(f"  Device types: {[d.device_kind for d in devices[:3]]}...")
        print(f"  Total devices (all hosts): {jax.device_count()}")
        print(f"{'='*70}\n")

    return jax.device_count()


def create_device_mesh(mesh_type: str = '2d', verbose: bool = False) -> Mesh:
    """
    Create device mesh for sharded computation

    Args:
        mesh_type: Type of mesh - '1d', '2d', or '3d'
        verbose: Print mesh info

    Returns:
        JAX Mesh object
    """
    devices = jax.devices()
    num_devices = len(devices)

    if mesh_type == '1d':
        # Simple 1D mesh along 'model' axis
        mesh = Mesh(devices, axis_names=('model',))

    elif mesh_type == '2d':
        # 2D mesh: (data, model)
        # For v5e-64: 4 hosts × 8 chips = 32 devices
        # Can reshape as (4, 8) or (8, 4) etc.

        # Try to create balanced 2D mesh
        if num_devices == 32:
            # v5e-64: 4 data parallel × 8 model parallel
            device_array = mesh_utils.create_device_mesh((4, 8), devices)
            mesh = Mesh(device_array, axis_names=('data', 'model'))
        else:
            # Generic: try to balance
            import math
            sqrt = int(math.sqrt(num_devices))
            if sqrt * sqrt == num_devices:
                device_array = mesh_utils.create_device_mesh((sqrt, sqrt), devices)
            else:
                # Fallback: (1, num_devices)
                device_array = mesh_utils.create_device_mesh((1, num_devices), devices)
            mesh = Mesh(device_array, axis_names=('data', 'model'))

    elif mesh_type == '3d':
        # 3D mesh: (data, fsdp, model)
        # For advanced sharding strategies
        if num_devices == 32:
            device_array = mesh_utils.create_device_mesh((2, 4, 4), devices)
            mesh = Mesh(device_array, axis_names=('data', 'fsdp', 'model'))
        else:
            raise ValueError(f"3D mesh not supported for {num_devices} devices")
    else:
        raise ValueError(f"Unknown mesh_type: {mesh_type}")

    if verbose:
        print(f"\nDevice Mesh Created:")
        print(f"  Type: {mesh_type.upper()}")
        print(f"  Shape: {mesh.shape}")
        print(f"  Axes: {mesh.axis_names}")
        print(f"  Devices: {num_devices}\n")

    return mesh


def create_sharding_strategy(mesh: Mesh) -> Dict[str, NamedSharding]:
    """
    Create sharding strategy for model parameters

    Args:
        mesh: Device mesh

    Returns:
        Dictionary of NamedSharding objects for different parameter types
    """
    if 'data' in mesh.axis_names and 'model' in mesh.axis_names:
        # 2D mesh: shard along model axis
        return {
            'weights': NamedSharding(mesh, P(None, 'model')),  # (hidden, model_dim) sharded
            'embed': NamedSharding(mesh, P('model', None)),    # (vocab, hidden) sharded
            'bias': NamedSharding(mesh, P('model')),           # (hidden,) sharded
            'layernorm': NamedSharding(mesh, P(None)),         # Replicated
            'replicated': NamedSharding(mesh, P(None, None))   # Fully replicated
        }
    elif 'model' in mesh.axis_names:
        # 1D mesh: shard along model axis
        return {
            'weights': NamedSharding(mesh, P(None, 'model')),
            'embed': NamedSharding(mesh, P('model', None)),
            'bias': NamedSharding(mesh, P('model')),
            'layernorm': NamedSharding(mesh, P(None)),
            'replicated': NamedSharding(mesh, P(None))
        }
    else:
        # 3D or other: default to replication
        return {
            'weights': NamedSharding(mesh, P(None, None)),
            'embed': NamedSharding(mesh, P(None, None)),
            'bias': NamedSharding(mesh, P(None)),
            'layernorm': NamedSharding(mesh, P(None)),
            'replicated': NamedSharding(mesh, P(None))
        }


def shard_params(params: Dict, sharding_strategy: Dict[str, NamedSharding]) -> Dict:
    """
    Apply sharding strategy to model parameters

    Args:
        params: Model parameters (nested dict)
        sharding_strategy: Sharding strategy from create_sharding_strategy()

    Returns:
        Sharded parameters
    """
    def shard_array(arr, name):
        """Determine sharding based on parameter name and shape"""
        if 'embed' in name or 'wte' in name:
            return jax.device_put(arr, sharding_strategy['embed'])
        elif 'ln' in name or 'norm' in name:
            return jax.device_put(arr, sharding_strategy['layernorm'])
        elif len(arr.shape) >= 2:
            return jax.device_put(arr, sharding_strategy['weights'])
        elif len(arr.shape) == 1:
            return jax.device_put(arr, sharding_strategy['bias'])
        else:
            return jax.device_put(arr, sharding_strategy['replicated'])

    # Recursively shard nested parameters
    def shard_tree(tree, prefix=''):
        if isinstance(tree, dict):
            return {k: shard_tree(v, f"{prefix}.{k}" if prefix else k)
                   for k, v in tree.items()}
        elif isinstance(tree, jnp.ndarray):
            return shard_array(tree, prefix)
        else:
            return tree

    return shard_tree(params)


@partial(jit, static_argnums=(0,))
def extract_activations_sharded(model, params, input_ids):
    """
    Extract activations with sharded model (JIT compiled)

    This is the core JIT-compiled function for activation extraction.
    It's decorated with @jit for XLA compilation and performance.

    Args:
        model: JAX model with hooks (static argument)
        params: Sharded model parameters
        input_ids: [batch, seq_len] - replicated across devices

    Returns:
        activations: Dict mapping layer names to activation tensors
    """
    # Single forward pass with sharded params - NO GENERATION!
    # The model automatically handles sharded computation across devices
    _, _, activations = model.apply(
        params,
        input_ids,
        return_activations=True
    )

    return activations


def pad_sequences(
    sequences: list,
    pad_token_id: int = 0,
    fixed_length: Optional[int] = None
) -> list:
    """
    Pad sequences to uniform length

    Args:
        sequences: List of sequences (lists of token IDs)
        pad_token_id: Token ID to use for padding
        fixed_length: If specified, pad to this exact length

    Returns:
        List of padded sequences
    """
    if not sequences:
        return []

    # Determine target length
    if fixed_length is not None:
        max_len = fixed_length
    else:
        max_len = max(len(seq) for seq in sequences)

    # Pad all sequences
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_token_id] * (max_len - len(seq)))
        elif len(seq) > max_len:
            # Truncate if too long
            padded.append(seq[:max_len])
        else:
            padded.append(seq)

    return padded


def get_device_memory_info() -> Dict[str, float]:
    """
    Get memory info for devices

    Returns:
        Dictionary with memory statistics in GB
    """
    try:
        devices = jax.devices()
        if not devices:
            return {}

        # Get memory from first device as representative
        device = devices[0]
        stats = device.memory_stats()

        return {
            'bytes_in_use': stats.get('bytes_in_use', 0) / 1e9,
            'bytes_limit': stats.get('bytes_limit', 0) / 1e9,
            'num_devices': len(devices)
        }
    except:
        return {}


# =============================================================================
# Multihost TPU Utilities
# =============================================================================

def initialize_multihost_auto(verbose: bool = False) -> Dict[str, any]:
    """
    Auto-initialize JAX distributed from environment variables
    
    This function detects multihost configuration from TPU pod environment
    variables set by Google Cloud TPU runtime.
    
    For TPU pods (v5e-64, etc.), JAX auto-detects the topology without
    explicit jax.distributed.initialize() call.
    
    Environment variables checked:
    - TPU_WORKER_HOSTNAMES: Comma-separated list of worker hostnames
    - MEGASCALE_COORDINATOR_ADDRESS: Alternative coordinator address
    - TPU_WORKER_ID / CLOUD_TPU_TASK_ID: This host's ID
    - TPU_WORKER_COUNT: Total number of hosts
    
    Returns:
        Dict with host_id, num_hosts, coordinator_address, total_devices
    """
    import os
    
    # Try to get coordinator address
    coordinator_address = None
    hostnames = os.environ.get('TPU_WORKER_HOSTNAMES', '')
    
    if hostnames:
        # Use first hostname as coordinator
        first_host = hostnames.split(',')[0].strip()
        coordinator_address = f"{first_host}:8476"
    else:
        # Try alternative env var
        coordinator_address = os.environ.get('MEGASCALE_COORDINATOR_ADDRESS')
    
    # Get host ID (default to 0 for single host or JAX auto-detection)
    host_id = int(os.environ.get('TPU_WORKER_ID', 
                   os.environ.get('CLOUD_TPU_TASK_ID', '0')))
    
    # Get number of hosts (default to 1, but JAX will auto-detect for pods)
    num_hosts = int(os.environ.get('TPU_WORKER_COUNT', '1'))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Multihost Auto-Detection")
        print(f"{'='*70}")
        print(f"  Host ID (from env): {host_id}")
        print(f"  Num hosts (from env): {num_hosts}")
        print(f"  Coordinator: {coordinator_address}")
    
    # For TPU pods, JAX auto-detects without explicit initialize()
    # Only call jax.distributed.initialize if we have explicit coordinator
    if coordinator_address and num_hosts > 1:
        if verbose:
            print(f"  Calling jax.distributed.initialize() with coordinator...")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_hosts,
            process_id=host_id
        )
    else:
        if verbose:
            print(f"  Letting JAX auto-detect TPU pod topology...")
    
    # After initialization (or auto-detection), get actual values from JAX
    total_devices = jax.device_count()
    local_devices = jax.local_device_count()
    actual_host_id = jax.process_index()
    actual_num_hosts = jax.process_count()
    
    if verbose:
        print(f"  JAX process_index(): {actual_host_id}")
        print(f"  JAX process_count(): {actual_num_hosts}")
        print(f"  Local devices: {local_devices}")
        print(f"  Total devices: {total_devices}")
        print(f"{'='*70}\n")
    
    return {
        'host_id': actual_host_id,
        'num_hosts': actual_num_hosts,
        'coordinator_address': coordinator_address,
        'total_devices': total_devices,
        'local_devices': local_devices,
        'is_primary': actual_host_id == 0,
    }


def get_host_info() -> Dict[str, any]:
    """
    Get information about the current host in a multihost setup
    
    Returns:
        Dict with:
        - host_id: This host's process ID
        - num_hosts: Total number of hosts/processes
        - local_device_count: Devices on this host
        - global_device_count: Total devices across all hosts
        - is_primary: Whether this is host 0
    """
    return {
        'host_id': jax.process_index(),
        'num_hosts': jax.process_count(),
        'local_device_count': jax.local_device_count(),
        'global_device_count': jax.device_count(),
        'is_primary': jax.process_index() == 0,
    }


def distribute_data_across_hosts(
    data: list,
    host_id: Optional[int] = None,
    num_hosts: Optional[int] = None
) -> list:
    """
    Distribute data across hosts in round-robin fashion
    
    Each host gets every Nth item where N = num_hosts.
    
    Args:
        data: List of items to distribute
        host_id: This host's ID (auto-detected if None)
        num_hosts: Total hosts (auto-detected if None)
        
    Returns:
        Subset of data for this host
    """
    if host_id is None:
        host_id = jax.process_index()
    if num_hosts is None:
        num_hosts = jax.process_count()
    
    # Round-robin distribution
    return data[host_id::num_hosts]


def gather_activations_to_primary(
    local_activations: Dict[str, jnp.ndarray],
    host_id: Optional[int] = None,
) -> Optional[Dict[str, jnp.ndarray]]:
    """
    Gather activations from all hosts to the primary host (host 0)
    
    Uses JAX multihost utilities for efficient cross-host communication.
    
    Args:
        local_activations: Dict of activations from this host
        host_id: This host's ID (auto-detected if None)
        
    Returns:
        On host 0: Dict of gathered activations (concatenated along batch dim)
        On other hosts: None
    """
    from jax.experimental import multihost_utils
    
    if host_id is None:
        host_id = jax.process_index()
    
    num_hosts = jax.process_count()
    
    if num_hosts == 1:
        # Single host, no gathering needed
        return local_activations
    
    # Gather each layer's activations
    gathered = {}
    for layer_name, activation in local_activations.items():
        # Use process_allgather to collect from all hosts
        # This returns the same gathered array on all hosts
        all_activations = multihost_utils.process_allgather(activation)
        gathered[layer_name] = all_activations
    
    # Only primary host returns the gathered data
    if host_id == 0:
        return gathered
    else:
        return None


def sync_hosts(tag: str = "sync", timeout_seconds: int = 60):
    """
    Synchronize all hosts at a barrier point
    
    All hosts must call this function before any can proceed.
    Useful for coordinating checkpointing and uploads.
    
    Args:
        tag: Unique identifier for this sync point
        timeout_seconds: Timeout for sync operation (used for logging only)
    """
    from jax.experimental import multihost_utils
    
    if jax.process_count() > 1:
        try:
            multihost_utils.sync_global_devices(tag)
        except Exception as e:
            # Log but don't crash - sync failures at end of job are often benign
            print(f"Warning: sync_hosts('{tag}') failed: {e}")
            print("  This is often expected during job finalization if hosts finish at different times.")


def is_primary_host() -> bool:
    """Check if this is the primary host (host 0)"""
    return jax.process_index() == 0


def get_per_host_batch_indices(
    total_samples: int,
    batch_size: int,
    host_id: Optional[int] = None,
    num_hosts: Optional[int] = None
) -> list:
    """
    Get batch indices for this host
    
    Divides samples evenly across hosts for balanced processing.
    
    Args:
        total_samples: Total number of samples
        batch_size: Batch size per host
        host_id: This host's ID
        num_hosts: Total number of hosts
        
    Returns:
        List of (start_idx, end_idx) tuples for batches this host should process
    """
    if host_id is None:
        host_id = jax.process_index()
    if num_hosts is None:
        num_hosts = jax.process_count()
    
    # Calculate samples per host
    samples_per_host = total_samples // num_hosts
    remainder = total_samples % num_hosts
    
    # Distribute remainder to first few hosts
    if host_id < remainder:
        start = host_id * (samples_per_host + 1)
        end = start + samples_per_host + 1
    else:
        start = host_id * samples_per_host + remainder
        end = start + samples_per_host
    
    # Generate batch indices for this host's portion
    batches = []
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        batches.append((batch_start, batch_end))
    
    return batches