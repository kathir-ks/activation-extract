"""
JAX Utilities for Multi-Host TPU Processing

This module provides utilities for:
- Multi-host TPU initialization
- Device mesh creation (1D, 2D, 3D)
- Model parameter sharding
- JIT-compiled activation extraction
- Sequence padding utilities
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
from typing import Dict, Optional, Tuple, List
import math

# Alias for convenience in external code
P = PartitionSpec


def initialize_multihost(
    coordinator_address: str,
    num_hosts: int,
    host_id: int,
    verbose: bool = False
) -> int:
    """
    Initialize JAX distributed for multi-host TPU processing.
    
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
    Create device mesh for sharded computation.
    
    Args:
        mesh_type: Type of mesh - '1d', '2d', or '3d'
        verbose: Print mesh info
        
    Returns:
        JAX Mesh object
    """
    devices = jax.devices()
    num_devices = len(devices)

    if mesh_type == '1d':
        mesh = Mesh(devices, axis_names=('model',))

    elif mesh_type == '2d':
        # 2D mesh: (data, model)
        if num_devices == 32:
            # v5e-64: 4 data parallel × 8 model parallel
            device_array = mesh_utils.create_device_mesh((4, 8), devices)
            mesh = Mesh(device_array, axis_names=('data', 'model'))
        else:
            sqrt = int(math.sqrt(num_devices))
            if sqrt * sqrt == num_devices:
                device_array = mesh_utils.create_device_mesh((sqrt, sqrt), devices)
            else:
                device_array = mesh_utils.create_device_mesh((1, num_devices), devices)
            mesh = Mesh(device_array, axis_names=('data', 'model'))

    elif mesh_type == '3d':
        # 3D mesh: (data, fsdp, model)
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
    Create sharding strategy for model parameters.
    
    Args:
        mesh: Device mesh
        
    Returns:
        Dictionary of NamedSharding objects for different parameter types
    """
    if 'data' in mesh.axis_names and 'model' in mesh.axis_names:
        # 2D mesh: shard along model axis
        return {
            'weights': NamedSharding(mesh, P(None, 'model')),
            'embed': NamedSharding(mesh, P('model', None)),
            'bias': NamedSharding(mesh, P('model')),
            'layernorm': NamedSharding(mesh, P(None)),
            'replicated': NamedSharding(mesh, P(None, None))
        }
    elif 'model' in mesh.axis_names:
        # 1D mesh
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
    Apply sharding strategy to model parameters.
    
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
    Extract activations with sharded model (JIT compiled).
    
    This is the core JIT-compiled function for activation extraction.
    Single forward pass - no generation.
    
    Args:
        model: JAX model with hooks (static argument)
        params: Sharded model parameters
        input_ids: [batch, seq_len] - replicated across devices
        
    Returns:
        activations: Dict mapping layer names to activation tensors
    """
    _, _, activations = model.apply(
        params,
        input_ids,
        return_activations=True
    )

    return activations


def pad_sequences(
    sequences: List,
    pad_token_id: int = 0,
    fixed_length: Optional[int] = None
) -> List:
    """
    Pad sequences to uniform length.
    
    Args:
        sequences: List of sequences (lists of token IDs)
        pad_token_id: Token ID to use for padding
        fixed_length: If specified, pad to this exact length
        
    Returns:
        List of padded sequences
    """
    if not sequences:
        return []

    if fixed_length is not None:
        max_len = fixed_length
    else:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_token_id] * (max_len - len(seq)))
        elif len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq)

    return padded


def get_device_memory_info() -> Dict[str, float]:
    """
    Get memory info for devices.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    try:
        devices = jax.devices()
        if not devices:
            return {}

        device = devices[0]
        stats = device.memory_stats()

        return {
            'bytes_in_use': stats.get('bytes_in_use', 0) / 1e9,
            'bytes_limit': stats.get('bytes_limit', 0) / 1e9,
            'num_devices': len(devices)
        }
    except:
        return {}
