"""
Pre-defined mesh configurations for TPU pod slices

This module provides:
- TPU topology configurations for common pod sizes
- Mesh creation utilities for multihost setups
- Recommended sharding specs for each topology

Supported TPU Types:
- v5e-8:   1 host  × 8 chips  = 8 devices
- v5e-64:  4 hosts × 8 chips  = 32 devices  
- v5e-128: 8 hosts × 8 chips  = 64 devices
- v5e-256: 16 hosts × 8 chips = 128 devices
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class TopologyConfig:
    """Configuration for a TPU pod topology"""
    name: str
    hosts: int
    chips_per_host: int
    mesh_shape: Tuple[int, int]
    axis_names: Tuple[str, str]
    recommended_batch_size: int
    recommended_seq_length: int
    
    @property
    def total_chips(self) -> int:
        return self.hosts * self.chips_per_host
    
    @property
    def data_parallel_size(self) -> int:
        """Number of data parallel replicas"""
        return self.mesh_shape[0]
    
    @property
    def model_parallel_size(self) -> int:
        """Number of model parallel shards"""
        return self.mesh_shape[1]


# Pre-defined TPU topologies
TPU_TOPOLOGIES: Dict[str, TopologyConfig] = {
    'v5e-8': TopologyConfig(
        name='v5e-8',
        hosts=1,
        chips_per_host=8,
        mesh_shape=(1, 8),
        axis_names=('data', 'model'),
        recommended_batch_size=8,
        recommended_seq_length=2048,
    ),
    'v5litepod-64': TopologyConfig(
        name='v5litepod-64',
        hosts=16,
        chips_per_host=4,
        mesh_shape=(16, 4),
        axis_names=('data', 'model'),
        recommended_batch_size=64,
        recommended_seq_length=2048,
    ),
    'v5e-64': TopologyConfig(
        name='v5e-64',
        hosts=4,
        chips_per_host=8,
        mesh_shape=(4, 8),
        axis_names=('data', 'model'),
        recommended_batch_size=32,
        recommended_seq_length=2048,
    ),
    'v5e-128': TopologyConfig(
        name='v5e-128',
        hosts=8,
        chips_per_host=8,
        mesh_shape=(8, 8),
        axis_names=('data', 'model'),
        recommended_batch_size=64,
        recommended_seq_length=2048,
    ),
    'v5e-256': TopologyConfig(
        name='v5e-256',
        hosts=16,
        chips_per_host=8,
        mesh_shape=(16, 8),
        axis_names=('data', 'model'),
        recommended_batch_size=128,
        recommended_seq_length=2048,
    ),
    # v4 TPU pods
    'v4-32': TopologyConfig(
        name='v4-32',
        hosts=4,
        chips_per_host=4,
        mesh_shape=(4, 4),
        axis_names=('data', 'model'),
        recommended_batch_size=16,
        recommended_seq_length=2048,
    ),
    'v4-64': TopologyConfig(
        name='v4-64',
        hosts=8,
        chips_per_host=4,
        mesh_shape=(8, 4),
        axis_names=('data', 'model'),
        recommended_batch_size=32,
        recommended_seq_length=2048,
    ),
}


def get_topology_config(topology: str) -> TopologyConfig:
    """
    Get configuration for a known TPU topology
    
    Args:
        topology: Topology name (e.g., 'v5e-64', 'v5e-128')
        
    Returns:
        TopologyConfig for the specified topology
        
    Raises:
        ValueError: If topology is not recognized
    """
    if topology not in TPU_TOPOLOGIES:
        available = ', '.join(TPU_TOPOLOGIES.keys())
        raise ValueError(f"Unknown topology '{topology}'. Available: {available}")
    return TPU_TOPOLOGIES[topology]


def detect_topology() -> Optional[str]:
    """
    Auto-detect TPU topology from current JAX devices
    
    Returns:
        Detected topology name, or None if not recognized
    """
    try:
        num_hosts = jax.process_count()
        num_local_devices = jax.local_device_count()
        total_devices = jax.device_count()
        
        # Match against known topologies
        for name, config in TPU_TOPOLOGIES.items():
            if (config.hosts == num_hosts and 
                config.chips_per_host == num_local_devices and
                config.total_chips == total_devices):
                return name
        
        return None
    except Exception:
        return None


def create_mesh_for_topology(
    topology: str,
    verbose: bool = False
) -> Mesh:
    """
    Create JAX device mesh for the specified TPU topology
    
    Args:
        topology: Topology name (e.g., 'v5e-64')
        verbose: Print mesh information
        
    Returns:
        JAX Mesh object configured for the topology
    """
    config = get_topology_config(topology)
    
    # Get all devices (includes devices from all hosts in multihost setup)
    devices = jax.devices()
    num_devices = len(devices)
    
    if num_devices != config.total_chips:
        raise RuntimeError(
            f"Topology {topology} expects {config.total_chips} devices, "
            f"but found {num_devices}. Make sure jax.distributed is initialized "
            f"for multihost setups."
        )
    
    # Create device mesh with specified shape
    device_array = mesh_utils.create_device_mesh(
        config.mesh_shape,
        devices=devices,
        allow_split_physical_axes=True
    )
    
    mesh = Mesh(device_array, axis_names=config.axis_names)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Device Mesh Created for {topology}")
        print(f"{'='*60}")
        print(f"  Hosts: {config.hosts}")
        print(f"  Chips per host: {config.chips_per_host}")
        print(f"  Total devices: {config.total_chips}")
        print(f"  Mesh shape: {config.mesh_shape}")
        print(f"  Axis names: {config.axis_names}")
        print(f"  Data parallel size: {config.data_parallel_size}")
        print(f"  Model parallel size: {config.model_parallel_size}")
        print(f"  Recommended batch size: {config.recommended_batch_size}")
        print(f"{'='*60}\n")
    
    return mesh


def create_sharding_specs(
    mesh: Mesh,
    config: TopologyConfig
) -> Dict[str, NamedSharding]:
    """
    Create sharding specifications for model parameters and data
    
    Args:
        mesh: JAX device mesh
        config: Topology configuration
        
    Returns:
        Dictionary of named shardings for different tensor types
    """
    P = PartitionSpec
    data_axis, model_axis = config.axis_names
    
    return {
        # Input data: sharded along data parallel axis
        'input': NamedSharding(mesh, P(data_axis, None)),
        
        # Embedding: vocabulary sharded along model axis
        'embed': NamedSharding(mesh, P(model_axis, None)),
        
        # Dense weights: sharded along model axis
        'weights': NamedSharding(mesh, P(None, model_axis)),
        
        # Layer norm: replicated
        'layernorm': NamedSharding(mesh, P(None)),
        
        # Bias: sharded along model axis
        'bias': NamedSharding(mesh, P(model_axis)),
        
        # Fully replicated
        'replicated': NamedSharding(mesh, P(None, None)),
        
        # Activations: sharded along data axis (batch dimension)
        'activations': NamedSharding(mesh, P(data_axis, None, None)),
    }


def get_per_host_batch_size(topology: str) -> int:
    """
    Get the recommended per-host batch size
    
    Args:
        topology: Topology name
        
    Returns:
        Batch size for each host
    """
    config = get_topology_config(topology)
    return config.recommended_batch_size // config.hosts


def validate_batch_size(batch_size: int, topology: str) -> bool:
    """
    Validate that batch size is compatible with topology
    
    Args:
        batch_size: Global batch size
        topology: Topology name
        
    Returns:
        True if batch size is valid
        
    Raises:
        ValueError: If batch size is incompatible
    """
    config = get_topology_config(topology)
    
    # Batch size must be divisible by data parallel size
    if batch_size % config.data_parallel_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by data parallel size "
            f"{config.data_parallel_size} for topology {topology}"
        )
    
    return True


def print_topology_info(topology: str):
    """Print detailed information about a topology"""
    config = get_topology_config(topology)
    
    print(f"\n{'='*60}")
    print(f"TPU Topology: {config.name}")
    print(f"{'='*60}")
    print(f"  Hosts:              {config.hosts}")
    print(f"  Chips per host:     {config.chips_per_host}")
    print(f"  Total chips:        {config.total_chips}")
    print(f"  Mesh shape:         {config.mesh_shape}")
    print(f"  Data parallel:      {config.data_parallel_size}x")
    print(f"  Model parallel:     {config.model_parallel_size}x")
    print(f"  Recommended batch:  {config.recommended_batch_size}")
    print(f"  Per-host batch:     {get_per_host_batch_size(topology)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Available TPU Topologies:")
    print("-" * 40)
    
    for name, config in TPU_TOPOLOGIES.items():
        print(f"  {name:12} -> {config.hosts} hosts × {config.chips_per_host} chips = {config.total_chips} devices")
    
    print("\nDetailed info for v5e-64:")
    print_topology_info('v5e-64')
