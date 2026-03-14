"""Multi-host TPU utilities for SAE training."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from typing import Dict, Optional


def create_sae_mesh(mesh_type: str = "auto", verbose: bool = False) -> Mesh:
    """Create device mesh for SAE training.

    SAE parameters are small enough to replicate across devices,
    so we primarily use data parallelism (shard the batch).

    Args:
        mesh_type: "auto", "1d", or "data_parallel".
        verbose: Print mesh info.

    Returns:
        JAX Mesh with 'data' axis for batch sharding.
    """
    num_hosts = jax.process_count()
    num_local = jax.local_device_count()
    total = jax.device_count()

    if mesh_type == "auto":
        mesh_type = "data_parallel"

    if mesh_type == "data_parallel":
        # Pure data parallelism: one axis for batch sharding
        if num_hosts > 1:
            device_array = mesh_utils.create_device_mesh(
                (total,), devices=None, allow_split_physical_axes=True
            )
        else:
            device_array = np.array(jax.devices())
        mesh = Mesh(device_array, axis_names=("data",))

    elif mesh_type == "1d":
        mesh = Mesh(np.array(jax.devices()), axis_names=("data",))

    else:
        raise ValueError(f"Unknown mesh_type for SAE: {mesh_type}")

    if verbose:
        print(f"\nSAE Device Mesh:")
        print(f"  Type: {mesh_type}")
        print(f"  Shape: {mesh.shape}")
        print(f"  Hosts: {num_hosts}, Local devices: {num_local}, Total: {total}")

    return mesh


def shard_batch(
    batch: jnp.ndarray,
    mesh: Mesh,
    num_hosts: int = 1,
) -> jnp.ndarray:
    """Shard a batch of activations across the data axis.

    In single-host mode, uses jax.device_put directly.
    In multi-host mode, each host provides its LOCAL portion and we use
    jax.make_array_from_single_device_arrays to create the global array.

    Args:
        batch: Local batch [per_host_batch_size, hidden_dim].
        mesh: Device mesh with 'data' axis.
        num_hosts: Number of hosts (1 = single-host).
    """
    sharding = NamedSharding(mesh, P("data", None))

    if num_hosts <= 1:
        return jax.device_put(batch, sharding)

    # Multi-host: split local batch across local devices
    local_devices = jax.local_devices()
    num_local = len(local_devices)
    per_device = batch.shape[0] // num_local
    global_batch_size = batch.shape[0] * num_hosts

    local_arrays = []
    for i, device in enumerate(local_devices):
        slab = batch[i * per_device : (i + 1) * per_device]
        local_arrays.append(jax.device_put(jnp.array(slab), device))

    return jax.make_array_from_single_device_arrays(
        shape=(global_batch_size, batch.shape[1]),
        sharding=sharding,
        arrays=local_arrays,
    )


def replicate_params(params, mesh: Mesh):
    """Replicate SAE parameters across all devices.

    SAE params are small (e.g., 50MB for 896*14336 in bfloat16),
    so replication is efficient.
    """
    replicated = NamedSharding(mesh, P())

    def _replicate(x):
        if hasattr(x, "shape"):
            return jax.device_put(x, replicated)
        return x

    return jax.tree.map(_replicate, params)


def get_host_info() -> Dict:
    """Get current host info for distributed setup."""
    return {
        "host_id": jax.process_index(),
        "num_hosts": jax.process_count(),
        "local_devices": jax.local_device_count(),
        "total_devices": jax.device_count(),
        "is_primary": jax.process_index() == 0,
    }


def initialize_distributed(verbose: bool = False) -> Dict:
    """Auto-initialize JAX distributed from environment.

    For TPU pods, JAX auto-detects the topology from environment
    variables set by the TPU runtime. This function just ensures
    initialization happens and returns host info.
    """
    import os

    # Check if we need explicit initialization
    coordinator = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if coordinator:
        hosts = coordinator.split(",")
        num_hosts = len(hosts)
        host_id = int(os.environ.get("CLOUD_TPU_TASK_ID", "0"))
        coord_addr = f"{hosts[0].strip()}:8476"

        if verbose:
            print(f"Initializing distributed: host {host_id}/{num_hosts}, coord={coord_addr}")

        jax.distributed.initialize(
            coordinator_address=coord_addr,
            num_processes=num_hosts,
            process_id=host_id,
        )
    # For single-host or already-initialized, just return info
    return get_host_info()
