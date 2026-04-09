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
            # Build mesh ordered by (process, local_device) to ensure
            # each host's devices are contiguous (required for multi-host ops)
            all_devices = jax.devices()
            by_process = {}
            for d in all_devices:
                pid = d.process_index
                by_process.setdefault(pid, []).append(d)
            ordered = []
            for pid in sorted(by_process.keys()):
                ordered.extend(sorted(by_process[pid], key=lambda d: d.id))
            device_array = np.array(ordered)
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

    if batch.shape[0] % num_local != 0:
        raise ValueError(
            f"Per-host batch size {batch.shape[0]} must be divisible by "
            f"local device count {num_local}. Use a global batch_size "
            f"divisible by {num_local * num_hosts} (total devices)."
        )

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

    On multi-host pods, we place local copies on each local device
    and assemble the global array, avoiding cross-host collectives.
    """
    replicated = NamedSharding(mesh, P())
    local_devices = jax.local_devices()

    def _replicate(x):
        if hasattr(x, "shape"):
            x = jnp.asarray(x)
            # Place a full copy on each local device
            per_device = [jax.device_put(x, d) for d in local_devices]
            return jax.make_array_from_single_device_arrays(
                x.shape, replicated, per_device
            )
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

    On TPU pods (v5e/v6e pod slices), JAX auto-detects the topology
    from TPU runtime metadata. We only call jax.distributed.initialize()
    explicitly for non-pod multi-host setups (e.g., standalone VMs with
    a coordinator).

    On TPU pods, simply accessing jax.devices() triggers auto-init.
    """
    import os

    # On TPU pods, the runtime handles multi-host init automatically.
    # CLOUD_TPU_TASK_ID being set indicates we're on a TPU pod.
    on_tpu_pod = os.environ.get("CLOUD_TPU_TASK_ID") is not None

    coordinator = os.environ.get("TPU_WORKER_HOSTNAMES", "")

    if coordinator and not on_tpu_pod:
        # Non-pod multi-host: need explicit distributed init
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
    elif on_tpu_pod:
        # TPU pod: explicit distributed init (required for multi-host pods)
        if verbose:
            task_id = os.environ.get("CLOUD_TPU_TASK_ID", "?")
            print(f"TPU pod detected (task_id={task_id}), calling jax.distributed.initialize()...")
        jax.distributed.initialize()

    return get_host_info()
