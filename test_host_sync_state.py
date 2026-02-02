#!/usr/bin/env python3
"""
Test JAX Distributed Multihost Synchronization

This script tests how JAX distributed works on TPU pods by:
1. Auto-detecting the TPU pod topology
2. Testing device visibility across hosts
3. Testing multihost synchronization primitives
4. Testing cross-host data operations

Run on all workers simultaneously:
    gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=all \
        --command="cd ~/activation-extract && python3 test_host_sync_state.py"
"""

import os
import time
import socket
from datetime import datetime
from functools import partial

# Record start time before any imports
script_start_time = time.time()
script_start_datetime = datetime.now().isoformat()

print(f"[{script_start_datetime}] Script started on {socket.gethostname()}")

# Import JAX
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils, multihost_utils

def log(msg: str):
    """Log with timestamp and host info"""
    elapsed = time.time() - script_start_time
    host_id = jax.process_index()
    print(f"[{elapsed:7.2f}s] Host {host_id}: {msg}")


def test_basic_topology():
    """Test 1: Basic TPU topology detection"""
    print("\n" + "="*60)
    print("TEST 1: Basic Topology Detection")
    print("="*60)
    
    host_id = jax.process_index()
    num_hosts = jax.process_count()
    local_devices = jax.local_device_count()
    total_devices = jax.device_count()
    
    log(f"process_index() = {host_id}")
    log(f"process_count() = {num_hosts}")
    log(f"local_device_count() = {local_devices}")
    log(f"device_count() = {total_devices}")
    
    devices = jax.devices()
    local_devs = jax.local_devices()
    log(f"local_devices: {[str(d) for d in local_devs]}")
    
    return {
        'host_id': host_id,
        'num_hosts': num_hosts,
        'local_devices': local_devices,
        'total_devices': total_devices,
    }


def test_device_mesh():
    """Test 2: Device mesh creation"""
    print("\n" + "="*60)
    print("TEST 2: Device Mesh Creation")
    print("="*60)
    
    devices = jax.devices()
    num_devices = len(devices)
    
    log(f"Total devices visible: {num_devices}")
    
    # Try creating a 1D mesh
    try:
        mesh_1d = Mesh(devices, axis_names=('model',))
        log(f"1D Mesh created: shape={mesh_1d.shape}")
    except Exception as e:
        log(f"1D Mesh FAILED: {e}")
        return None
    
    # Try creating a 2D mesh using mesh_utils
    try:
        # For 64 devices: (8, 8) or (16, 4) mesh
        if num_devices == 64:
            device_array = mesh_utils.create_device_mesh(
                (8, 8), 
                devices=devices,
                allow_split_physical_axes=True
            )
        else:
            # Try to find factors
            for d1 in [8, 4, 2, 1]:
                if num_devices % d1 == 0:
                    d2 = num_devices // d1
                    device_array = mesh_utils.create_device_mesh(
                        (d1, d2), 
                        devices=devices,
                        allow_split_physical_axes=True
                    )
                    break
        
        mesh_2d = Mesh(device_array, axis_names=('data', 'model'))
        log(f"2D Mesh created: shape={mesh_2d.shape}")
        return mesh_2d
    except Exception as e:
        log(f"2D Mesh FAILED: {e}")
        return mesh_1d


def test_sync_global_devices():
    """Test 3: Global device synchronization"""
    print("\n" + "="*60)
    print("TEST 3: Global Device Synchronization")
    print("="*60)
    
    host_id = jax.process_index()
    
    # Test sync at multiple points
    for sync_point in ["barrier_1", "barrier_2", "barrier_3"]:
        start = time.time()
        log(f"Entering sync point: {sync_point}")
        
        try:
            multihost_utils.sync_global_devices(sync_point)
            elapsed = time.time() - start
            log(f"Sync {sync_point} completed in {elapsed:.3f}s")
        except Exception as e:
            log(f"Sync {sync_point} FAILED: {e}")
            return False
    
    log("All sync points passed!")
    return True


def test_broadcast_one_to_all():
    """Test 4: Broadcast data from one host to all"""
    print("\n" + "="*60)
    print("TEST 4: Broadcast One to All")
    print("="*60)
    
    host_id = jax.process_index()
    
    # Create host-specific data
    local_data = jnp.array([host_id * 100 + i for i in range(4)])
    log(f"Local data before broadcast: {local_data}")
    
    try:
        # Broadcast from host 0 to all
        broadcasted = multihost_utils.broadcast_one_to_all(local_data)
        log(f"Broadcasted data: {broadcasted}")
        
        # All hosts should have host 0's data
        expected = jnp.array([0, 1, 2, 3])  # Host 0's data
        if jnp.allclose(broadcasted, expected):
            log("Broadcast verification PASSED")
            return True
        else:
            log(f"Broadcast verification FAILED: expected {expected}")
            return False
    except Exception as e:
        log(f"Broadcast FAILED: {e}")
        return False


def test_sharded_jit():
    """Test 5: Sharded JIT computation"""
    print("\n" + "="*60)
    print("TEST 5: Sharded JIT Computation")
    print("="*60)
    
    host_id = jax.process_index()
    num_devices = jax.device_count()
    
    # Create a mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('devices',))
    
    # Create sharded input
    global_shape = (num_devices, 1024)
    local_shape = (jax.local_device_count(), 1024)
    
    log(f"Creating sharded array: global={global_shape}, local={local_shape}")
    
    try:
        # Create local data for this host's devices
        local_data = jnp.ones(local_shape) * (host_id + 1)
        
        # Create sharding
        sharding = NamedSharding(mesh, PartitionSpec('devices', None))
        
        # Create global array from local data
        global_array = jax.make_array_from_single_device_arrays(
            global_shape,
            sharding,
            [jax.device_put(local_data[i:i+1], d) for i, d in enumerate(jax.local_devices())]
        )
        
        log(f"Global array shape: {global_array.shape}")
        log(f"Global array sharding: {global_array.sharding}")
        
        # Define a sharded computation
        @jax.jit
        def sharded_sum(x):
            return jnp.sum(x, axis=1)
        
        # Compile and run
        result = sharded_sum(global_array)
        log(f"Sharded sum result shape: {result.shape}")
        log(f"First few results: {result[:4]}")
        
        return True
    except Exception as e:
        log(f"Sharded JIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_psum():
    """Test 6: All-reduce with psum using shard_map"""
    print("\n" + "="*60)
    print("TEST 6: All-Reduce (psum via shard_map)")
    print("="*60)
    
    from jax.experimental.shard_map import shard_map
    
    host_id = jax.process_index()
    num_hosts = jax.process_count()
    devices = jax.devices()
    num_devices = jax.device_count()
    
    mesh = Mesh(devices, axis_names=('devices',))
    
    try:
        # Create sharded input - each device has value 1.0
        local_input = jnp.ones((jax.local_device_count(),))
        sharding = NamedSharding(mesh, PartitionSpec('devices'))
        
        global_input = jax.make_array_from_single_device_arrays(
            (num_devices,),
            sharding,
            [jax.device_put(local_input[i:i+1], d) for i, d in enumerate(jax.local_devices())]
        )
        
        log(f"Input shape: {global_input.shape}, sharding: {global_input.sharding}")
        
        # Use shard_map for collective operations
        # Each shard gets a scalar, psum across 'devices' axis
        @jax.jit
        @partial(shard_map, mesh=mesh, 
                 in_specs=(PartitionSpec('devices'),),
                 out_specs=PartitionSpec('devices'),
                 check_rep=False)
        def psum_fn(x):
            # x is a (1,) array on each device
            return jax.lax.psum(x, 'devices')
        
        result = psum_fn(global_input)
        
        log(f"psum result shape: {result.shape}")
        log(f"psum result (first 4): {result[:4]}")
        
        # Each device should now have the sum = num_devices
        expected = float(num_devices)
        first_val = float(result[0])
        if abs(first_val - expected) < 0.01:
            log(f"psum verification PASSED (all values = {expected})")
            return True
        else:
            log(f"psum verification FAILED: got {first_val}, expected {expected}")
            return False
                
    except Exception as e:
        log(f"psum FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("JAX DISTRIBUTED MULTIHOST SYNCHRONIZATION TEST")
    print("="*70)
    print(f"Hostname: {socket.gethostname()}")
    print(f"Script start: {script_start_datetime}")
    print("="*70)
    
    results = {}
    
    # Test 1: Basic topology
    try:
        topology = test_basic_topology()
        results['topology'] = topology is not None
    except Exception as e:
        log(f"Test 1 EXCEPTION: {e}")
        results['topology'] = False
    
    # Test 2: Device mesh
    try:
        mesh = test_device_mesh()
        results['device_mesh'] = mesh is not None
    except Exception as e:
        log(f"Test 2 EXCEPTION: {e}")
        results['device_mesh'] = False
    
    # Test 3: Global sync
    try:
        results['sync'] = test_sync_global_devices()
    except Exception as e:
        log(f"Test 3 EXCEPTION: {e}")
        results['sync'] = False
    
    # Test 4: Broadcast
    try:
        results['broadcast'] = test_broadcast_one_to_all()
    except Exception as e:
        log(f"Test 4 EXCEPTION: {e}")
        results['broadcast'] = False
    
    # Test 5: Sharded JIT
    try:
        results['sharded_jit'] = test_sharded_jit()
    except Exception as e:
        log(f"Test 5 EXCEPTION: {e}")
        results['sharded_jit'] = False
    
    # Test 6: psum
    try:
        results['psum'] = test_psum()
    except Exception as e:
        log(f"Test 6 EXCEPTION: {e}")
        results['psum'] = False
    
    # Final sync
    try:
        multihost_utils.sync_global_devices("final_sync")
    except Exception as e:
        log(f"Final sync failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    host_id = jax.process_index()
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        log(f"{test_name}: {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    log(f"Total: {total_passed}/{total_tests} tests passed")
    
    print("="*70)
    elapsed = time.time() - script_start_time
    log(f"Test completed in {elapsed:.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()
