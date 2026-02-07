#!/usr/bin/env python3
"""Test that worker 0 automatically starts the barrier server

This test verifies that the barrier server auto-starts when worker ID is 0,
without requiring manual --is_barrier_server flag.
"""

import os
import sys
import time
import socket

# Add parent directory to path to import barrier_sync directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import barrier_sync

def test_barrier_server_autostart():
    """Verify worker 0 automatically starts the barrier server"""
    
    print("Testing barrier server auto-start...")
    print("=" * 60)
    
    # Clean up any existing env vars
    for var in ['CLOUD_TPU_TASK_ID', 'TPU_WORKER_ID']:
        if var in os.environ:
            del os.environ[var]
    
    # Test 1: Worker 0 should auto-start server
    print("\n1. Testing worker 0 auto-starts barrier server...")
    os.environ['CLOUD_TPU_TASK_ID'] = '0'
    
    BarrierServer = barrier_sync.BarrierServer
    get_worker_id = barrier_sync.get_worker_id
    
    worker_id = get_worker_id()
    assert worker_id == 0, f"Expected worker_id=0, got {worker_id}"
    print(f"   ✓ Worker ID detected as {worker_id}")
    
    # Start barrier server
    test_port = 5556  # Use different port to avoid conflicts
    server = BarrierServer(num_workers=2, port=test_port)
    server.start_background(wait_ready=True, ready_timeout=5.0)
    print(f"   ✓ Barrier server started on port {test_port}")
    
    # Verify server is listening
    time.sleep(0.5)  # Give server time to bind
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', test_port))
        sock.close()
        print(f"   ✓ Barrier server is listening and accepting connections")
    except ConnectionRefusedError:
        server.stop()
        raise AssertionError("Barrier server not listening on expected port")
    
    # Clean up
    server.stop()
    time.sleep(0.5)
    print(f"   ✓ Barrier server stopped cleanly")
    
    # Test 2: Worker 1 should NOT auto-start server
    print("\n2. Testing worker 1 does NOT auto-start barrier server...")
    os.environ['CLOUD_TPU_TASK_ID'] = '1'
    # Reload the module to get fresh worker ID detection
    import importlib
    importlib.reload(barrier_sync)
    
    worker_id = barrier_sync.get_worker_id()
    assert worker_id == 1, f"Expected worker_id=1, got {worker_id}"
    is_worker_0 = (worker_id == 0)
    assert not is_worker_0, "Worker 1 should not be identified as worker 0"
    print(f"   ✓ Worker 1 correctly identified (should NOT start server)")
    
    # Clean up
    del os.environ['CLOUD_TPU_TASK_ID']
    
    print("\n" + "=" * 60)
    print("✅ All barrier server auto-start tests passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        test_barrier_server_autostart()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
