#!/usr/bin/env python3
"""Test worker ID detection from environment variables

This test verifies that the barrier synchronization system can correctly
detect worker IDs from environment variables before JAX initialization.
"""

import os
import sys

# Add parent directory to path to import barrier_sync directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
import barrier_sync

def test_worker_detection():
    """Test that worker ID is correctly detected from environment"""
    get_worker_id = barrier_sync.get_worker_id
    
    print("Testing worker ID detection...")
    print("=" * 60)
    
    # Test 1: CLOUD_TPU_TASK_ID (primary, set by gcloud ssh --worker=all)
    print("\n1. Testing CLOUD_TPU_TASK_ID (highest priority)...")
    os.environ['CLOUD_TPU_TASK_ID'] = '5'
    detected_id = get_worker_id()
    assert detected_id == 5, f"Expected 5, got {detected_id}"
    print(f"   ✓ Correctly detected worker_id={detected_id} from CLOUD_TPU_TASK_ID")
    del os.environ['CLOUD_TPU_TASK_ID']
    
    # Test 2: TPU_WORKER_ID (secondary)
    print("\n2. Testing TPU_WORKER_ID (second priority)...")
    os.environ['TPU_WORKER_ID'] = '3'
    detected_id = get_worker_id()
    assert detected_id == 3, f"Expected 3, got {detected_id}"
    print(f"   ✓ Correctly detected worker_id={detected_id} from TPU_WORKER_ID")
    del os.environ['TPU_WORKER_ID']
    
    # Test 3: CLOUD_TPU_TASK_ID takes precedence over TPU_WORKER_ID
    print("\n3. Testing priority: CLOUD_TPU_TASK_ID > TPU_WORKER_ID...")
    os.environ['CLOUD_TPU_TASK_ID'] = '7'
    os.environ['TPU_WORKER_ID'] = '2'
    detected_id = get_worker_id()
    assert detected_id == 7, f"Expected 7 (from CLOUD_TPU_TASK_ID), got {detected_id}"
    print(f"   ✓ Correctly prioritized CLOUD_TPU_TASK_ID over TPU_WORKER_ID")
    del os.environ['CLOUD_TPU_TASK_ID']
    del os.environ['TPU_WORKER_ID']
    
    # Test 4: Fallback to 0 when no env vars set
    print("\n4. Testing fallback to 0 when no env vars set...")
    detected_id = get_worker_id()
    assert detected_id == 0, f"Expected 0 (default), got {detected_id}"
    print(f"   ✓ Correctly defaulted to worker_id={detected_id}")
    
    # Test 5: Worker 0 detection
    print("\n5. Testing worker 0 detection (for barrier server)...")
    os.environ['CLOUD_TPU_TASK_ID'] = '0'
    detected_id = get_worker_id()
    assert detected_id == 0, f"Expected 0, got {detected_id}"
    is_worker_0 = (detected_id == 0)
    assert is_worker_0, "Worker 0 should be correctly identified"
    print(f"   ✓ Worker 0 correctly identified (is_worker_0={is_worker_0})")
    del os.environ['CLOUD_TPU_TASK_ID']
    
    print("\n" + "=" * 60)
    print("✅ All worker detection tests passed!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        test_worker_detection()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
