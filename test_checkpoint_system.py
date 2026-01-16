#!/usr/bin/env python3
"""
Quick test to verify checkpoint system works correctly
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from extract_activations import (
    load_checkpoint,
    save_checkpoint,
    get_worker_id,
    ExtractionConfig
)


def test_checkpoint_save_load():
    """Test checkpoint save and load"""
    print("Testing checkpoint save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.json")

        # Save checkpoint
        checkpoint_data = {
            'worker_id': 5,
            'last_processed_sample_idx': 1250,
            'total_samples_processed': 1251,
            'total_shards': 3,
            'dataset_path': 'data/test.jsonl',
            'model_path': 'Qwen/Qwen2.5-0.5B',
            'status': 'in_progress'
        }
        save_checkpoint(checkpoint_path, checkpoint_data)

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path)

        # Verify
        assert loaded == checkpoint_data, "Checkpoint data mismatch"
        print("  ✓ Checkpoint save/load works")


def test_worker_id_detection():
    """Test worker ID auto-detection"""
    print("\nTesting worker ID detection...")

    # Test with TPU_WORKER_ID
    os.environ['TPU_WORKER_ID'] = '7'
    worker_id = get_worker_id()
    assert worker_id == 7, f"Expected 7, got {worker_id}"
    print("  ✓ TPU_WORKER_ID detection works")

    # Test with WORKER_ID
    del os.environ['TPU_WORKER_ID']
    os.environ['WORKER_ID'] = '9'
    worker_id = get_worker_id()
    assert worker_id == 9, f"Expected 9, got {worker_id}"
    print("  ✓ WORKER_ID fallback works")

    # Test default (no env vars)
    del os.environ['WORKER_ID']
    worker_id = get_worker_id()
    assert worker_id == 0, f"Expected 0, got {worker_id}"
    print("  ✓ Default worker ID (0) works")


def test_extraction_config():
    """Test ExtractionConfig with worker settings"""
    print("\nTesting ExtractionConfig...")

    # Set worker ID in environment
    os.environ['TPU_WORKER_ID'] = '5'

    # Create config
    config = ExtractionConfig(
        dataset_path='test.jsonl',
        upload_to_gcs=True,
        gcs_bucket='test-bucket',
        gcs_prefix='activations'
    )

    # Verify worker_id was auto-detected
    assert config.worker_id == 5, f"Expected worker_id=5, got {config.worker_id}"
    print("  ✓ Worker ID auto-detection in config works")

    # Verify GCS prefix includes worker folder
    expected_prefix = 'activations/tpu_5'
    assert config.gcs_prefix == expected_prefix, f"Expected '{expected_prefix}', got '{config.gcs_prefix}'"
    print(f"  ✓ GCS prefix correctly set to: {config.gcs_prefix}")

    # Test with explicit worker_id
    config2 = ExtractionConfig(
        worker_id=10,
        dataset_path='test.jsonl',
        upload_to_gcs=True,
        gcs_bucket='test-bucket',
        gcs_prefix='activations'
    )
    assert config2.worker_id == 10, f"Expected worker_id=10, got {config2.worker_id}"
    assert config2.gcs_prefix == 'activations/tpu_10', f"Expected 'activations/tpu_10', got '{config2.gcs_prefix}'"
    print("  ✓ Explicit worker_id override works")


def test_checkpoint_with_missing_file():
    """Test loading checkpoint when file doesn't exist"""
    print("\nTesting checkpoint with missing file...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "nonexistent.json")
        loaded = load_checkpoint(checkpoint_path)
        assert loaded == {}, f"Expected empty dict, got {loaded}"
        print("  ✓ Missing checkpoint returns empty dict")


def main():
    print("="*70)
    print("CHECKPOINT SYSTEM TESTS")
    print("="*70)

    try:
        test_checkpoint_save_load()
        test_worker_id_detection()
        test_extraction_config()
        test_checkpoint_with_missing_file()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
