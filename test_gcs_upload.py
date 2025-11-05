"""
Test GCS Upload Functionality

Tests the GCS upload feature of the ActivationStorage class
without running the full extraction pipeline.
"""

import numpy as np
import sys
from pathlib import Path

# Import the ActivationStorage class
from extract_activations_fineweb import ActivationStorage


def test_gcs_upload(gcs_bucket: str, gcs_prefix: str):
    """Test GCS upload with dummy activation data"""

    print("="*70)
    print("GCS UPLOAD TEST")
    print("="*70)
    print(f"Bucket: {gcs_bucket}")
    print(f"Prefix: {gcs_prefix}")
    print("="*70)

    # Create a temporary output directory for testing
    output_dir = "./test_gcs_output"

    # Initialize ActivationStorage with GCS upload enabled
    print("\n1. Initializing ActivationStorage with GCS upload...")
    try:
        storage = ActivationStorage(
            output_dir=output_dir,
            upload_to_gcs=True,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            shard_size_gb=0.001,  # Very small shard size (1MB) for testing
            compress_shards=True,
            delete_local_after_upload=False,  # Keep local files for verification
            verbose=True
        )
        print("✓ ActivationStorage initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ActivationStorage: {e}")
        return False

    # Create some dummy activations
    print("\n2. Creating dummy activation data...")
    num_samples = 5
    seq_length = 100
    hidden_dim = 896  # Qwen2.5-0.5B hidden size

    for sample_idx in range(num_samples):
        # Create dummy activation for multiple layers
        for layer_idx in [10, 11, 12]:
            # Random activation: [seq_length, hidden_dim]
            dummy_activation = np.random.randn(seq_length, hidden_dim).astype(np.float32)

            # Add to storage
            storage.add_activation(
                layer_idx=layer_idx,
                activation=dummy_activation,
                sample_idx=sample_idx,
                text_preview=f"This is test sample {sample_idx} for layer {layer_idx}"
            )

    print(f"✓ Created {num_samples} samples across 3 layers")

    # Finalize storage (will trigger shard save and upload)
    print("\n3. Finalizing storage (uploading to GCS)...")
    try:
        storage.finalize()
        print("✓ Storage finalized successfully")
    except Exception as e:
        print(f"✗ Failed to finalize storage: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify upload
    print("\n4. Verifying upload...")
    if storage.upload_to_gcs and storage.metadata:
        print(f"\n{'='*70}")
        print("UPLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Total shards created: {storage.shard_count}")
        print(f"Total samples: {storage.total_samples}")
        print(f"\nUploaded files:")

        for shard_meta in storage.metadata:
            if shard_meta.get('gcs_path'):
                print(f"  ✓ {shard_meta['gcs_path']}")
                print(f"    Size: {shard_meta['file_size_mb']:.2f} MB")
                print(f"    Samples: {shard_meta['total_samples_in_shard']}")
            else:
                print(f"  ✗ {shard_meta['filename']} - Upload failed")

        print(f"{'='*70}")

        # Check if all uploads succeeded
        failed_uploads = [s for s in storage.metadata if not s.get('gcs_path')]
        if failed_uploads:
            print(f"\n⚠ Warning: {len(failed_uploads)} shard(s) failed to upload")
            return False
        else:
            print("\n✓ All shards uploaded successfully!")
            return True
    else:
        print("✗ No uploads performed")
        return False


def verify_gcs_access(gcs_bucket: str):
    """Verify we can access the GCS bucket"""
    print("\n" + "="*70)
    print("VERIFYING GCS ACCESS")
    print("="*70)

    try:
        import fsspec
        fs = fsspec.filesystem('gs')

        # Try to list bucket contents
        print(f"Attempting to access bucket: {gcs_bucket}")
        bucket_path = f"gs://{gcs_bucket}"

        # List a few items (or empty if bucket is empty)
        items = fs.ls(bucket_path, detail=False)[:5]

        print(f"✓ Successfully connected to GCS bucket")
        print(f"  Bucket exists and is accessible")
        if items:
            print(f"  Found {len(items)} items (showing first 5)")
            for item in items:
                print(f"    - {item}")
        else:
            print(f"  Bucket is empty or has no visible items")

        return True

    except Exception as e:
        print(f"✗ Failed to access GCS bucket: {e}")
        print(f"\nPlease ensure:")
        print(f"  1. The bucket '{gcs_bucket}' exists")
        print(f"  2. You have proper authentication (GOOGLE_APPLICATION_CREDENTIALS)")
        print(f"  3. You have write permissions to the bucket")
        return False


if __name__ == '__main__':
    # Configuration
    GCS_BUCKET = "fineweb-edu-us-central1"
    GCS_PREFIX = "test_activations"  # Using underscore instead of space

    print("\n" + "="*70)
    print("GCS UPLOAD TEST SCRIPT")
    print("="*70)
    print(f"This script will test uploading dummy activation data to GCS")
    print(f"Target: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print("="*70)

    # First verify GCS access
    if not verify_gcs_access(GCS_BUCKET):
        print("\n⚠ GCS access verification failed. Continuing with test anyway...")

    # Run the test
    print("\n")
    success = test_gcs_upload(GCS_BUCKET, GCS_PREFIX)

    if success:
        print("\n" + "="*70)
        print("✓ GCS UPLOAD TEST PASSED!")
        print("="*70)
        print(f"\nYou can verify the upload at:")
        print(f"  gs://{GCS_BUCKET}/{GCS_PREFIX}/")
        print(f"\nOr use: gsutil ls gs://{GCS_BUCKET}/{GCS_PREFIX}/")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ GCS UPLOAD TEST FAILED")
        print("="*70)
        sys.exit(1)
