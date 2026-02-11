"""
Activation Storage Utilities

This module handles saving activations to disk and optionally uploading to GCS
with automatic sharding based on size.
"""

import json
import pickle
import gzip
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict


class ActivationStorage:
    """Handle saving activations to disk and optionally uploading to GCS with sharding"""

    def __init__(
        self,
        output_dir: str,
        upload_to_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = 'activations',
        shard_size_gb: float = 1.0,
        compress_shards: bool = True,
        delete_local_after_upload: bool = False,
        verbose: bool = True
    ):
        """
        Initialize activation storage

        Args:
            output_dir: Local directory for saving activations
            upload_to_gcs: Whether to upload to Google Cloud Storage
            gcs_bucket: GCS bucket name (required if upload_to_gcs=True)
            gcs_prefix: Prefix for GCS paths
            shard_size_gb: Target size for each shard in GB
            compress_shards: Whether to gzip compress shards
            delete_local_after_upload: Delete local files after GCS upload
            verbose: Print progress messages
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = {}  # layer_idx -> list of activations
        self.buffer_size_bytes = 0  # Track buffer size in bytes
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes

        self.metadata = []
        self.shard_count = 0
        self.total_activations = 0  # Total activation tensors stored (layers × samples)
        self.seen_sample_indices = set()  # Track unique sample indices
        self.verbose = verbose

        # GCS settings
        self.upload_to_gcs = upload_to_gcs
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.compress_shards = compress_shards
        self.delete_local_after_upload = delete_local_after_upload

        # Initialize fsspec filesystem for GCS if needed
        self.fs = None
        if self.upload_to_gcs:
            try:
                import fsspec
                self.fs = fsspec.filesystem('gs')  # GCS filesystem
                if self.verbose:
                    print(f"✓ fsspec GCS filesystem initialized for bucket: {gcs_bucket}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize fsspec GCS filesystem: {e}")

    def add_activation(
        self,
        layer_idx: int,
        activation: np.ndarray,
        sample_idx: int,
        text_preview: str
    ):
        """
        Add activation to buffer and check if shard size exceeded

        Args:
            layer_idx: Layer index
            activation: Activation tensor (numpy array)
            sample_idx: Sample index for this activation
            text_preview: Text preview for metadata
        """
        if layer_idx not in self.buffer:
            self.buffer[layer_idx] = []

        activation_data = {
            'sample_idx': sample_idx,
            'activation': activation,
            'shape': activation.shape,
            'text_preview': text_preview[:200]  # Save first 200 chars for reference
        }

        self.buffer[layer_idx].append(activation_data)
        self.total_activations += 1
        self.seen_sample_indices.add(sample_idx)

        # Estimate size (activation array + metadata overhead)
        activation_size = activation.nbytes
        metadata_overhead = 1024  # Rough estimate for metadata
        self.buffer_size_bytes += activation_size + metadata_overhead

        # Check if we should save a shard
        if self.buffer_size_bytes >= self.shard_size_bytes:
            self._save_shard()

    def _save_shard(self):
        """Save current buffer as a shard"""
        if not self.buffer:
            return

        self.shard_count += 1
        shard_name = f"shard_{self.shard_count:04d}.pkl"

        if self.compress_shards:
            shard_name += ".gz"

        shard_path = self.output_dir / shard_name

        if self.verbose:
            size_mb = self.buffer_size_bytes / (1024 * 1024)
            print(f"\n{'='*70}")
            print(f"Saving shard {self.shard_count}: {shard_name} (~{size_mb:.1f} MB)")
            print(f"{'='*70}")

        # Save to local file
        if self.compress_shards:
            with gzip.open(shard_path, 'wb') as f:
                pickle.dump(self.buffer, f)
        else:
            with open(shard_path, 'wb') as f:
                pickle.dump(self.buffer, f)

        # Get actual file size
        file_size_mb = shard_path.stat().st_size / (1024 * 1024)

        if self.verbose:
            print(f"  ✓ Saved locally: {file_size_mb:.1f} MB")
            for layer_idx, activations in self.buffer.items():
                print(f"    Layer {layer_idx}: {len(activations)} samples")

        # Upload to GCS if enabled
        gcs_path = None
        if self.upload_to_gcs:
            gcs_path = self._upload_to_gcs(shard_path, shard_name)

        # Record metadata
        self.metadata.append({
            'shard_id': self.shard_count,
            'filename': shard_name,
            'local_path': str(shard_path),
            'gcs_path': gcs_path,
            'file_size_mb': file_size_mb,
            'buffer_size_mb': self.buffer_size_bytes / (1024 * 1024),
            'layers': list(self.buffer.keys()),
            'samples_per_layer': {layer: len(acts) for layer, acts in self.buffer.items()},
            'total_samples_in_shard': sum(len(acts) for acts in self.buffer.values())
        })

        # Delete local file if requested
        if self.upload_to_gcs and self.delete_local_after_upload and gcs_path:
            shard_path.unlink()
            if self.verbose:
                print(f"  ✓ Deleted local file (uploaded to GCS)")

        # Clear buffer
        self.buffer = {}
        self.buffer_size_bytes = 0

    def _upload_to_gcs(self, local_path: Path, shard_name: str) -> Optional[str]:
        """Upload shard to GCS using fsspec"""
        try:
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{shard_name}"

            if self.verbose:
                print(f"  ⬆ Uploading to {gcs_path}...")

            # Upload using fsspec
            self.fs.put_file(str(local_path), gcs_path)

            if self.verbose:
                print(f"  ✓ Uploaded to GCS: {gcs_path}")

            return gcs_path
        except Exception as e:
            print(f"  ✗ Failed to upload to GCS: {e}")
            return None

    def finalize(self):
        """Save remaining activations and metadata"""
        # Save any remaining data
        if self.buffer:
            self._save_shard()

        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'total_shards': self.shard_count,
                'total_activations': self.total_activations,
                'total_unique_samples': len(self.seen_sample_indices),
                'shard_size_gb': self.shard_size_bytes / (1024 * 1024 * 1024),
                'upload_to_gcs': self.upload_to_gcs,
                'gcs_bucket': self.gcs_bucket,
                'gcs_prefix': self.gcs_prefix,
                'shards': self.metadata
            }, f, indent=2)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STORAGE SUMMARY")
            print(f"{'='*70}")
            print(f"  Total shards: {self.shard_count}")
            print(f"  Total samples: {len(self.seen_sample_indices)}")
            print(f"  Total activations: {self.total_activations}")
            print(f"  Metadata: {metadata_file}")
            if self.upload_to_gcs:
                print(f"  GCS bucket: gs://{self.gcs_bucket}/{self.gcs_prefix}/")
            print(f"{'='*70}")


def load_activation_shard(shard_path: str, compressed: bool = True) -> Dict:
    """
    Load activation shard from disk

    Args:
        shard_path: Path to shard file
        compressed: Whether the shard is gzip compressed

    Returns:
        Dictionary mapping layer indices to activation lists
    """
    if compressed:
        with gzip.open(shard_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(shard_path, 'rb') as f:
            return pickle.load(f)
