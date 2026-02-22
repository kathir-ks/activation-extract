#!/usr/bin/env python3
"""
Unit tests for resume logic (GCS-backed checkpoints) and stream manager.

Tests:
  1. ActivationStorage resume_from_shard numbering continuity
  2. Checkpoint save/load roundtrip (local)
  3. StreamManager: create manifest, claim, complete, status
  4. StreamManager: resume in_progress stream for same pod
  5. StreamManager: stream_range filtering
  6. Checkpoint filename consistency

Designed to run without JAX/gcsfs/fsspec dependencies (CPU-only, local-only).
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
import importlib.util
import numpy as np

# ============================================================================
# Direct module imports to avoid core/__init__.py pulling in gcsfs/JAX
# ============================================================================
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)


def _import_module_from_file(name, path):
    """Import a single .py file as a module without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # Register so sub-imports resolve
    spec.loader.exec_module(mod)
    return mod


_storage_mod = _import_module_from_file(
    'activation_storage',
    os.path.join(_root, 'core', 'activation_storage.py')
)
_stream_mod = _import_module_from_file(
    'stream_manager',
    os.path.join(_root, 'core', 'stream_manager.py')
)

ActivationStorage = _storage_mod.ActivationStorage
StreamManager = _stream_mod.StreamManager


# ============================================================================
# Inline checkpoint helpers (extracted from multihost_extract.py)
# to avoid importing the full module which requires JAX at module level.
# ============================================================================

def _checkpoint_filename(topology, host_id):
    return f"checkpoint_{topology}_host_{host_id:02d}.json"


def save_checkpoint_local(checkpoint_data, checkpoint_dir, topology, host_id):
    filename = _checkpoint_filename(topology, host_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    local_path = os.path.join(checkpoint_dir, filename)
    with open(local_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint_local(checkpoint_dir, topology, host_id):
    filename = _checkpoint_filename(topology, host_id)
    local_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            data = json.load(f)
            data['_source'] = 'local'
            return data
    return {}


# ============================================================================
# Tests
# ============================================================================

class TestActivationStorageResume(unittest.TestCase):
    """Test ActivationStorage shard numbering on resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fresh_start_shard_count(self):
        """Fresh storage starts at shard_count=0."""
        storage = ActivationStorage(output_dir=self.tmpdir, verbose=False)
        self.assertEqual(storage.shard_count, 0)
        self.assertEqual(storage.total_activations, 0)

    def test_resume_from_shard(self):
        """Resume sets shard_count and total_activations correctly."""
        storage = ActivationStorage(
            output_dir=self.tmpdir,
            verbose=False,
            resume_from_shard=5,
            resume_from_activations=1200,
        )
        self.assertEqual(storage.shard_count, 5)
        self.assertEqual(storage.total_activations, 1200)

    def test_resume_shard_naming(self):
        """After resume, next shard is numbered correctly (no overwrite)."""
        storage = ActivationStorage(
            output_dir=self.tmpdir,
            verbose=False,
            resume_from_shard=3,
            shard_size_gb=0.0000001,  # ~107 bytes — very tiny to trigger save
        )
        # Add an activation large enough to exceed threshold
        storage.add_activation(
            layer_idx=0,
            activation=np.zeros((10, 100), dtype=np.float32),  # 4000 bytes
            sample_idx=100,
            text_preview="test sample",
        )
        # shard_count should now be 4 (3 + 1)
        self.assertEqual(storage.shard_count, 4)
        # Check file was named shard_0004
        files = os.listdir(self.tmpdir)
        shard_files = [f for f in files if f.startswith('shard_')]
        self.assertTrue(any('0004' in f for f in shard_files),
                        f"Expected shard_0004 in {shard_files}")

    def test_default_backwards_compatible(self):
        """Default params produce same behavior as before (shard_count=0)."""
        storage = ActivationStorage(output_dir=self.tmpdir, verbose=False)
        self.assertEqual(storage.shard_count, 0)
        self.assertEqual(storage.total_activations, 0)
        self.assertEqual(len(storage.seen_sample_indices), 0)


class TestCheckpointSaveLoad(unittest.TestCase):
    """Test checkpoint save/load roundtrip (local only, no GCS)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_local(self):
        """Checkpoint roundtrips through local filesystem."""
        data = {
            'topology': 'v5e-64',
            'host_id': 0,
            'last_processed_sample_idx': 999,
            'total_samples_processed': 1000,
            'total_shards': 3,
            'total_activations': 600,
        }
        save_checkpoint_local(data, self.tmpdir, 'v5e-64', 0)
        loaded = load_checkpoint_local(self.tmpdir, 'v5e-64', 0)

        self.assertEqual(loaded['last_processed_sample_idx'], 999)
        self.assertEqual(loaded['total_shards'], 3)
        self.assertEqual(loaded['total_activations'], 600)
        self.assertEqual(loaded['_source'], 'local')

    def test_load_missing_returns_empty(self):
        """Loading non-existent checkpoint returns empty dict."""
        loaded = load_checkpoint_local(self.tmpdir, 'v5e-64', 5)
        self.assertEqual(loaded, {})

    def test_checkpoint_filename_consistency(self):
        """Checkpoint filename format is consistent."""
        fname = _checkpoint_filename('v5e-128', 3)
        self.assertEqual(fname, 'checkpoint_v5e-128_host_03.json')

    def test_resume_values_from_checkpoint(self):
        """Verify the resume values that would be extracted from a checkpoint."""
        data = {
            'last_processed_sample_idx': 15999,
            'total_shards': 4,
            'total_activations': 3200,
        }
        save_checkpoint_local(data, self.tmpdir, 'v5e-64', 0)
        loaded = load_checkpoint_local(self.tmpdir, 'v5e-64', 0)

        start_sample_idx = loaded.get('last_processed_sample_idx', 0) + 1
        resume_shard_count = loaded.get('total_shards', 0)
        resume_activation_count = loaded.get('total_activations', 0)

        self.assertEqual(start_sample_idx, 16000)
        self.assertEqual(resume_shard_count, 4)
        self.assertEqual(resume_activation_count, 3200)


class TestStreamManager(unittest.TestCase):
    """Test stream manifest create/claim/complete cycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.tmpdir, 'stream_manifest.json')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_manifest(self):
        """Create a manifest with correct structure."""
        sm = StreamManager(self.manifest_path, verbose=False)
        manifest = sm.create_manifest(total_streams=4, dataset_dir='/data/streams')

        self.assertEqual(manifest['total_streams'], 4)
        self.assertEqual(len(manifest['streams']), 4)
        self.assertEqual(manifest['streams']['0']['status'], 'pending')
        self.assertIn('stream_000.jsonl', manifest['streams']['0']['dataset_path'])

    def test_claim_and_complete(self):
        """Claim a stream, then mark complete."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=3, dataset_dir='/data')

        stream = sm.claim_next_stream('pod-a')
        self.assertIsNotNone(stream)
        self.assertEqual(stream['stream_id'], 0)

        status = sm.get_status_summary()
        self.assertEqual(status['in_progress'], 1)
        self.assertEqual(status['pending'], 2)

        sm.mark_stream_complete(0)
        status = sm.get_status_summary()
        self.assertEqual(status['completed'], 1)
        self.assertEqual(status['in_progress'], 0)

    def test_claim_respects_range(self):
        """Only claims streams within the assigned range."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=8, dataset_dir='/data')

        stream = sm.claim_next_stream('pod-a', stream_range=(0, 3))
        self.assertEqual(stream['stream_id'], 0)

        sm.mark_stream_complete(0)
        for _ in range(3):
            s = sm.claim_next_stream('pod-a', stream_range=(0, 3))
            if s:
                sm.mark_stream_complete(s['stream_id'])

        no_more = sm.claim_next_stream('pod-a', stream_range=(0, 3))
        self.assertIsNone(no_more)

        stream_b = sm.claim_next_stream('pod-b', stream_range=(4, 7))
        self.assertIsNotNone(stream_b)
        self.assertEqual(stream_b['stream_id'], 4)

    def test_resume_in_progress_stream(self):
        """Same pod re-claims its in_progress stream on restart."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=3, dataset_dir='/data')

        stream = sm.claim_next_stream('pod-a')
        self.assertEqual(stream['stream_id'], 0)

        # Simulate restart: new StreamManager, same manifest
        sm2 = StreamManager(self.manifest_path, verbose=False)
        resumed = sm2.claim_next_stream('pod-a')
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed['stream_id'], 0)
        self.assertTrue(resumed.get('resumed', False))

    def test_status_summary(self):
        """Status summary has correct percentages."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=4, dataset_dir='/data')

        sm.claim_next_stream('pod-a')
        sm.mark_stream_complete(0)
        sm.claim_next_stream('pod-a')

        status = sm.get_status_summary()
        self.assertEqual(status['pct_complete'], 25.0)
        self.assertIn('pod-a', status['pods_active'])

    def test_get_stream_info(self):
        """Can retrieve info for a specific stream."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=3, dataset_dir='/data')

        info = sm.get_stream_info(1)
        self.assertIsNotNone(info)
        self.assertEqual(info['status'], 'pending')

    def test_no_overwrite_existing(self):
        """create_manifest does not overwrite existing manifest by default."""
        sm = StreamManager(self.manifest_path, verbose=False)
        sm.create_manifest(total_streams=4, dataset_dir='/data')
        sm.claim_next_stream('pod-a')  # Mark one as in_progress

        # Create again without overwrite
        manifest = sm.create_manifest(total_streams=8, dataset_dir='/other')
        # Should still be 4 streams
        self.assertEqual(manifest['total_streams'], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
