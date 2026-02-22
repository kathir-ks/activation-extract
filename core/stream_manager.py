"""
Stream Manager — Track and orchestrate multi-stream extraction workloads.

Uses a GCS-backed JSON manifest to track which dataset streams are
pending / in-progress / completed.  Each TPU pod is assigned a fixed
range of stream IDs (to avoid race conditions on GCS), and processes
them sequentially with checkpoint/resume on preemption.

Usage from multihost_extract.py (--stream_mode):
    sm = StreamManager("gs://bucket/activations/stream_manifest.json")
    while True:
        stream = sm.claim_next_stream(pod_id, stream_range=(0, 7))
        if stream is None:
            break  # All streams done
        # ... run extraction for stream ...
        sm.mark_stream_complete(stream['stream_id'])
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple


class StreamManager:
    """Manage multi-stream extraction workloads via a GCS-backed manifest."""

    def __init__(
        self,
        manifest_path: str,
        verbose: bool = True,
    ):
        """
        Args:
            manifest_path: GCS path or local path to manifest JSON file.
                           e.g. "gs://bucket/activations/stream_manifest.json"
                           or   "./stream_manifest.json"
            verbose: Print progress messages
        """
        self.manifest_path = manifest_path
        self.verbose = verbose
        self.is_gcs = manifest_path.startswith("gs://")
        self._fs = None

    @property
    def fs(self):
        """Lazy-initialize fsspec GCS filesystem."""
        if self._fs is None and self.is_gcs:
            import fsspec
            self._fs = fsspec.filesystem('gs')
        return self._fs

    def _read_manifest(self) -> Dict:
        """Read manifest from GCS or local path."""
        try:
            if self.is_gcs:
                if not self.fs.exists(self.manifest_path):
                    return {}
                with self.fs.open(self.manifest_path, 'r') as f:
                    return json.load(f)
            else:
                if not os.path.exists(self.manifest_path):
                    return {}
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read manifest: {e}")
            return {}

    def _write_manifest(self, manifest: Dict):
        """Write manifest to GCS or local path."""
        try:
            if self.is_gcs:
                with self.fs.open(self.manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
            else:
                os.makedirs(os.path.dirname(self.manifest_path) or '.', exist_ok=True)
                with open(self.manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write manifest: {e}")

    def create_manifest(
        self,
        total_streams: int,
        dataset_dir: str,
        overwrite: bool = False,
    ) -> Dict:
        """Create a new stream manifest.

        Args:
            total_streams: Total number of streams
            dataset_dir: Directory/GCS path containing stream_NNN.jsonl files
            overwrite: Overwrite existing manifest
        """
        existing = self._read_manifest()
        if existing and not overwrite:
            if self.verbose:
                print(f"Manifest already exists at {self.manifest_path} "
                      f"({existing.get('total_streams', '?')} streams). "
                      f"Use overwrite=True to reset.")
            return existing

        now = datetime.now(timezone.utc).isoformat()
        manifest = {
            'total_streams': total_streams,
            'dataset_dir': dataset_dir,
            'created_at': now,
            'updated_at': now,
            'streams': {},
        }

        for i in range(total_streams):
            manifest['streams'][str(i)] = {
                'status': 'pending',
                'dataset_path': f"{dataset_dir}/stream_{i:03d}.jsonl",
            }

        self._write_manifest(manifest)
        if self.verbose:
            print(f"✅ Created manifest with {total_streams} streams at {self.manifest_path}")
        return manifest

    def claim_next_stream(
        self,
        pod_id: str,
        stream_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict]:
        """Claim the next pending stream within the assigned range.

        Args:
            pod_id: Identifier for this TPU pod (e.g. "v5e-64-eu")
            stream_range: (start, end) inclusive range of stream IDs assigned to this pod.
                          None = claim from any pending stream.

        Returns:
            Dict with stream info, or None if all streams in range are done.
        """
        manifest = self._read_manifest()
        if not manifest:
            print("Warning: No manifest found.")
            return None

        now = datetime.now(timezone.utc).isoformat()

        for stream_id_str, stream_info in manifest['streams'].items():
            stream_id = int(stream_id_str)

            # Filter by range if specified
            if stream_range is not None:
                if stream_id < stream_range[0] or stream_id > stream_range[1]:
                    continue

            if stream_info['status'] == 'pending':
                # Claim it
                stream_info['status'] = 'in_progress'
                stream_info['pod_id'] = pod_id
                stream_info['started_at'] = now
                manifest['updated_at'] = now
                self._write_manifest(manifest)

                if self.verbose:
                    print(f"📋 Claimed stream {stream_id} "
                          f"({stream_info['dataset_path']})")
                return {
                    'stream_id': stream_id,
                    'dataset_path': stream_info['dataset_path'],
                    'pod_id': pod_id,
                }

            elif stream_info['status'] == 'in_progress':
                # Check if this was our interrupted stream (resume)
                if stream_info.get('pod_id') == pod_id:
                    if self.verbose:
                        print(f"📌 Resuming stream {stream_id} "
                              f"(started at {stream_info.get('started_at', '?')})")
                    return {
                        'stream_id': stream_id,
                        'dataset_path': stream_info['dataset_path'],
                        'pod_id': pod_id,
                        'resumed': True,
                    }

        if self.verbose:
            pending_in_range = sum(
                1 for sid, s in manifest['streams'].items()
                if s['status'] != 'completed'
                and (stream_range is None
                     or stream_range[0] <= int(sid) <= stream_range[1])
            )
            if pending_in_range == 0:
                print("✅ All streams in range completed!")
            else:
                print(f"⏳ {pending_in_range} streams still in progress (by other pods)")

        return None

    def mark_stream_complete(self, stream_id: int):
        """Mark a stream as completed."""
        manifest = self._read_manifest()
        if not manifest:
            return

        now = datetime.now(timezone.utc).isoformat()
        stream_key = str(stream_id)

        if stream_key in manifest['streams']:
            manifest['streams'][stream_key]['status'] = 'completed'
            manifest['streams'][stream_key]['completed_at'] = now
            manifest['updated_at'] = now
            self._write_manifest(manifest)

            if self.verbose:
                print(f"✅ Stream {stream_id} marked as completed")

    def get_status_summary(self) -> Dict:
        """Get overall progress summary."""
        manifest = self._read_manifest()
        if not manifest:
            return {'error': 'No manifest found'}

        streams = manifest.get('streams', {})
        total = len(streams)
        completed = sum(1 for s in streams.values() if s['status'] == 'completed')
        in_progress = sum(1 for s in streams.values() if s['status'] == 'in_progress')
        pending = sum(1 for s in streams.values() if s['status'] == 'pending')

        # Group in_progress by pod
        pods_active = {}
        for sid, s in streams.items():
            if s['status'] == 'in_progress':
                pod = s.get('pod_id', 'unknown')
                pods_active.setdefault(pod, []).append(int(sid))

        return {
            'total': total,
            'completed': completed,
            'in_progress': in_progress,
            'pending': pending,
            'pct_complete': round(completed / total * 100, 1) if total > 0 else 0,
            'pods_active': pods_active,
            'dataset_dir': manifest.get('dataset_dir', ''),
        }

    def get_stream_info(self, stream_id: int) -> Optional[Dict]:
        """Get info for a specific stream."""
        manifest = self._read_manifest()
        if not manifest:
            return None
        return manifest.get('streams', {}).get(str(stream_id))
