#!/usr/bin/env python3
"""
Shard Manager - Handle shard allocation and tracking for distributed processing

This module provides utilities for:
- Claiming an available shard
- Marking shards as in-use/completed
- Getting shard file lists
- Coordinating between multiple workers
"""

import json
import time
import fcntl
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
import gcsfs
from datetime import datetime


class ShardManager:
    """Manage shard allocation and tracking"""

    def __init__(self, dataset_dir: str, worker_id: str = None):
        """
        Initialize shard manager

        Args:
            dataset_dir: Directory containing sharded dataset
            worker_id: Unique identifier for this worker (e.g., host_id)
        """
        self.dataset_dir = dataset_dir
        self.worker_id = worker_id or f"worker_{int(time.time())}"
        self.is_gcs = dataset_dir.startswith("gs://")

        if self.is_gcs:
            self.fs = gcsfs.GCSFileSystem()
        else:
            self.fs = None

    def claim_shard(self, preferred_shard_id: Optional[int] = None) -> Optional[Dict]:
        """
        Claim an available shard for processing

        Args:
            preferred_shard_id: Preferred shard ID (if available), otherwise get any available

        Returns:
            Shard info dict if successful, None if no shards available
        """
        master_file = f"{self.dataset_dir}/master_metadata.json"

        # Load master metadata
        master = self._read_json(master_file)
        if not master:
            raise FileNotFoundError(f"Master metadata not found: {master_file}")

        # Try preferred shard first
        if preferred_shard_id is not None:
            shard_dir = f"{self.dataset_dir}/shard_{preferred_shard_id:03d}"
            shard = self._try_claim_shard(shard_dir, preferred_shard_id)
            if shard:
                return shard
          
        # Otherwise, find any available shard
        for shard_info in master["shards"]:
            shard_id = shard_info["shard_id"]
            shard_dir = f"{self.dataset_dir}/{shard_info['directory']}"

            shard = self._try_claim_shard(shard_dir, shard_id)
            if shard:
                return shard

        return None  # No available shards

    def _try_claim_shard(self, shard_dir: str, shard_id: int) -> Optional[Dict]:
        """Try to claim a specific shard"""
        metadata_file = f"{shard_dir}/metadata.json"

        # Lock and read metadata
        metadata = self._read_json(metadata_file)
        if not metadata:
            return None

        # Check if available
        if metadata.get("status") != "available":
            return None

        # Claim the shard
        metadata["status"] = "in_use"
        metadata["assigned_to"] = self.worker_id
        metadata["started_at"] = datetime.utcnow().isoformat()

        # Write back
        success = self._write_json(metadata_file, metadata)
        if not success:
            return None

        # Return shard info
        return {
            "shard_id": shard_id,
            "shard_dir": shard_dir,
            "metadata": metadata,
            "chunks": self._get_chunk_files(shard_dir, metadata)
        }

    def mark_completed(self, shard_id: int):
        """Mark a shard as completed"""
        shard_dir = f"{self.dataset_dir}/shard_{shard_id:03d}"
        metadata_file = f"{shard_dir}/metadata.json"

        metadata = self._read_json(metadata_file)
        if metadata:
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.utcnow().isoformat()
            self._write_json(metadata_file, metadata)

    def mark_failed(self, shard_id: int):
        """Mark a shard as available again (failed processing)"""
        shard_dir = f"{self.dataset_dir}/shard_{shard_id:03d}"
        metadata_file = f"{shard_dir}/metadata.json"

        metadata = self._read_json(metadata_file)
        if metadata:
            metadata["status"] = "available"
            metadata["assigned_to"] = None
            metadata["started_at"] = None
            self._write_json(metadata_file, metadata)

    def get_shard_status(self) -> List[Dict]:
        """Get status of all shards"""
        master_file = f"{self.dataset_dir}/master_metadata.json"
        master = self._read_json(master_file)

        if not master:
            return []

        statuses = []
        for shard_info in master["shards"]:
            shard_dir = f"{self.dataset_dir}/{shard_info['directory']}"
            metadata_file = f"{shard_dir}/metadata.json"
            metadata = self._read_json(metadata_file)

            if metadata:
                statuses.append({
                    "shard_id": shard_info["shard_id"],
                    "status": metadata.get("status", "unknown"),
                    "assigned_to": metadata.get("assigned_to"),
                    "total_tasks": metadata.get("total_tasks", 0),
                    "num_chunks": metadata.get("num_chunks", 0)
                })

        return statuses

    def _get_chunk_files(self, shard_dir: str, metadata: Dict) -> List[str]:
        """Get list of chunk files for a shard"""
        chunks = []
        for chunk_info in metadata.get("chunks", []):
            chunk_file = f"{shard_dir}/{chunk_info['file']}"
            chunks.append(chunk_file)
        return chunks

    def _read_json(self, file_path: str) -> Optional[Dict]:
        """Read JSON file (local or GCS)"""
        try:
            if self.is_gcs:
                if not self.fs.exists(file_path):
                    return None
                with self.fs.open(file_path, 'r') as f:
                    return json.load(f)
            else:
                if not Path(file_path).exists():
                    return None
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return None

    def _write_json(self, file_path: str, data: Dict) -> bool:
        """Write JSON file with atomic operations"""
        try:
            if self.is_gcs:
                # GCS writes are atomic
                with self.fs.open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            else:
                # Local write with file locking
                temp_file = file_path + ".tmp"
                with open(temp_file, 'w') as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Atomic rename
                Path(temp_file).rename(file_path)
                return True
        except Exception as e:
            print(f"Warning: Failed to write {file_path}: {e}")
            return False


def load_shard_chunks(chunk_files: List[str], is_gcs: bool = False) -> List[Dict]:
    """
    Load all tasks from shard chunk files

    Args:
        chunk_files: List of chunk file paths
        is_gcs: Whether files are on GCS

    Returns:
        List of task dictionaries
    """
    tasks = []

    if is_gcs:
        fs = gcsfs.GCSFileSystem()

    for chunk_file in chunk_files:
        try:
            if is_gcs:
                with fs.open(chunk_file, 'r') as f:
                    for line in f:
                        tasks.append(json.loads(line.strip()))
            else:
                with open(chunk_file, 'r') as f:
                    for line in f:
                        tasks.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Warning: Failed to load chunk {chunk_file}: {e}")

    return tasks


def main():
    """Test shard manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Shard Manager CLI")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Sharded dataset directory")
    parser.add_argument('--action', type=str, choices=['claim', 'status', 'complete', 'reset'],
                       required=True, help="Action to perform")
    parser.add_argument('--shard_id', type=int, help="Shard ID (for complete/reset)")
    parser.add_argument('--worker_id', type=str, help="Worker ID")

    args = parser.parse_args()

    manager = ShardManager(args.dataset_dir, args.worker_id)

    if args.action == 'claim':
        shard = manager.claim_shard()
        if shard:
            print(f"✓ Claimed shard {shard['shard_id']}")
            print(f"  Directory: {shard['shard_dir']}")
            print(f"  Tasks: {shard['metadata']['total_tasks']}")
            print(f"  Chunks: {len(shard['chunks'])}")
        else:
            print("✗ No available shards")

    elif args.action == 'status':
        statuses = manager.get_shard_status()
        print("Shard Status:")
        print("-" * 70)
        for status in statuses:
            print(f"  Shard {status['shard_id']}: {status['status']}")
            if status['assigned_to']:
                print(f"    Assigned to: {status['assigned_to']}")
            print(f"    Tasks: {status['total_tasks']}, Chunks: {status['num_chunks']}")

    elif args.action == 'complete':
        if args.shard_id is None:
            print("Error: --shard_id required for complete action")
            return
        manager.mark_completed(args.shard_id)
        print(f"✓ Marked shard {args.shard_id} as completed")

    elif args.action == 'reset':
        if args.shard_id is None:
            print("Error: --shard_id required for reset action")
            return
        manager.mark_failed(args.shard_id)
        print(f"✓ Reset shard {args.shard_id} to available")


if __name__ == "__main__":
    main()
