"""
Shard Management Utilities

Provides utilities for distributed dataset processing:
- Shard allocation and tracking
- Worker coordination
- Support for both local and GCS storage
"""

import json
import time
import fcntl
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class ShardManager:
    """Manage shard allocation and tracking for distributed processing."""

    def __init__(self, dataset_dir: str, worker_id: Optional[str] = None):
        """
        Initialize shard manager.

        Args:
            dataset_dir: Directory containing sharded dataset
            worker_id: Unique identifier for this worker
        """
        self.dataset_dir = dataset_dir
        self.worker_id = worker_id or f"worker_{int(time.time())}"
        self.is_gcs = dataset_dir.startswith("gs://")

        if self.is_gcs:
            import gcsfs
            self.fs = gcsfs.GCSFileSystem()
        else:
            self.fs = None

    def claim_shard(self, preferred_shard_id: Optional[int] = None) -> Optional[Dict]:
        """
        Claim an available shard for processing.

        Args:
            preferred_shard_id: Preferred shard ID (if available)

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

        return None

    def _try_claim_shard(self, shard_dir: str, shard_id: int) -> Optional[Dict]:
        """Try to claim a specific shard."""
        metadata_file = f"{shard_dir}/metadata.json"

        metadata = self._read_json(metadata_file)
        if not metadata:
            return None

        if metadata.get("status") != "available":
            return None

        # Claim the shard
        metadata["status"] = "in_use"
        metadata["assigned_to"] = self.worker_id
        metadata["started_at"] = datetime.utcnow().isoformat()

        success = self._write_json(metadata_file, metadata)
        if not success:
            return None

        return {
            "shard_id": shard_id,
            "shard_dir": shard_dir,
            "metadata": metadata,
            "chunks": self._get_chunk_files(shard_dir, metadata)
        }

    def mark_completed(self, shard_id: int) -> None:
        """Mark a shard as completed."""
        shard_dir = f"{self.dataset_dir}/shard_{shard_id:03d}"
        metadata_file = f"{shard_dir}/metadata.json"

        metadata = self._read_json(metadata_file)
        if metadata:
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.utcnow().isoformat()
            self._write_json(metadata_file, metadata)

    def mark_failed(self, shard_id: int) -> None:
        """Mark a shard as available again (failed processing)."""
        shard_dir = f"{self.dataset_dir}/shard_{shard_id:03d}"
        metadata_file = f"{shard_dir}/metadata.json"

        metadata = self._read_json(metadata_file)
        if metadata:
            metadata["status"] = "available"
            metadata["assigned_to"] = None
            metadata["started_at"] = None
            self._write_json(metadata_file, metadata)

    def get_shard_status(self) -> List[Dict]:
        """Get status of all shards."""
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
        """Get list of chunk files for a shard."""
        chunks = []
        for chunk_info in metadata.get("chunks", []):
            chunk_file = f"{shard_dir}/{chunk_info['file']}"
            chunks.append(chunk_file)
        return chunks

    def _read_json(self, file_path: str) -> Optional[Dict]:
        """Read JSON file (local or GCS)."""
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
        """Write JSON file with atomic operations."""
        try:
            if self.is_gcs:
                with self.fs.open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            else:
                temp_file = file_path + ".tmp"
                with open(temp_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(data, f, indent=2)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                Path(temp_file).rename(file_path)
                return True
        except Exception as e:
            print(f"Warning: Failed to write {file_path}: {e}")
            return False


def load_shard_chunks(chunk_files: List[str], is_gcs: bool = False) -> List[Dict]:
    """
    Load all tasks from shard chunk files.

    Args:
        chunk_files: List of chunk file paths
        is_gcs: Whether files are on GCS

    Returns:
        List of task dictionaries
    """
    tasks = []

    if is_gcs:
        import gcsfs
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


def create_sharded_dataset(
    input_file: str,
    output_dir: str,
    tasks_per_shard: int = 1000,
    tasks_per_chunk: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Create a sharded dataset from a JSONL file.

    Args:
        input_file: Input JSONL file path
        output_dir: Output directory for sharded dataset
        tasks_per_shard: Number of tasks per shard
        tasks_per_chunk: Number of tasks per chunk within a shard
        verbose: Print progress

    Returns:
        Dictionary with shard metadata
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)

    # Count tasks
    task_count = 0
    with open(input_file, 'r') as f:
        for _ in f:
            task_count += 1

    if verbose:
        print(f"Total tasks: {task_count}")

    num_shards = (task_count + tasks_per_shard - 1) // tasks_per_shard

    shard_metadata = {
        "total_tasks": task_count,
        "num_shards": num_shards,
        "tasks_per_shard": tasks_per_shard,
        "tasks_per_chunk": tasks_per_chunk,
        "shards": []
    }

    with open(input_file, 'r') as f:
        for shard_id in range(num_shards):
            shard_dir = os.path.join(output_dir, f"shard_{shard_id:03d}")
            os.makedirs(shard_dir, exist_ok=True)

            shard_tasks = []
            for _ in range(tasks_per_shard):
                line = f.readline()
                if not line:
                    break
                shard_tasks.append(json.loads(line.strip()))

            # Write chunks
            chunks = []
            for chunk_id, i in enumerate(range(0, len(shard_tasks), tasks_per_chunk)):
                chunk_tasks = shard_tasks[i:i + tasks_per_chunk]
                chunk_file = f"chunk_{chunk_id:03d}.jsonl"
                chunk_path = os.path.join(shard_dir, chunk_file)

                with open(chunk_path, 'w') as cf:
                    for task in chunk_tasks:
                        cf.write(json.dumps(task) + '\n')

                chunks.append({
                    "file": chunk_file,
                    "num_tasks": len(chunk_tasks)
                })

            # Write shard metadata
            shard_info = {
                "shard_id": shard_id,
                "total_tasks": len(shard_tasks),
                "num_chunks": len(chunks),
                "chunks": chunks,
                "status": "available"
            }

            with open(os.path.join(shard_dir, "metadata.json"), 'w') as mf:
                json.dump(shard_info, mf, indent=2)

            shard_metadata["shards"].append({
                "shard_id": shard_id,
                "directory": f"shard_{shard_id:03d}",
                "total_tasks": len(shard_tasks)
            })

            if verbose:
                print(f"  Created shard {shard_id}: {len(shard_tasks)} tasks, {len(chunks)} chunks")

    # Write master metadata
    with open(os.path.join(output_dir, "master_metadata.json"), 'w') as f:
        json.dump(shard_metadata, f, indent=2)

    return shard_metadata
