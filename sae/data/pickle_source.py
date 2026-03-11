"""Load activations from gzipped pickle shards (activation-extract format)."""

import gzip
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional

from .base import ActivationSource


class PickleShardSource(ActivationSource):
    """Load activations from the gzipped pickle shard format.

    This reads the output of the activation-extract pipeline:
        shard_NNNN.pkl.gz files where each shard is a dict:
        {layer_idx: [{'sample_idx': int, 'activation': np.ndarray, ...}, ...]}

    Args:
        shard_dir: Directory containing shard files and metadata.json.
        layer_index: Which layer to load activations from.
        compressed: Whether shards are gzip compressed.
        shuffle_shards: Randomize shard loading order.
        seed: Random seed for shard shuffling.
        gcs_path: If set, load from GCS via fsspec (e.g. "gs://bucket/prefix").
    """

    def __init__(
        self,
        shard_dir: str,
        layer_index: int,
        compressed: bool = True,
        shuffle_shards: bool = True,
        seed: int = 42,
        gcs_path: Optional[str] = None,
    ):
        self.layer_index = layer_index
        self.compressed = compressed
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self._hidden_dim = None

        if gcs_path:
            self._init_gcs(gcs_path)
        else:
            self._init_local(shard_dir)

    def _init_local(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        self.fs = None

        # Discover shard files
        ext = "*.pkl.gz" if self.compressed else "*.pkl"
        self._shard_files = sorted(self.shard_dir.glob(ext))

        if not self._shard_files:
            raise FileNotFoundError(f"No {ext} files in {self.shard_dir}")

        # Try to load metadata
        meta_path = self.shard_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = None

    def _init_gcs(self, gcs_path: str):
        import fsspec

        self.fs = fsspec.filesystem("gs")
        prefix = gcs_path.replace("gs://", "")

        # List shard files on GCS
        ext = ".pkl.gz" if self.compressed else ".pkl"
        all_files = self.fs.ls(prefix)
        self._shard_files = sorted([f for f in all_files if f.endswith(ext)])

        if not self._shard_files:
            raise FileNotFoundError(f"No {ext} files at {gcs_path}")

        # Try metadata
        meta_path = f"{prefix}/metadata.json"
        if self.fs.exists(meta_path):
            with self.fs.open(meta_path, "r") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = None

    def _load_shard(self, path) -> dict:
        """Load a single shard file."""
        if self.fs:
            with self.fs.open(path, "rb") as raw:
                if self.compressed:
                    data = gzip.decompress(raw.read())
                    return pickle.loads(data)
                else:
                    return pickle.load(raw)
        else:
            if self.compressed:
                with gzip.open(path, "rb") as f:
                    return pickle.load(f)
            else:
                with open(path, "rb") as f:
                    return pickle.load(f)

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            # Peek at first shard to detect
            shard = self._load_shard(self._shard_files[0])
            for layer_key in [self.layer_index, str(self.layer_index)]:
                if layer_key in shard:
                    samples = shard[layer_key]
                    if samples:
                        self._hidden_dim = samples[0]["activation"].shape[-1]
                        break
            if self._hidden_dim is None:
                raise ValueError(
                    f"Layer {self.layer_index} not found in shard. "
                    f"Available: {list(shard.keys())}"
                )
        return self._hidden_dim

    def _get_ordered_files(self) -> List:
        files = list(self._shard_files)
        if self.shuffle_shards:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(files)
        return files

    def iter_vectors(self) -> Iterator[np.ndarray]:
        for shard_path in self._get_ordered_files():
            shard = self._load_shard(shard_path)

            # Try both int and string keys
            samples = None
            for layer_key in [self.layer_index, str(self.layer_index)]:
                if layer_key in shard:
                    samples = shard[layer_key]
                    break

            if samples is None:
                continue

            for sample in samples:
                act = sample["activation"].astype(np.float32)
                # act is [seq_len, hidden_dim] — yield each token position
                if act.ndim == 2:
                    for i in range(act.shape[0]):
                        yield act[i]
                elif act.ndim == 1:
                    yield act
                else:
                    # [batch, seq_len, hidden_dim] or other
                    flat = act.reshape(-1, act.shape[-1])
                    for i in range(flat.shape[0]):
                        yield flat[i]
