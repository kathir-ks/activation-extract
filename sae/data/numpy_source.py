"""Load activations from .npy / .npz files."""

import glob as globlib
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional

from .base import ActivationSource


class NumpySource(ActivationSource):
    """Load activations from numpy files.

    Supports:
        - Single .npy file: expected shape [N, hidden_dim]
        - Single .npz file: specify key to load
        - Directory of .npy files: loaded in sorted order
        - 3D arrays [N, seq_len, hidden_dim]: flattened to [N*seq_len, hidden_dim]

    Args:
        path: Path to .npy file, .npz file, or directory of .npy files.
        key: Key for .npz files (ignored for .npy).
        flatten_sequences: If True, reshape [N, seq_len, dim] -> [N*seq_len, dim].
    """

    def __init__(
        self,
        path: str,
        key: Optional[str] = None,
        flatten_sequences: bool = True,
    ):
        self.path = Path(path)
        self.key = key
        self.flatten_sequences = flatten_sequences
        self._files = self._discover_files()
        self._hidden_dim = self._detect_hidden_dim()

    def _discover_files(self) -> List[Path]:
        if self.path.is_file():
            return [self.path]
        elif self.path.is_dir():
            files = sorted(self.path.glob("*.npy"))
            if not files:
                raise FileNotFoundError(f"No .npy files in {self.path}")
            return files
        else:
            # Try as glob pattern
            matches = sorted(globlib.glob(str(self.path)))
            if not matches:
                raise FileNotFoundError(f"No files matching {self.path}")
            return [Path(m) for m in matches]

    def _detect_hidden_dim(self) -> int:
        f = self._files[0]
        if f.suffix == ".npz":
            with np.load(f) as data:
                arr = data[self.key] if self.key else data[list(data.keys())[0]]
        else:
            arr = np.load(f, mmap_mode="r")
        return arr.shape[-1]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def iter_vectors(self) -> Iterator[np.ndarray]:
        for f in self._files:
            if f.suffix == ".npz":
                with np.load(f) as data:
                    arr = data[self.key] if self.key else data[list(data.keys())[0]]
            else:
                arr = np.load(f)

            arr = arr.astype(np.float32)

            if arr.ndim == 3 and self.flatten_sequences:
                # [N, seq_len, hidden_dim] -> [N*seq_len, hidden_dim]
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)

            for i in range(arr.shape[0]):
                yield arr[i]

    def iter_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        """Efficient batched loading from numpy arrays."""
        for f in self._files:
            if f.suffix == ".npz":
                with np.load(f) as data:
                    arr = data[self.key] if self.key else data[list(data.keys())[0]]
            else:
                arr = np.load(f)

            arr = arr.astype(np.float32)

            if arr.ndim == 3 and self.flatten_sequences:
                arr = arr.reshape(-1, arr.shape[-1])

            for start in range(0, arr.shape[0], batch_size):
                yield arr[start : start + batch_size]
