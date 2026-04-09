"""Load activations from safetensors files."""

import glob as globlib
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional

from .base import ActivationSource


class SafetensorsSource(ActivationSource):
    """Load activations from safetensors files.

    Args:
        path: Path to .safetensors file or directory of them.
        tensor_name: Name of the tensor key to load.
        flatten_sequences: If True, reshape [N, seq_len, dim] -> [N*seq_len, dim].
    """

    def __init__(
        self,
        path: str,
        tensor_name: str = "activations",
        flatten_sequences: bool = True,
    ):
        self.path = Path(path)
        self.tensor_name = tensor_name
        self.flatten_sequences = flatten_sequences
        self._files = self._discover_files()
        self._hidden_dim = self._detect_hidden_dim()

    def _discover_files(self) -> List[Path]:
        if self.path.is_file():
            return [self.path]
        elif self.path.is_dir():
            files = sorted(self.path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No .safetensors files in {self.path}")
            return files
        else:
            matches = sorted(globlib.glob(str(self.path)))
            if not matches:
                raise FileNotFoundError(f"No files matching {self.path}")
            return [Path(m) for m in matches]

    def _detect_hidden_dim(self) -> int:
        from safetensors.numpy import load_file

        data = load_file(str(self._files[0]))
        if self.tensor_name in data:
            return data[self.tensor_name].shape[-1]
        # Fall back to first available key
        first_key = list(data.keys())[0]
        return data[first_key].shape[-1]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def iter_vectors(self) -> Iterator[np.ndarray]:
        from safetensors.numpy import load_file

        for f in self._files:
            data = load_file(str(f))
            arr = data.get(self.tensor_name)
            if arr is None:
                arr = data[list(data.keys())[0]]

            arr = arr.astype(np.float32)

            if arr.ndim == 3 and self.flatten_sequences:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)

            for i in range(arr.shape[0]):
                yield arr[i]
