"""Load activations from HuggingFace datasets."""

import numpy as np
from typing import Iterator, Optional

from .base import ActivationSource


class HFDatasetSource(ActivationSource):
    """Load activations from a HuggingFace dataset.

    Expects a dataset with a column containing activation arrays.

    Args:
        dataset_name: HuggingFace dataset name or local path.
        column: Column name containing activation arrays.
        split: Dataset split (e.g. "train").
        streaming: Use streaming mode for large datasets.
        flatten_sequences: If True, flatten [seq_len, dim] -> individual vectors.
    """

    def __init__(
        self,
        dataset_name: str,
        column: str = "activations",
        split: str = "train",
        streaming: bool = True,
        flatten_sequences: bool = True,
    ):
        self.dataset_name = dataset_name
        self.column = column
        self.split = split
        self.streaming = streaming
        self.flatten_sequences = flatten_sequences
        self._hidden_dim = None

    def _load_dataset(self):
        from datasets import load_dataset

        return load_dataset(
            self.dataset_name, split=self.split, streaming=self.streaming
        )

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            ds = self._load_dataset()
            sample = next(iter(ds))
            arr = np.array(sample[self.column], dtype=np.float32)
            self._hidden_dim = arr.shape[-1]
        return self._hidden_dim

    def iter_vectors(self) -> Iterator[np.ndarray]:
        ds = self._load_dataset()
        for sample in ds:
            arr = np.array(sample[self.column], dtype=np.float32)

            if arr.ndim == 2 and self.flatten_sequences:
                # [seq_len, hidden_dim] -> individual vectors
                for i in range(arr.shape[0]):
                    yield arr[i]
            elif arr.ndim == 1:
                yield arr
            elif arr.ndim == 3 and self.flatten_sequences:
                flat = arr.reshape(-1, arr.shape[-1])
                for i in range(flat.shape[0]):
                    yield flat[i]
            else:
                yield arr.reshape(-1)
