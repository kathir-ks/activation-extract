"""Abstract base class for activation data sources."""

import abc
import numpy as np
from typing import Iterator, Optional


class ActivationSource(abc.ABC):
    """Base class for pluggable activation data sources.

    Subclasses load activations from any format and yield flat vectors
    of shape [hidden_dim]. The pipeline handles batching and shuffling.

    To add a new format:
        1. Subclass ActivationSource
        2. Implement iter_vectors() and hidden_dim
        3. Register with the source registry
    """

    @abc.abstractmethod
    def iter_vectors(self) -> Iterator[np.ndarray]:
        """Yield individual activation vectors, shape [hidden_dim].

        For sequence data (shape [seq_len, hidden_dim]), each token
        position should be yielded as a separate vector.
        """
        ...

    @property
    @abc.abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of activation vectors."""
        ...

    @property
    def total_vectors(self) -> Optional[int]:
        """Total number of vectors, or None if unknown (streaming)."""
        return None

    def iter_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        """Yield batches of shape [batch_size, hidden_dim].

        Default implementation collects from iter_vectors(). Subclasses
        may override for more efficient batched loading.
        """
        batch = []
        for vec in self.iter_vectors():
            batch.append(vec)
            if len(batch) == batch_size:
                yield np.stack(batch)
                batch = []
        # Yield remainder
        if batch:
            yield np.stack(batch)
