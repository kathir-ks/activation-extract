"""Shuffle buffer for decorrelating streaming activations."""

import numpy as np
import jax.numpy as jnp
from typing import Iterator

from .base import ActivationSource


class ShuffleBuffer:
    """In-memory shuffle buffer for streaming activation data.

    Accumulates vectors from a source into a fixed-size buffer, shuffles them,
    and yields batches. This breaks temporal correlations in sequential data
    (tokens from the same document end up separated).

    Memory usage: buffer_size * hidden_dim * 4 bytes (float32).
    Example: 256k * 896 * 4 = ~915 MB

    Args:
        source: ActivationSource to read from.
        buffer_size: Number of vectors to buffer before shuffling.
        seed: Random seed for shuffling.
    """

    def __init__(self, source: ActivationSource, buffer_size: int, seed: int = 42):
        self.source = source
        self.buffer_size = buffer_size
        self.seed = seed
        self.hidden_dim = source.hidden_dim

    def iter_batches(self, batch_size: int) -> Iterator[jnp.ndarray]:
        """Yield shuffled batches as JAX arrays.

        Fills the buffer from the source, shuffles, yields all complete
        batches, then refills. Continues until the source is exhausted.
        """
        rng = np.random.default_rng(self.seed)
        buffer = np.empty((self.buffer_size, self.hidden_dim), dtype=np.float32)
        buf_idx = 0
        source_iter = self.source.iter_vectors()
        source_exhausted = False

        while True:
            # Fill buffer
            while buf_idx < self.buffer_size:
                try:
                    vec = next(source_iter)
                    buffer[buf_idx] = vec
                    buf_idx += 1
                except StopIteration:
                    source_exhausted = True
                    break

            if buf_idx == 0:
                break

            # Shuffle the filled portion
            filled = buffer[:buf_idx]
            rng.shuffle(filled)

            # Yield complete batches
            for start in range(0, buf_idx - batch_size + 1, batch_size):
                yield jnp.array(filled[start : start + batch_size])

            buf_idx = 0

            if source_exhausted:
                break


class MultiEpochBuffer:
    """Wraps a source with shuffle buffer, repeating for multiple epochs.

    Args:
        source_factory: Callable that creates a fresh ActivationSource each epoch.
        buffer_size: Shuffle buffer size.
        seed: Base random seed (incremented per epoch).
    """

    def __init__(self, source_factory, buffer_size: int, seed: int = 42):
        self.source_factory = source_factory
        self.buffer_size = buffer_size
        self.seed = seed

    def iter_batches(self, batch_size: int, num_epochs: int = 1) -> Iterator[jnp.ndarray]:
        for epoch in range(num_epochs):
            source = self.source_factory()
            buf = ShuffleBuffer(source, self.buffer_size, seed=self.seed + epoch)
            yield from buf.iter_batches(batch_size)
