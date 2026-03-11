"""Activation data pipeline: source -> shuffle -> batch -> device."""

import jax
import jax.numpy as jnp
from typing import Iterator

from ..configs.training import TrainingConfig
from .base import ActivationSource
from .buffer import ShuffleBuffer
from .registry import create_source


class ActivationPipeline:
    """End-to-end data pipeline for SAE training.

    Composes: source -> shuffle buffer -> batch -> JAX array transfer.

    Args:
        source: Pre-created ActivationSource, or None to create from config.
        config: TrainingConfig (used for buffer size, batch size, source creation).
    """

    def __init__(
        self,
        config: TrainingConfig,
        source: ActivationSource = None,
    ):
        self.config = config

        if source is None:
            source = create_source(config.source_type, **config.source_kwargs)

        self.source = source
        self.hidden_dim = source.hidden_dim

    def iter_batches(self) -> Iterator[jnp.ndarray]:
        """Yield shuffled batches of [batch_size, hidden_dim] JAX arrays."""
        buf = ShuffleBuffer(
            self.source,
            buffer_size=self.config.shuffle_buffer_size,
            seed=self.config.seed,
        )
        yield from buf.iter_batches(self.config.batch_size)

    def iter_batches_unbuffered(self) -> Iterator[jnp.ndarray]:
        """Yield batches directly from source (no shuffling). For eval."""
        for batch_np in self.source.iter_batches(self.config.eval_batch_size):
            yield jnp.array(batch_np)
