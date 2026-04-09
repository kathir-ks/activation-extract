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

    In multi-host mode, each host loads a different subset of shards
    and yields per-host batches (global_batch_size // num_hosts).

    Args:
        source: Pre-created ActivationSource, or None to create from config.
        config: TrainingConfig (used for buffer size, batch size, source creation).
        host_id: This host's index (0-based). Used for per-host data sharding.
        num_hosts: Total number of hosts in the training job.
    """

    def __init__(
        self,
        config: TrainingConfig,
        source: ActivationSource = None,
        host_id: int = 0,
        num_hosts: int = 1,
    ):
        self.config = config
        self.host_id = host_id
        self.num_hosts = num_hosts

        if source is None:
            # Inject host info into source kwargs for per-host shard claiming
            kwargs = dict(config.source_kwargs)
            if num_hosts > 1:
                kwargs["host_id"] = host_id
                kwargs["num_hosts"] = num_hosts
            source = create_source(config.source_type, **kwargs)

        self.source = source
        self.hidden_dim = source.hidden_dim

    def iter_batches(self) -> Iterator[jnp.ndarray]:
        """Yield shuffled batches of [per_host_batch_size, hidden_dim] JAX arrays.

        In multi-host mode, yields global_batch_size // num_hosts per batch.
        The trainer assembles these into a globally-sharded array.
        """
        per_host_batch = self.config.batch_size // max(self.num_hosts, 1)

        # Loop indefinitely — training loop has its own num_steps limit.
        # This handles epoch boundaries by restarting the source.
        epoch = 0
        while True:
            buf = ShuffleBuffer(
                self.source,
                buffer_size=self.config.shuffle_buffer_size,
                seed=self.config.seed + self.host_id + epoch * 1000,
            )
            exhausted = False
            for batch in buf.iter_batches(per_host_batch):
                yield batch
                exhausted = True
            if not exhausted:
                # Source had no data at all
                break
            epoch += 1

    def iter_batches_unbuffered(self) -> Iterator[jnp.ndarray]:
        """Yield batches directly from source (no shuffling). For eval."""
        for batch_np in self.source.iter_batches(self.config.eval_batch_size):
            yield jnp.array(batch_np)
