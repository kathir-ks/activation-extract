"""SAE training state."""

import jax.numpy as jnp
from flax.training import train_state


class SAETrainState(train_state.TrainState):
    """Extended train state for SAE training.

    Adds tracking for dead neurons and total tokens processed.
    """

    # Steps since each neuron last activated. Shape: [dict_size]
    dead_neuron_steps: jnp.ndarray = None

    # Total activation vectors processed
    total_tokens: int = 0
