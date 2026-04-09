"""SAE evaluation metrics."""

import jax.numpy as jnp
from typing import Dict


def compute_metrics(
    x: jnp.ndarray,
    x_hat: jnp.ndarray,
    z: jnp.ndarray,
) -> Dict[str, float]:
    """Compute standard SAE evaluation metrics.

    Args:
        x: [batch, hidden_dim] - input activations
        x_hat: [batch, hidden_dim] - reconstructions
        z: [batch, dict_size] - sparse latent codes

    Returns:
        Dict of metric_name -> scalar value.
    """
    x = x.astype(jnp.float32)
    x_hat = x_hat.astype(jnp.float32)
    z = z.astype(jnp.float32)

    mse = jnp.mean((x - x_hat) ** 2)
    variance = jnp.var(x)
    explained_var = 1.0 - jnp.var(x - x_hat) / jnp.maximum(variance, 1e-8)
    normalized_mse = mse / jnp.maximum(variance, 1e-8)

    # Sparsity
    active = z != 0
    l0 = jnp.mean(jnp.sum(active, axis=-1))  # mean active features per input
    l0_frac = l0 / z.shape[-1]  # as fraction of total features

    # Dead neurons: features that never activate in this batch
    ever_active = jnp.any(active, axis=0)  # [dict_size]
    dead_frac = 1.0 - jnp.mean(ever_active)

    # Feature density: how often each feature fires
    feature_density = jnp.mean(active.astype(jnp.float32), axis=0)  # [dict_size]

    return {
        "mse": float(mse),
        "explained_variance": float(explained_var),
        "normalized_mse": float(normalized_mse),
        "l0": float(l0),
        "l0_frac": float(l0_frac),
        "dead_neuron_frac": float(dead_frac),
        "mean_feature_density": float(jnp.mean(feature_density)),
        "max_feature_density": float(jnp.max(feature_density)),
        "min_feature_density": float(jnp.min(feature_density)),
    }


def compute_dead_neurons(
    dead_neuron_steps: jnp.ndarray, window: int
) -> Dict[str, float]:
    """Compute dead neuron statistics from tracking array.

    Args:
        dead_neuron_steps: [dict_size] steps since last activation.
        window: Number of steps to consider "dead".
    """
    dead = dead_neuron_steps >= window
    return {
        "dead_count": int(jnp.sum(dead)),
        "dead_frac": float(jnp.mean(dead)),
        "max_inactive_steps": int(jnp.max(dead_neuron_steps)),
    }
