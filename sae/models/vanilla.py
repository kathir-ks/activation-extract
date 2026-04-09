"""Vanilla SAE: ReLU activation + L1 sparsity penalty."""

import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Tuple, Optional

from .base import BaseSAE
from .losses import reconstruction_loss, l1_sparsity, l0_sparsity


class VanillaSAE(BaseSAE):
    """Standard Sparse Autoencoder with ReLU and L1 penalty.

    Loss = MSE(x, x_hat) + l1_coeff * mean(sum(|z|))
    """

    def apply_sparsity(
        self, z_pre: jnp.ndarray, *, x_centered: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        z = nn.relu(z_pre)
        return z, {}

    def loss(
        self,
        x: jnp.ndarray,
        x_hat: jnp.ndarray,
        z: jnp.ndarray,
        aux: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # Upcast to float32 for stable loss computation
        x_f32 = x.astype(jnp.float32)
        x_hat_f32 = x_hat.astype(jnp.float32)
        z_f32 = z.astype(jnp.float32)

        mse = reconstruction_loss(x_f32, x_hat_f32)
        l1 = l1_sparsity(z_f32, self.config.l1_coeff)
        total = mse + l1

        return total, {
            "total": total,
            "mse": mse,
            "l1": l1,
            "l0": l0_sparsity(z_f32),
        }
