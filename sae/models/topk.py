"""TopK SAE: hard top-k sparsity constraint."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Tuple, Optional

from .base import BaseSAE
from .losses import reconstruction_loss, l0_sparsity, topk_auxiliary_loss


class TopKSAE(BaseSAE):
    """Top-K Sparse Autoencoder.

    Only the top-k pre-activations are kept (after ReLU). No L1 penalty needed
    since sparsity is enforced structurally.

    Includes optional auxiliary loss to provide gradient signal to non-selected
    features (prevents dead features).
    """

    def apply_sparsity(
        self, z_pre: jnp.ndarray, *, x_centered: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        k = self.config.k
        batch_size = z_pre.shape[0]

        # Get top-k indices
        topk_values, topk_indices = jax.lax.top_k(z_pre, k)

        # Build sparse output: zero everywhere except top-k positions
        z = jnp.zeros_like(z_pre)
        batch_idx = jnp.arange(batch_size)[:, None]
        z = z.at[batch_idx, topk_indices].set(nn.relu(topk_values))

        return z, {"z_pre": z_pre, "topk_indices": topk_indices}

    def loss(
        self,
        x: jnp.ndarray,
        x_hat: jnp.ndarray,
        z: jnp.ndarray,
        aux: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x_f32 = x.astype(jnp.float32)
        x_hat_f32 = x_hat.astype(jnp.float32)

        mse = reconstruction_loss(x_f32, x_hat_f32)

        # Auxiliary loss for dead feature gradients
        aux_loss = topk_auxiliary_loss(
            z_pre=aux["z_pre"].astype(jnp.float32),
            x=x_f32,
            W_dec=self.W_dec.astype(jnp.float32),
            b_dec=self.b_dec.astype(jnp.float32),
            k=self.config.k,
            coeff=self.config.topk_aux_coeff,
        )

        total = mse + aux_loss

        return total, {
            "total": total,
            "mse": mse,
            "aux_loss": aux_loss,
            "l0": l0_sparsity(z.astype(jnp.float32)),
        }
