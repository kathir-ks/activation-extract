"""JumpReLU SAE: learnable per-feature thresholds."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Tuple, Optional

from .base import BaseSAE
from .losses import reconstruction_loss, jumprelu, heaviside_ste


class JumpReLUSAE(BaseSAE):
    """JumpReLU Sparse Autoencoder (Anthropic, 2024).

    Uses a learnable per-feature threshold:
        z = z_pre * (z_pre > threshold)

    The threshold is parameterized in log-space for stability.
    Gradients through the step function use a straight-through estimator.

    Loss = MSE + l1_coeff * L0(z)  (using STE for L0 gradient)
    """

    def setup(self):
        super().setup()
        dtype = self._get_dtype()
        d = self.config.dict_size

        # Learnable threshold in log-space
        init_val = jnp.log(jnp.array(self.config.jumprelu_init_threshold, dtype=dtype))
        self.log_threshold = self.param(
            "log_threshold",
            lambda key, shape, dtype=dtype: jnp.full(shape, init_val, dtype=dtype),
            (d,),
            dtype,
        )

    def apply_sparsity(
        self, z_pre: jnp.ndarray, *, x_centered: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        threshold = jnp.exp(self.log_threshold)
        bandwidth = self.config.jumprelu_bandwidth
        z = jumprelu(z_pre, threshold, bandwidth)
        return z, {"threshold": threshold, "z_pre": z_pre, "bandwidth": bandwidth}

    def loss(
        self,
        x: jnp.ndarray,
        x_hat: jnp.ndarray,
        z: jnp.ndarray,
        aux: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x_f32 = x.astype(jnp.float32)
        x_hat_f32 = x_hat.astype(jnp.float32)
        threshold = aux["threshold"]
        z_pre = aux["z_pre"]
        bandwidth = aux["bandwidth"]

        mse = reconstruction_loss(x_f32, x_hat_f32)

        # L0 with STE: use heaviside_ste so threshold gets gradient signal
        active_ste = heaviside_ste(z_pre - threshold, bandwidth)
        l0 = jnp.mean(jnp.sum(active_ste, axis=-1))
        l0_penalty = self.config.l1_coeff * l0

        total = mse + l0_penalty

        return total, {
            "total": total,
            "mse": mse,
            "l0_penalty": l0_penalty,
            "l0": l0,
            "mean_threshold": jnp.mean(aux["threshold"]),
        }
