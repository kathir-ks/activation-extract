"""Gated SAE: separate gating network determines active features."""

import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Tuple, Optional

from .base import BaseSAE
from .losses import reconstruction_loss, l1_sparsity, l0_sparsity, heaviside_ste


class GatedSAE(BaseSAE):
    """Gated Sparse Autoencoder (DeepMind, 2024).

    Uses a separate gating pathway to determine which features are active,
    and a magnitude pathway to determine their values:
        gate = Heaviside(W_gate @ x_centered + b_gate)
        magnitude = ReLU(z_pre) * exp(r_mag)
        z = gate * magnitude

    Loss = MSE + l1_coeff * mean(sum(|gate_pre|))
    """

    def setup(self):
        super().setup()
        dtype = self._get_dtype()
        h = self.config.hidden_dim
        d = self.config.dict_size

        # Gating pathway weights
        self.W_gate = self.param(
            "W_gate", nn.initializers.kaiming_normal(), (h, d), dtype
        )
        self.b_gate = self.param("b_gate", nn.initializers.zeros, (d,), dtype)

        # Magnitude rescaling (log-space for stability)
        self.r_mag = self.param("r_mag", nn.initializers.zeros, (d,), dtype)

    def apply_sparsity(
        self, z_pre: jnp.ndarray, *, x_centered: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        assert x_centered is not None, "GatedSAE requires x_centered"

        # Gating pathway
        gate_pre = x_centered @ self.W_gate + self.b_gate
        gate = heaviside_ste(gate_pre, self.config.gated_bandwidth)

        # Magnitude pathway
        magnitude = nn.relu(z_pre) * jnp.exp(self.r_mag)

        z = gate * magnitude
        return z, {"gate_pre": gate_pre, "gate": gate}

    def loss(
        self,
        x: jnp.ndarray,
        x_hat: jnp.ndarray,
        z: jnp.ndarray,
        aux: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x_f32 = x.astype(jnp.float32)
        x_hat_f32 = x_hat.astype(jnp.float32)
        gate_pre_f32 = aux["gate_pre"].astype(jnp.float32)

        mse = reconstruction_loss(x_f32, x_hat_f32)
        # Sparsity on the gating pre-activations
        gate_l1 = l1_sparsity(nn.relu(gate_pre_f32), self.config.l1_coeff)
        total = mse + gate_l1

        return total, {
            "total": total,
            "mse": mse,
            "gate_l1": gate_l1,
            "l0": l0_sparsity(z.astype(jnp.float32)),
        }
