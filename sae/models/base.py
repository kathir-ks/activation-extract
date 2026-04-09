"""Base SAE module. All architectures inherit from this."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Tuple, Optional

from ..configs.base import SAEConfig


class BaseSAE(nn.Module):
    """Base Sparse Autoencoder.

    Subclasses must implement:
        - apply_sparsity(z_pre, *, x_centered=None) -> (z, aux_dict)
        - loss(x, x_hat, z, aux) -> (total_loss, loss_dict)
    """

    config: SAEConfig

    def _get_dtype(self):
        return jnp.bfloat16 if self.config.dtype == "bfloat16" else jnp.float32

    def _dec_init(self, key, shape, dtype=jnp.float32):
        """Initialize decoder with controlled column norm."""
        W = jax.random.normal(key, shape, dtype=dtype)
        # Normalize each row (feature) to dec_init_norm
        norms = jnp.linalg.norm(W, axis=-1, keepdims=True)
        return W / jnp.maximum(norms, 1e-8) * self.config.dec_init_norm

    def setup(self):
        dtype = self._get_dtype()
        h = self.config.hidden_dim
        d = self.config.dict_size

        self.W_enc = self.param(
            "W_enc", nn.initializers.kaiming_normal(), (h, d), dtype
        )
        self.b_enc = self.param("b_enc", nn.initializers.zeros, (d,), dtype)
        self.W_dec = self.param("W_dec", self._dec_init, (d, h), dtype)
        self.b_dec = self.param("b_dec", nn.initializers.zeros, (h,), dtype)

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode: x [batch, hidden_dim] -> z_pre [batch, dict_size]."""
        x_centered = x - self.b_dec
        return x_centered @ self.W_enc + self.b_enc

    def apply_sparsity(
        self, z_pre: jnp.ndarray, *, x_centered: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply sparsity to pre-activations. Override in subclasses."""
        raise NotImplementedError

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode: z [batch, dict_size] -> x_hat [batch, hidden_dim]."""
        return z @ self.W_dec + self.b_dec

    def __call__(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Forward pass.

        Args:
            x: [batch, hidden_dim]

        Returns:
            x_hat: [batch, hidden_dim] - reconstruction
            z: [batch, dict_size] - sparse latent codes
            aux: dict - auxiliary data for loss computation
        """
        x_centered = x - self.b_dec
        z_pre = x_centered @ self.W_enc + self.b_enc
        z, aux = self.apply_sparsity(z_pre, x_centered=x_centered)
        x_hat = self.decode(z)
        return x_hat, z, aux

    def compute_loss(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Forward pass + loss in one call. Used by the trainer via model.apply.

        Returns:
            (total_loss, loss_dict) where loss_dict includes 'x_hat', 'z', and metrics.
        """
        x_hat, z, aux = self(x)
        total_loss, loss_dict = self.loss(x, x_hat, z, aux)
        # Include z in loss_dict for dead neuron tracking
        loss_dict["_z"] = z
        return total_loss, loss_dict

    def loss(
        self,
        x: jnp.ndarray,
        x_hat: jnp.ndarray,
        z: jnp.ndarray,
        aux: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss. Override in subclasses."""
        raise NotImplementedError

