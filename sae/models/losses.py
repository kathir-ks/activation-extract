"""Loss functions and activation utilities for SAE training."""

import jax
import jax.numpy as jnp


def normalized_mse(x: jnp.ndarray, x_hat: jnp.ndarray) -> jnp.ndarray:
    """MSE normalized by input variance."""
    mse = jnp.mean((x - x_hat) ** 2)
    variance = jnp.var(x)
    return mse / jnp.maximum(variance, 1e-8)


def reconstruction_loss(x: jnp.ndarray, x_hat: jnp.ndarray) -> jnp.ndarray:
    """Mean squared reconstruction error."""
    return jnp.mean((x - x_hat) ** 2)


def l1_sparsity(z: jnp.ndarray, coeff: float) -> jnp.ndarray:
    """L1 sparsity penalty: coeff * mean(sum(|z|, axis=-1))."""
    return coeff * jnp.mean(jnp.sum(jnp.abs(z), axis=-1))


def l0_sparsity(z: jnp.ndarray) -> jnp.ndarray:
    """L0 sparsity: average number of non-zero features per input."""
    return jnp.mean(jnp.sum(z != 0, axis=-1))


def explained_variance(x: jnp.ndarray, x_hat: jnp.ndarray) -> jnp.ndarray:
    """Fraction of input variance explained by reconstruction."""
    return 1.0 - jnp.var(x - x_hat) / jnp.maximum(jnp.var(x), 1e-8)


# -- JumpReLU with straight-through estimator --

@jax.custom_jvp
def jumprelu(z_pre: jnp.ndarray, threshold: jnp.ndarray, bandwidth: float = 0.001) -> jnp.ndarray:
    """JumpReLU: z * (z > threshold). Hard threshold in forward pass."""
    return jnp.where(z_pre > threshold, z_pre, 0.0)


@jumprelu.defjvp
def jumprelu_jvp(primals, tangents):
    """STE for JumpReLU with proper threshold gradients.

    Forward: z * (z > threshold)
    Backward w.r.t. z_pre: pass through where active (z > threshold)
    Backward w.r.t. threshold: -delta_approx * z_pre, where delta_approx
      is a rectangular kernel of width `bandwidth` centered at the threshold.
      This gives the threshold gradient signal to move the boundary.
    """
    z_pre, threshold, bandwidth = primals
    dz, dthreshold, _ = tangents
    active = (z_pre > threshold).astype(z_pre.dtype)
    primal_out = jumprelu(z_pre, threshold, bandwidth)
    # Rectangular kernel approximation of Dirac delta at the boundary
    delta_approx = jnp.where(
        jnp.abs(z_pre - threshold) < bandwidth / 2,
        1.0 / bandwidth,
        0.0,
    )
    # dL/d(threshold) via chain rule: -delta * z_pre * dthreshold
    tangent_out = active * dz - delta_approx * z_pre * dthreshold
    return primal_out, tangent_out


@jax.custom_jvp
def heaviside_ste(x: jnp.ndarray, bandwidth: float = 0.001) -> jnp.ndarray:
    """Heaviside step function with straight-through estimator for gradients."""
    return (x > 0).astype(x.dtype)


@heaviside_ste.defjvp
def heaviside_ste_jvp(primals, tangents):
    """STE using rectangular kernel of given bandwidth."""
    x, bandwidth = primals
    dx, _ = tangents
    primal_out = heaviside_ste(x, bandwidth)
    # Rectangular kernel: gradient is 1/bandwidth where |x| < bandwidth/2
    kernel = jnp.where(jnp.abs(x) < bandwidth / 2, 1.0 / bandwidth, 0.0)
    tangent_out = kernel * dx
    return primal_out, tangent_out


def topk_auxiliary_loss(
    z_pre: jnp.ndarray,
    x: jnp.ndarray,
    W_dec: jnp.ndarray,
    b_dec: jnp.ndarray,
    k: int,
    coeff: float = 1.0 / 32,
) -> jnp.ndarray:
    """Auxiliary loss for TopK SAE to provide gradients to non-selected features.

    Uses a second reconstruction through all ReLU'd activations (not just top-k)
    to give gradient signal to dead/inactive features.
    """
    z_aux = jax.nn.relu(z_pre)
    x_hat_aux = z_aux @ W_dec + b_dec
    return coeff * jnp.mean((x - x_hat_aux) ** 2)
