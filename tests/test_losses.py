"""Tests for loss functions and STE custom JVPs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sae.models.losses import (
    normalized_mse,
    reconstruction_loss,
    l1_sparsity,
    l0_sparsity,
    explained_variance,
    jumprelu,
    heaviside_ste,
    topk_auxiliary_loss,
)


class TestReconstructionLoss:
    def test_zero_error(self):
        x = jnp.ones((4, 8))
        assert float(reconstruction_loss(x, x)) == 0.0

    def test_known_value(self):
        x = jnp.zeros((2, 2))
        x_hat = jnp.ones((2, 2))
        # MSE = mean of 1^2 = 1.0
        np.testing.assert_allclose(float(reconstruction_loss(x, x_hat)), 1.0)

    def test_positive(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 32))
        x_hat = x + 0.1
        assert float(reconstruction_loss(x, x_hat)) > 0


class TestNormalizedMSE:
    def test_perfect_reconstruction(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert float(normalized_mse(x, x)) == 0.0

    def test_unit_variance(self):
        """When input has unit variance, normalized MSE equals raw MSE."""
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1000, 8))
        x_hat = x + 0.1 * jax.random.normal(jax.random.PRNGKey(1), x.shape)
        nmse = float(normalized_mse(x, x_hat))
        mse = float(reconstruction_loss(x, x_hat))
        # For large N, var(x) ≈ 1, so nmse ≈ mse
        np.testing.assert_allclose(nmse, mse, rtol=0.1)


class TestL1Sparsity:
    def test_zero_activations(self):
        z = jnp.zeros((4, 8))
        assert float(l1_sparsity(z, 1.0)) == 0.0

    def test_scales_with_coeff(self):
        z = jnp.ones((4, 8))
        val1 = float(l1_sparsity(z, 1.0))
        val2 = float(l1_sparsity(z, 2.0))
        np.testing.assert_allclose(val2, 2.0 * val1, rtol=1e-5)

    def test_known_value(self):
        z = jnp.ones((1, 4))  # sum |z| = 4, mean = 4
        np.testing.assert_allclose(float(l1_sparsity(z, 0.5)), 2.0, rtol=1e-5)


class TestL0Sparsity:
    def test_all_zero(self):
        z = jnp.zeros((4, 8))
        assert float(l0_sparsity(z)) == 0.0

    def test_all_nonzero(self):
        z = jnp.ones((4, 8))
        np.testing.assert_allclose(float(l0_sparsity(z)), 8.0)

    def test_partial(self):
        z = jnp.array([[1, 0, 1, 0], [0, 0, 1, 0]])
        # Row 1: 2 nonzero, Row 2: 1 nonzero -> mean = 1.5
        np.testing.assert_allclose(float(l0_sparsity(z)), 1.5)


class TestExplainedVariance:
    def test_perfect(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(float(explained_variance(x, x)), 1.0)

    def test_bad_reconstruction(self):
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (100, 8))
        x_hat = jax.random.normal(jax.random.PRNGKey(1), (100, 8))
        ev = float(explained_variance(x, x_hat))
        assert ev < 0.5, "Random reconstruction should have low explained variance"


class TestJumpReLU:
    def test_forward_thresholding(self):
        z_pre = jnp.array([0.5, 0.1, -0.3, 0.8])
        threshold = jnp.array([0.3, 0.3, 0.3, 0.3])
        out = jumprelu(z_pre, threshold)
        expected = jnp.array([0.5, 0.0, 0.0, 0.8])
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_ste_z_gradient(self):
        """Gradient w.r.t. z_pre should pass through where active."""
        z_pre = jnp.array([0.5, 0.1, -0.3, 0.8])
        threshold = jnp.array([0.3, 0.3, 0.3, 0.3])

        grad_fn = jax.grad(lambda z: jnp.sum(jumprelu(z, threshold, 0.01)))
        g = grad_fn(z_pre)
        # Active (z > threshold): grad = 1, Inactive: grad = 0
        expected = jnp.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(g, expected, atol=1e-5)

    def test_ste_threshold_gradient(self):
        """Gradient w.r.t. threshold should be non-zero near the boundary."""
        # Place z_pre right at the threshold boundary
        z_pre = jnp.array([0.305, 0.295, 0.1, 0.9])
        threshold = jnp.array([0.3, 0.3, 0.3, 0.3])
        bandwidth = 0.05

        grad_fn = jax.grad(
            lambda t: jnp.sum(jumprelu(z_pre, t, bandwidth)),
            argnums=0,
        )
        g = grad_fn(threshold)
        # Near boundary (|z - thresh| < bw/2): should have non-zero grad
        # z=0.305, thresh=0.3: |0.005| < 0.025 -> non-zero
        assert float(g[0]) != 0.0, "Should get gradient near boundary"
        # z=0.295, thresh=0.3: |0.005| < 0.025 -> non-zero
        assert float(g[1]) != 0.0, "Should get gradient near boundary"
        # z=0.1, thresh=0.3: |0.2| > 0.025 -> zero
        np.testing.assert_allclose(float(g[2]), 0.0, atol=1e-6)


class TestHeavisideSTE:
    def test_forward(self):
        x = jnp.array([-1.0, -0.01, 0.01, 1.0])
        out = heaviside_ste(x)
        expected = jnp.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(out, expected)

    def test_ste_gradient(self):
        """Rectangular kernel: non-zero gradient only near zero."""
        x = jnp.array([-1.0, -0.0001, 0.0001, 1.0])
        bandwidth = 0.001
        grad_fn = jax.grad(lambda x: jnp.sum(heaviside_ste(x, bandwidth)))
        g = grad_fn(x)
        # Only x near 0 (within bandwidth/2) should get gradient
        assert float(g[0]) == 0.0
        assert float(g[1]) != 0.0  # -0.0001 is within 0.0005 of 0
        assert float(g[2]) != 0.0  # 0.0001 is within 0.0005 of 0
        assert float(g[3]) == 0.0


class TestTopKAuxLoss:
    def test_zero_when_perfect(self):
        """Aux loss should be zero when ReLU reconstruction is perfect."""
        z_pre = jnp.ones((4, 8))
        W_dec = jnp.eye(8)
        b_dec = jnp.zeros(8)
        x = jax.nn.relu(z_pre) @ W_dec + b_dec  # perfect reconstruction
        loss = topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k=4, coeff=1.0)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-5)

    def test_positive_when_imperfect(self):
        rng = jax.random.PRNGKey(0)
        z_pre = jax.random.normal(rng, (8, 16))
        W_dec = jax.random.normal(jax.random.PRNGKey(1), (16, 8))
        b_dec = jnp.zeros(8)
        x = jax.random.normal(jax.random.PRNGKey(2), (8, 8))
        loss = topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k=4)
        assert float(loss) > 0

    def test_scales_with_coeff(self):
        rng = jax.random.PRNGKey(0)
        z_pre = jax.random.normal(rng, (8, 16))
        W_dec = jax.random.normal(jax.random.PRNGKey(1), (16, 8))
        b_dec = jnp.zeros(8)
        x = jax.random.normal(jax.random.PRNGKey(2), (8, 8))
        loss1 = float(topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k=4, coeff=1.0))
        loss2 = float(topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k=4, coeff=2.0))
        np.testing.assert_allclose(loss2, 2.0 * loss1, rtol=1e-5)
