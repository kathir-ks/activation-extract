"""Tests for evaluation metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sae.evaluation.metrics import compute_metrics, compute_dead_neurons


class TestComputeMetrics:
    def test_perfect_reconstruction(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
        z = jnp.ones((32, 64))  # all features active
        metrics = compute_metrics(x, x, z)

        np.testing.assert_allclose(metrics["mse"], 0.0, atol=1e-6)
        np.testing.assert_allclose(metrics["explained_variance"], 1.0, atol=1e-6)
        np.testing.assert_allclose(metrics["normalized_mse"], 0.0, atol=1e-6)
        np.testing.assert_allclose(metrics["l0"], 64.0)
        np.testing.assert_allclose(metrics["l0_frac"], 1.0)
        np.testing.assert_allclose(metrics["dead_neuron_frac"], 0.0)

    def test_all_dead_neurons(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
        x_hat = jnp.zeros_like(x)
        z = jnp.zeros((32, 64))  # all features dead
        metrics = compute_metrics(x, x_hat, z)

        assert metrics["mse"] > 0
        np.testing.assert_allclose(metrics["l0"], 0.0)
        np.testing.assert_allclose(metrics["dead_neuron_frac"], 1.0)

    def test_partial_sparsity(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 8))
        x_hat = x + 0.1
        # 2 active features per sample
        z = jnp.zeros((4, 16))
        z = z.at[:, 0].set(1.0)
        z = z.at[:, 1].set(0.5)
        metrics = compute_metrics(x, x_hat, z)

        np.testing.assert_allclose(metrics["l0"], 2.0)
        np.testing.assert_allclose(metrics["l0_frac"], 2.0 / 16)
        np.testing.assert_allclose(metrics["dead_neuron_frac"], 14.0 / 16)

    def test_return_types(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
        x_hat = x * 0.9
        z = jnp.ones((8, 8))
        metrics = compute_metrics(x, x_hat, z)

        expected_keys = {
            "mse", "explained_variance", "normalized_mse",
            "l0", "l0_frac", "dead_neuron_frac",
            "mean_feature_density", "max_feature_density", "min_feature_density",
        }
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_bfloat16_input(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 4), dtype=jnp.bfloat16)
        x_hat = x * 0.9
        z = jnp.ones((8, 8), dtype=jnp.bfloat16)
        metrics = compute_metrics(x, x_hat, z)
        # Should not raise and metrics should be finite
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_feature_density(self):
        z = jnp.zeros((10, 4))
        z = z.at[:, 0].set(1.0)       # feature 0: fires always
        z = z.at[:5, 1].set(1.0)      # feature 1: fires 50%
        # feature 2, 3: never fire
        metrics = compute_metrics(
            jnp.ones((10, 2)), jnp.ones((10, 2)), z
        )
        np.testing.assert_allclose(metrics["max_feature_density"], 1.0)
        np.testing.assert_allclose(metrics["min_feature_density"], 0.0)
        np.testing.assert_allclose(
            metrics["mean_feature_density"], (1.0 + 0.5 + 0 + 0) / 4
        )


class TestComputeDeadNeurons:
    def test_no_dead(self):
        steps = jnp.zeros(100, dtype=jnp.int32)
        info = compute_dead_neurons(steps, window=1000)
        assert info["dead_count"] == 0
        np.testing.assert_allclose(info["dead_frac"], 0.0)

    def test_all_dead(self):
        steps = jnp.full(100, 5000, dtype=jnp.int32)
        info = compute_dead_neurons(steps, window=1000)
        assert info["dead_count"] == 100
        np.testing.assert_allclose(info["dead_frac"], 1.0)

    def test_partial_dead(self):
        steps = jnp.array([0, 500, 1000, 2000, 999], dtype=jnp.int32)
        info = compute_dead_neurons(steps, window=1000)
        # Dead: steps >= 1000 -> indices 2, 3 -> 2 dead out of 5
        assert info["dead_count"] == 2
        np.testing.assert_allclose(info["dead_frac"], 0.4)
        assert info["max_inactive_steps"] == 2000

    def test_window_boundary(self):
        steps = jnp.array([999, 1000], dtype=jnp.int32)
        info = compute_dead_neurons(steps, window=1000)
        # 999 < 1000: alive, 1000 >= 1000: dead
        assert info["dead_count"] == 1
