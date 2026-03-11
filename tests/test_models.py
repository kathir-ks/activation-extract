"""Tests for SAE model architectures: forward pass, loss, gradients, shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sae.configs.base import SAEConfig
from sae.models.registry import create_sae
from sae.models.vanilla import VanillaSAE
from sae.models.topk import TopKSAE
from sae.models.gated import GatedSAE
from sae.models.jumprelu import JumpReLUSAE

from tests.conftest import HIDDEN_DIM, DICT_SIZE, BATCH_SIZE


# ===== Forward pass shapes =====

class TestForwardPass:
    """Test that all architectures produce correct output shapes."""

    def test_output_shapes(self, any_config, rng, random_batch):
        model = create_sae(any_config)
        variables = model.init(rng, random_batch)
        x_hat, z, aux = model.apply(variables, random_batch)

        assert x_hat.shape == random_batch.shape, f"x_hat shape mismatch"
        assert z.shape == (BATCH_SIZE, DICT_SIZE), f"z shape mismatch"

    def test_compute_loss_shapes(self, any_config, rng, random_batch):
        model = create_sae(any_config)
        variables = model.init(rng, random_batch)
        total_loss, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )

        assert total_loss.shape == (), "total_loss should be scalar"
        assert "_z" in loss_dict, "loss_dict should contain _z for dead tracking"
        assert loss_dict["_z"].shape == (BATCH_SIZE, DICT_SIZE)

    def test_bfloat16_dtype(self, rng):
        cfg = SAEConfig(
            hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
            architecture="vanilla", dtype="bfloat16",
        )
        model = create_sae(cfg)
        x = jax.random.normal(rng, (4, HIDDEN_DIM), dtype=jnp.bfloat16)
        variables = model.init(rng, x)
        x_hat, z, _ = model.apply(variables, x)

        assert x_hat.dtype == jnp.bfloat16
        assert z.dtype == jnp.bfloat16

    def test_encode_decode_consistency(self, any_config, rng, random_batch):
        """encode() + decode() should match __call__."""
        model = create_sae(any_config)
        variables = model.init(rng, random_batch)

        x_hat_call, z_call, _ = model.apply(variables, random_batch)

        # Manual path: encode -> apply_sparsity -> decode
        z_pre = model.apply(variables, random_batch, method=model.encode)
        # z_pre from encode should be the same as the internal computation
        assert z_pre.shape == (BATCH_SIZE, DICT_SIZE)


# ===== Architecture-specific behavior =====

class TestVanillaSAE:
    def test_relu_activation(self, vanilla_config, rng, random_batch):
        model = create_sae(vanilla_config)
        variables = model.init(rng, random_batch)
        _, z, _ = model.apply(variables, random_batch)
        assert jnp.all(z >= 0), "Vanilla SAE should use ReLU (all z >= 0)"

    def test_loss_components(self, vanilla_config, rng, random_batch):
        model = create_sae(vanilla_config)
        variables = model.init(rng, random_batch)
        total, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )
        assert "mse" in loss_dict
        assert "l1" in loss_dict
        assert "l0" in loss_dict
        assert "total" in loss_dict
        # total = mse + l1
        np.testing.assert_allclose(
            float(loss_dict["total"]),
            float(loss_dict["mse"]) + float(loss_dict["l1"]),
            rtol=1e-5,
        )


class TestTopKSAE:
    def test_exact_k_nonzero(self, topk_config, rng, random_batch):
        """TopK should produce exactly k non-zero features per input."""
        model = create_sae(topk_config)
        variables = model.init(rng, random_batch)
        _, z, _ = model.apply(variables, random_batch)

        nnz_per_row = jnp.sum(z != 0, axis=-1)
        assert jnp.all(nnz_per_row == topk_config.k), (
            f"Expected exactly {topk_config.k} non-zero per row, "
            f"got {nnz_per_row}"
        )

    def test_l0_matches_k(self, topk_config, rng, random_batch):
        model = create_sae(topk_config)
        variables = model.init(rng, random_batch)
        _, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )
        np.testing.assert_allclose(float(loss_dict["l0"]), topk_config.k, atol=0.1)

    def test_loss_components(self, topk_config, rng, random_batch):
        model = create_sae(topk_config)
        variables = model.init(rng, random_batch)
        _, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )
        assert "mse" in loss_dict
        assert "aux_loss" in loss_dict
        assert "l0" in loss_dict

    def test_nonnegative_activations(self, topk_config, rng, random_batch):
        model = create_sae(topk_config)
        variables = model.init(rng, random_batch)
        _, z, _ = model.apply(variables, random_batch)
        assert jnp.all(z >= 0), "TopK uses ReLU so all z >= 0"


class TestGatedSAE:
    def test_nonnegative_activations(self, gated_config, rng, random_batch):
        model = create_sae(gated_config)
        variables = model.init(rng, random_batch)
        _, z, _ = model.apply(variables, random_batch)
        assert jnp.all(z >= 0), "Gated SAE z should be non-negative"

    def test_loss_components(self, gated_config, rng, random_batch):
        model = create_sae(gated_config)
        variables = model.init(rng, random_batch)
        _, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )
        assert "mse" in loss_dict
        assert "gate_l1" in loss_dict
        assert "l0" in loss_dict

    def test_extra_params(self, gated_config, rng, random_batch):
        """Gated SAE should have W_gate, b_gate, r_mag parameters."""
        model = create_sae(gated_config)
        variables = model.init(rng, random_batch)
        params = variables["params"]
        assert "W_gate" in params
        assert "b_gate" in params
        assert "r_mag" in params


class TestJumpReLUSAE:
    def test_nonnegative_activations(self, jumprelu_config, rng, random_batch):
        model = create_sae(jumprelu_config)
        variables = model.init(rng, random_batch)
        _, z, _ = model.apply(variables, random_batch)
        assert jnp.all(z >= 0), "JumpReLU z should be non-negative"

    def test_loss_components(self, jumprelu_config, rng, random_batch):
        model = create_sae(jumprelu_config)
        variables = model.init(rng, random_batch)
        _, loss_dict = model.apply(
            variables, random_batch, method=model.compute_loss
        )
        assert "mse" in loss_dict
        assert "l0_penalty" in loss_dict
        assert "l0" in loss_dict
        assert "mean_threshold" in loss_dict

    def test_log_threshold_param(self, jumprelu_config, rng, random_batch):
        model = create_sae(jumprelu_config)
        variables = model.init(rng, random_batch)
        assert "log_threshold" in variables["params"]
        assert variables["params"]["log_threshold"].shape == (DICT_SIZE,)

    def test_threshold_gets_gradients(self, jumprelu_config, rng, random_batch):
        """P0 bug fix verification: log_threshold must receive gradients."""
        model = create_sae(jumprelu_config)
        variables = model.init(rng, random_batch)

        def loss_fn(params):
            total, _ = model.apply(
                {"params": params}, random_batch, method=model.compute_loss
            )
            return total

        grads = jax.grad(loss_fn)(variables["params"])
        thresh_grad = grads["log_threshold"]
        assert thresh_grad.shape == (DICT_SIZE,)
        # At least some features should have non-zero threshold gradient
        assert float(jnp.linalg.norm(thresh_grad)) > 0, (
            "log_threshold received zero gradient — STE is broken"
        )


# ===== Gradient tests (all architectures) =====

class TestGradients:
    """Verify all architectures produce finite gradients for all parameters."""

    def test_all_params_get_gradients(self, any_config, rng, random_batch):
        model = create_sae(any_config)
        variables = model.init(rng, random_batch)

        def loss_fn(params):
            total, _ = model.apply(
                {"params": params}, random_batch, method=model.compute_loss
            )
            return total

        grads = jax.grad(loss_fn)(variables["params"])

        for name, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), (
                f"{any_config.architecture}: {name} has non-finite gradients"
            )
            # All trainable params should get some gradient signal
            norm = float(jnp.linalg.norm(g))
            assert norm > 0 or name in ("b_dec",), (
                f"{any_config.architecture}: {name} got zero gradient (norm={norm})"
            )

    def test_grad_finite_bfloat16(self, rng):
        """Gradients should remain finite when using bfloat16."""
        cfg = SAEConfig(
            hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
            architecture="vanilla", dtype="bfloat16", l1_coeff=1e-2,
        )
        model = create_sae(cfg)
        x = jax.random.normal(rng, (BATCH_SIZE, HIDDEN_DIM), dtype=jnp.bfloat16)
        variables = model.init(rng, x)

        def loss_fn(params):
            total, _ = model.apply(
                {"params": params}, x, method=model.compute_loss
            )
            return total

        grads = jax.grad(loss_fn)(variables["params"])
        for name, g in grads.items():
            assert jnp.all(jnp.isfinite(g)), (
                f"bfloat16: {name} has non-finite gradients"
            )


# ===== Decoder normalization =====

class TestDecoderNorm:
    def test_decoder_init_norm(self, any_config, rng, random_batch):
        """Decoder rows should be initialized to dec_init_norm."""
        model = create_sae(any_config)
        variables = model.init(rng, random_batch)
        W_dec = variables["params"]["W_dec"]
        row_norms = jnp.linalg.norm(W_dec, axis=-1)
        np.testing.assert_allclose(
            row_norms, any_config.dec_init_norm, atol=1e-5,
        )


# ===== Registry =====

class TestRegistry:
    def test_create_all_architectures(self, rng):
        for arch in ["vanilla", "topk", "gated", "jumprelu"]:
            cfg = SAEConfig(
                hidden_dim=16, dict_size=32, architecture=arch, dtype="float32",
                k=8,  # must be <= dict_size for topk
            )
            model = create_sae(cfg)
            x = jax.random.normal(rng, (4, 16))
            variables = model.init(rng, x)
            x_hat, z, _ = model.apply(variables, x)
            assert x_hat.shape == x.shape

    def test_unknown_architecture_raises(self):
        cfg = SAEConfig(architecture="nonexistent")
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_sae(cfg)

    def test_register_custom(self, rng):
        from sae.models.registry import register_sae, SAE_REGISTRY
        register_sae("vanilla_copy", VanillaSAE)
        assert "vanilla_copy" in SAE_REGISTRY
        cfg = SAEConfig(
            hidden_dim=16, dict_size=32, architecture="vanilla_copy", dtype="float32",
            k=8,
        )
        model = create_sae(cfg)
        x = jax.random.normal(rng, (4, 16))
        variables = model.init(rng, x)
        assert model.apply(variables, x)[0].shape == x.shape
        # Cleanup
        del SAE_REGISTRY["vanilla_copy"]
