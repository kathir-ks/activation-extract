"""Tests for the training loop, checkpointing, LR schedules, and distributed."""

import tempfile
import shutil
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from sae.configs.base import SAEConfig
from sae.configs.training import TrainingConfig
from sae.data.numpy_source import NumpySource
from sae.models.registry import create_sae
from sae.training.checkpointing import (
    save_checkpoint, load_checkpoint, restore_params, restore_opt_state,
)
from sae.training.lr_schedule import create_lr_schedule, create_optimizer
from sae.training.train_state import SAETrainState
from sae.training.trainer import SAETrainer
from sae.training.distributed import create_sae_mesh, shard_batch, replicate_params

from tests.conftest import HIDDEN_DIM, DICT_SIZE, BATCH_SIZE


# ===== Checkpointing =====

class TestCheckpointing:
    """Test save/load/restore round-trip including optimizer state."""

    @pytest.fixture
    def state_and_model(self, vanilla_config, rng):
        model = create_sae(vanilla_config)
        x = jax.random.normal(rng, (BATCH_SIZE, HIDDEN_DIM))
        variables = model.init(rng, x)
        optimizer = optax.adam(1e-3)
        state = SAETrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            dead_neuron_steps=jnp.zeros(DICT_SIZE, dtype=jnp.int32),
            total_tokens=0,
        )
        # Run a few steps to populate optimizer state
        for _ in range(3):
            def loss_fn(params):
                total, _ = model.apply(
                    {"params": params}, x, method=model.compute_loss
                )
                return total
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
        return state, model, vanilla_config

    def test_save_and_load(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        save_checkpoint(
            tmp_dir, step=3, params=state.params, opt_state=state.opt_state,
            dead_neuron_steps=state.dead_neuron_steps, total_tokens=24,
            sae_config=cfg, training_config=tcfg,
        )

        ckpt = load_checkpoint(tmp_dir)
        assert ckpt is not None
        assert ckpt["metadata"]["step"] == 3
        assert ckpt["metadata"]["total_tokens"] == 24
        assert len(ckpt["params_flat"]) > 0
        assert ckpt["opt_state_flat"] is not None

    def test_params_round_trip(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        save_checkpoint(
            tmp_dir, step=1, params=state.params, opt_state=state.opt_state,
            dead_neuron_steps=state.dead_neuron_steps, total_tokens=0,
            sae_config=cfg, training_config=tcfg,
        )
        ckpt = load_checkpoint(tmp_dir)
        restored = restore_params(ckpt, state.params)

        for key in state.params:
            np.testing.assert_allclose(
                np.array(state.params[key]),
                np.array(restored[key]),
                atol=1e-6,
                err_msg=f"Param {key} mismatch after round-trip",
            )

    def test_opt_state_round_trip(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        save_checkpoint(
            tmp_dir, step=1, params=state.params, opt_state=state.opt_state,
            dead_neuron_steps=state.dead_neuron_steps, total_tokens=0,
            sae_config=cfg, training_config=tcfg,
        )
        ckpt = load_checkpoint(tmp_dir)
        restored_opt = restore_opt_state(ckpt, state.opt_state)

        assert restored_opt is not None
        orig_flat = jax.tree.leaves(state.opt_state)
        rest_flat = jax.tree.leaves(restored_opt)
        assert len(orig_flat) == len(rest_flat)
        for o, r in zip(orig_flat, rest_flat):
            np.testing.assert_allclose(np.array(o), np.array(r), atol=1e-6)

    def test_dead_neuron_steps_round_trip(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        state = state.replace(
            dead_neuron_steps=jnp.arange(DICT_SIZE, dtype=jnp.int32)
        )
        tcfg = TrainingConfig()

        save_checkpoint(
            tmp_dir, step=1, params=state.params, opt_state=state.opt_state,
            dead_neuron_steps=state.dead_neuron_steps, total_tokens=0,
            sae_config=cfg, training_config=tcfg,
        )
        ckpt = load_checkpoint(tmp_dir)
        np.testing.assert_array_equal(
            ckpt["dead_neuron_steps"],
            np.arange(DICT_SIZE, dtype=np.int32),
        )

    def test_load_latest(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        for step in [10, 20, 30]:
            save_checkpoint(
                tmp_dir, step=step, params=state.params,
                opt_state=state.opt_state,
                dead_neuron_steps=state.dead_neuron_steps,
                total_tokens=step * BATCH_SIZE, sae_config=cfg,
                training_config=tcfg, keep_last_n=5,
            )

        ckpt = load_checkpoint(tmp_dir)
        assert ckpt["metadata"]["step"] == 30

    def test_load_specific_step(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        for step in [10, 20, 30]:
            save_checkpoint(
                tmp_dir, step=step, params=state.params,
                opt_state=state.opt_state,
                dead_neuron_steps=state.dead_neuron_steps,
                total_tokens=0, sae_config=cfg, training_config=tcfg,
                keep_last_n=5,
            )

        ckpt = load_checkpoint(tmp_dir, step=20)
        assert ckpt["metadata"]["step"] == 20

    def test_cleanup_old_checkpoints(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig()

        for step in range(1, 6):
            save_checkpoint(
                tmp_dir, step=step, params=state.params,
                opt_state=state.opt_state,
                dead_neuron_steps=state.dead_neuron_steps,
                total_tokens=0, sae_config=cfg, training_config=tcfg,
                keep_last_n=2,
            )

        from pathlib import Path
        remaining = sorted(Path(tmp_dir).glob("step_*"))
        assert len(remaining) == 2
        # Should keep the latest two
        assert remaining[-1].name == "step_00000005"
        assert remaining[-2].name == "step_00000004"

    def test_load_nonexistent_returns_none(self, tmp_dir):
        ckpt = load_checkpoint(f"{tmp_dir}/nonexistent")
        assert ckpt is None

    def test_metadata_contains_configs(self, state_and_model, tmp_dir):
        state, model, cfg = state_and_model
        tcfg = TrainingConfig(batch_size=256, num_steps=50000)

        save_checkpoint(
            tmp_dir, step=1, params=state.params, opt_state=state.opt_state,
            dead_neuron_steps=state.dead_neuron_steps, total_tokens=0,
            sae_config=cfg, training_config=tcfg,
        )
        ckpt = load_checkpoint(tmp_dir)
        meta = ckpt["metadata"]
        assert "sae_config" in meta
        assert meta["sae_config"]["hidden_dim"] == HIDDEN_DIM
        assert meta["sae_config"]["architecture"] == "vanilla"
        assert "training_config" in meta
        assert meta["training_config"]["batch_size"] == 256


# ===== LR Schedules =====

class TestLRSchedule:
    def test_cosine_schedule(self):
        cfg = TrainingConfig(
            learning_rate=1e-3, lr_warmup_steps=100,
            num_steps=1000, lr_decay="cosine",
        )
        schedule = create_lr_schedule(cfg)
        # At step 0: should be near 0 (warmup start)
        assert float(schedule(0)) < 1e-5
        # At peak (step 100): should be ~1e-3
        np.testing.assert_allclose(float(schedule(100)), 1e-3, rtol=0.01)
        # At end: should decay to ~1e-4
        end_lr = float(schedule(1000))
        assert end_lr < 1e-3

    def test_constant_schedule(self):
        cfg = TrainingConfig(
            learning_rate=5e-4, lr_warmup_steps=10,
            num_steps=100, lr_decay="constant",
        )
        schedule = create_lr_schedule(cfg)
        assert float(schedule(0)) < 5e-4  # warmup
        np.testing.assert_allclose(float(schedule(50)), 5e-4, rtol=0.01)
        np.testing.assert_allclose(float(schedule(99)), 5e-4, rtol=0.01)

    def test_constant_no_warmup(self):
        cfg = TrainingConfig(
            learning_rate=1e-3, lr_warmup_steps=0,
            num_steps=100, lr_decay="constant",
        )
        schedule = create_lr_schedule(cfg)
        np.testing.assert_allclose(float(schedule(0)), 1e-3, rtol=0.01)

    def test_linear_schedule(self):
        cfg = TrainingConfig(
            learning_rate=1e-3, lr_warmup_steps=10,
            num_steps=100, lr_decay="linear",
        )
        schedule = create_lr_schedule(cfg)
        assert float(schedule(0)) < 1e-4
        np.testing.assert_allclose(float(schedule(10)), 1e-3, rtol=0.01)
        end_lr = float(schedule(99))
        assert end_lr < 1e-3  # should decay toward 0

    def test_unknown_schedule_raises(self):
        cfg = TrainingConfig(lr_decay="exponential")
        with pytest.raises(ValueError, match="Unknown lr_decay"):
            create_lr_schedule(cfg)


class TestCreateOptimizer:
    def test_adam(self):
        cfg = TrainingConfig(optimizer="adam", num_steps=100, lr_warmup_steps=10)
        opt = create_optimizer(cfg)
        assert opt is not None

    def test_adamw(self):
        cfg = TrainingConfig(
            optimizer="adamw", weight_decay=0.01,
            num_steps=100, lr_warmup_steps=10,
        )
        opt = create_optimizer(cfg)
        assert opt is not None

    def test_unknown_optimizer_raises(self):
        cfg = TrainingConfig(optimizer="sgd", num_steps=100, lr_warmup_steps=10)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(cfg)


# ===== Train State =====

class TestTrainState:
    def test_create(self, vanilla_config, rng, random_batch):
        model = create_sae(vanilla_config)
        variables = model.init(rng, random_batch)
        optimizer = optax.adam(1e-3)

        state = SAETrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            dead_neuron_steps=jnp.zeros(DICT_SIZE, dtype=jnp.int32),
            total_tokens=0,
        )
        assert state.step == 0
        assert state.total_tokens == 0
        assert state.dead_neuron_steps.shape == (DICT_SIZE,)

    def test_apply_gradients(self, vanilla_config, rng, random_batch):
        model = create_sae(vanilla_config)
        variables = model.init(rng, random_batch)
        optimizer = optax.adam(1e-3)

        state = SAETrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
        )

        def loss_fn(params):
            total, _ = model.apply(
                {"params": params}, random_batch, method=model.compute_loss
            )
            return total

        grads = jax.grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        assert int(new_state.step) == 1
        # Params should have changed
        for key in state.params:
            assert not np.array_equal(
                np.array(state.params[key]),
                np.array(new_state.params[key]),
            ), f"Param {key} didn't change after gradient step"


# ===== Distributed =====

class TestDistributed:
    def test_create_mesh(self):
        mesh = create_sae_mesh("data_parallel")
        assert "data" in mesh.shape
        assert mesh.shape["data"] == jax.device_count()

    def test_shard_batch(self):
        mesh = create_sae_mesh("data_parallel")
        n_devices = jax.device_count()
        batch = jnp.ones((n_devices * 4, HIDDEN_DIM))
        with mesh:
            sharded = shard_batch(batch, mesh)
        assert sharded.shape == batch.shape

    def test_replicate_params(self, vanilla_config, rng, random_batch):
        model = create_sae(vanilla_config)
        variables = model.init(rng, random_batch)
        mesh = create_sae_mesh("data_parallel")
        with mesh:
            replicated = replicate_params(variables["params"], mesh)
        # Should have same structure
        assert set(replicated.keys()) == set(variables["params"].keys())


# ===== Trainer Integration =====

class TestTrainer:
    """Integration tests for the full training loop."""

    def _make_trainer(self, sae_config, train_config, numpy_data_path):
        source = NumpySource(path=numpy_data_path)
        trainer = SAETrainer(sae_config, train_config, source=source)
        return trainer

    def test_setup(self, vanilla_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            vanilla_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        assert trainer.state is not None
        assert trainer.model is not None
        assert trainer.mesh is not None
        assert trainer.pipeline is not None

    def test_train_vanilla(self, vanilla_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            vanilla_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()
        # Should have trained for some steps
        assert int(trainer.state.step) > 0

    def test_train_topk(self, topk_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            topk_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()
        assert int(trainer.state.step) > 0

    def test_train_gated(self, gated_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            gated_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()
        assert int(trainer.state.step) > 0

    def test_train_jumprelu(self, jumprelu_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            jumprelu_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()
        assert int(trainer.state.step) > 0

    def test_loss_decreases(self, vanilla_config, tmp_dir):
        """Loss should decrease over training steps on a fixed eval batch."""
        # Create enough data for the full run
        data = np.random.randn(5000, HIDDEN_DIM).astype(np.float32)
        data_path = f"{tmp_dir}/big.npy"
        np.save(data_path, data)

        cfg = TrainingConfig(
            batch_size=BATCH_SIZE, num_steps=100,
            learning_rate=3e-3, lr_warmup_steps=5, lr_decay="constant",
            shuffle_buffer_size=1024, log_every=50, eval_every=999,
            checkpoint_every=999,
            checkpoint_dir=f"{tmp_dir}/ckpt", log_dir=f"{tmp_dir}/logs",
            dead_neuron_resample_steps=0,
        )
        source = NumpySource(path=data_path)

        # Fixed eval batch (same data for before/after)
        eval_batch = jnp.array(data[:BATCH_SIZE])

        trainer = SAETrainer(vanilla_config, cfg, source=source)
        trainer.setup(resume=False)

        # Capture initial loss on eval batch
        _, loss_dict_before = trainer.model.apply(
            {"params": trainer.state.params}, eval_batch,
            method=trainer.model.compute_loss,
        )
        loss_before = float(loss_dict_before["mse"])

        trainer.train()

        # Capture final loss on same eval batch
        _, loss_dict_after = trainer.model.apply(
            {"params": trainer.state.params}, eval_batch,
            method=trainer.model.compute_loss,
        )
        loss_after = float(loss_dict_after["mse"])
        assert loss_after < loss_before, (
            f"MSE should decrease: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_resume_from_checkpoint(self, vanilla_config, numpy_data_path, tmp_dir):
        """Training should resume from checkpoint with correct step."""
        cfg = TrainingConfig(
            batch_size=BATCH_SIZE, num_steps=10,
            learning_rate=1e-3, lr_warmup_steps=2, lr_decay="constant",
            shuffle_buffer_size=128, log_every=5, eval_every=999,
            checkpoint_every=5,
            checkpoint_dir=f"{tmp_dir}/ckpt", log_dir=f"{tmp_dir}/logs",
            dead_neuron_resample_steps=0,
        )
        # First run
        source = NumpySource(path=numpy_data_path)
        trainer = SAETrainer(vanilla_config, cfg, source=source)
        trainer.setup(resume=False)
        trainer.train()
        final_step = int(trainer.state.step)

        # Resume run
        source2 = NumpySource(path=numpy_data_path)
        trainer2 = SAETrainer(vanilla_config, cfg, source=source2)
        trainer2.setup(resume=True)
        resumed_step = int(trainer2.state.step)

        assert resumed_step > 0
        # Should resume from a checkpoint step
        assert resumed_step == final_step or resumed_step == 10 or resumed_step == 5

    def test_hidden_dim_mismatch_raises(self, numpy_data_path, tmp_dir):
        """Setup should fail if SAE hidden_dim doesn't match data."""
        bad_cfg = SAEConfig(
            hidden_dim=999, dict_size=128,
            architecture="vanilla", dtype="float32",
        )
        train_cfg = TrainingConfig(
            batch_size=16, num_steps=5,
            learning_rate=1e-3, lr_warmup_steps=2, lr_decay="constant",
            shuffle_buffer_size=64,
            checkpoint_dir=f"{tmp_dir}/ckpt", log_dir=f"{tmp_dir}/logs",
        )
        source = NumpySource(path=numpy_data_path)
        trainer = SAETrainer(bad_cfg, train_cfg, source=source)
        with pytest.raises(ValueError, match="hidden_dim mismatch"):
            trainer.setup(resume=False)

    def test_data_exhaustion_warning(self, vanilla_config, tmp_dir, capsys):
        """Should warn when data runs out before num_steps."""
        # Create tiny dataset
        data = np.random.randn(30, HIDDEN_DIM).astype(np.float32)
        path = f"{tmp_dir}/tiny.npy"
        np.save(path, data)

        cfg = TrainingConfig(
            batch_size=BATCH_SIZE, num_steps=100,
            learning_rate=1e-3, lr_warmup_steps=0, lr_decay="constant",
            shuffle_buffer_size=32, log_every=999, eval_every=999,
            checkpoint_every=999,
            checkpoint_dir=f"{tmp_dir}/ckpt", log_dir=f"{tmp_dir}/logs",
            dead_neuron_resample_steps=0,
        )
        source = NumpySource(path=path)
        trainer = SAETrainer(vanilla_config, cfg, source=source)
        trainer.setup(resume=False)
        trainer.train()

        captured = capsys.readouterr()
        assert "WARNING: Data source exhausted" in captured.out

    def test_checkpoint_creates_files(self, vanilla_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            vanilla_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()

        from pathlib import Path
        ckpt_dir = Path(small_training_config.checkpoint_dir)
        assert ckpt_dir.exists()
        ckpt_dirs = list(ckpt_dir.glob("step_*"))
        assert len(ckpt_dirs) > 0

        # Check checkpoint structure
        latest = sorted(ckpt_dirs)[-1]
        assert (latest / "metadata.json").exists()
        assert (latest / "params").is_dir()
        assert (latest / "opt_state").is_dir()
        assert (latest / "dead_neuron_steps.npy").exists()

    def test_log_file_created(self, vanilla_config, small_training_config, numpy_data_path):
        trainer = self._make_trainer(
            vanilla_config, small_training_config, numpy_data_path
        )
        trainer.setup(resume=False)
        trainer.train()

        from pathlib import Path
        log_dir = Path(small_training_config.log_dir)
        assert (log_dir / "train_log.jsonl").exists()
        assert (log_dir / "sae_config.json").exists()
        assert (log_dir / "training_config.json").exists()

        # Verify JSONL is parseable
        with open(log_dir / "train_log.jsonl") as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            entry = json.loads(line)
            assert "step" in entry
            assert "type" in entry
