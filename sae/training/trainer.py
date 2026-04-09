"""Main SAE trainer: orchestrates training loop, evaluation, checkpointing."""

import time
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..configs.base import SAEConfig
from ..configs.training import TrainingConfig
from ..data.base import ActivationSource
from ..data.pipeline import ActivationPipeline
from ..evaluation.metrics import compute_metrics, compute_dead_neurons
from ..models.base import BaseSAE
from ..models.registry import create_sae
from .checkpointing import (
    save_checkpoint, load_checkpoint, restore_params, restore_opt_state,
    upload_checkpoint_to_gcs, download_checkpoint_from_gcs,
)
from .distributed import create_sae_mesh, shard_batch, replicate_params, get_host_info
from .lr_schedule import create_optimizer
from .train_state import SAETrainState


class SAETrainer:
    """Sparse Autoencoder trainer for JAX/TPU.

    Handles:
        - Model initialization and parameter management
        - JIT-compiled training step with gradient updates
        - Dead neuron tracking and resampling
        - Decoder norm constraint
        - Periodic evaluation and logging
        - Checkpointing and resume
        - Single-host and multi-host TPU support

    Usage:
        trainer = SAETrainer(sae_config, training_config)
        trainer.setup()
        trainer.train()
    """

    def __init__(
        self,
        sae_config: SAEConfig,
        training_config: TrainingConfig,
        source: Optional[ActivationSource] = None,
    ):
        self.sae_config = sae_config
        self.train_config = training_config
        self.source = source

        self.model: BaseSAE = None
        self.state: SAETrainState = None
        self.mesh = None
        self.pipeline: ActivationPipeline = None
        self.host_info = None

        # Logging
        self.log_dir = Path(training_config.log_dir)
        self.log_file = None

    def setup(self, resume: bool = True):
        """Initialize everything: model, optimizer, mesh, data, optionally resume.

        Args:
            resume: If True, attempt to resume from latest checkpoint.
        """
        self.host_info = get_host_info()
        is_primary = self.host_info["is_primary"]

        if is_primary:
            print(f"\n{'='*70}")
            print(f"SAE Trainer Setup")
            print(f"{'='*70}")
            print(f"  Architecture: {self.sae_config.architecture}")
            print(f"  Hidden dim: {self.sae_config.hidden_dim}")
            print(f"  Dict size: {self.sae_config.dict_size}")
            print(f"  Expansion: {self.sae_config.dict_size / self.sae_config.hidden_dim:.0f}x")
            actual_dtype = "float32" if self._get_dtype() == jnp.float32 else "bfloat16"
            if actual_dtype != self.sae_config.dtype:
                print(f"  Dtype: {actual_dtype} (fallback from {self.sae_config.dtype})")
            else:
                print(f"  Dtype: {self.sae_config.dtype}")
            print(f"  Backend: {jax.default_backend()}")
            print(f"  Hosts: {self.host_info['num_hosts']}")
            print(f"  Devices: {self.host_info['total_devices']}")

        # 1. Create device mesh
        self.mesh = create_sae_mesh(
            self.train_config.mesh_type,
            verbose=is_primary,
        )

        # 2. Create model
        self.model = create_sae(self.sae_config)

        # 3. Initialize parameters
        rng = jax.random.PRNGKey(self.train_config.seed)
        dummy_input = jnp.zeros((1, self.sae_config.hidden_dim), dtype=self._get_dtype())
        variables = self.model.init(rng, dummy_input)

        # 4. Create optimizer
        optimizer = create_optimizer(self.train_config)

        # 5. Create JIT-compiled train step (captures model for compute_loss)
        self._jit_train_step = self._make_train_step(self.model)

        # 6. Create train state
        self.state = SAETrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=optimizer,
            dead_neuron_steps=jnp.zeros(self.sae_config.dict_size, dtype=jnp.int32),
            total_tokens=0,
        )

        # 7. Try to resume from checkpoint (GCS first, then local)
        start_step = 0
        if resume and self.train_config.upload_checkpoints_to_gcs and is_primary:
            download_checkpoint_from_gcs(
                self.train_config.checkpoint_dir,
                self.train_config.gcs_bucket,
                self.train_config.gcs_prefix,
            )
        if resume:
            ckpt = load_checkpoint(self.train_config.checkpoint_dir)
            if ckpt is not None:
                restored_params = restore_params(
                    ckpt, self.state.params
                )
                replacements = {
                    "params": restored_params,
                    "step": ckpt["metadata"]["step"],
                    "total_tokens": ckpt["metadata"].get("total_tokens", 0),
                }
                # Restore optimizer state (Adam momentum/velocity)
                restored_opt = restore_opt_state(ckpt, self.state.opt_state)
                if restored_opt is not None:
                    replacements["opt_state"] = restored_opt
                if ckpt["dead_neuron_steps"] is not None:
                    replacements["dead_neuron_steps"] = jnp.array(ckpt["dead_neuron_steps"])
                self.state = self.state.replace(**replacements)
                start_step = ckpt["metadata"]["step"]
                if is_primary:
                    print(f"  Resumed from step {start_step}")

        # 8. Replicate params across devices
        with self.mesh:
            self.state = replicate_params(self.state, self.mesh)

        # 9. Setup data pipeline (per-host data sharding in multi-host)
        self.pipeline = ActivationPipeline(
            config=self.train_config,
            source=self.source,
            host_id=self.host_info["host_id"],
            num_hosts=self.host_info["num_hosts"],
        )

        # 10. Setup logging
        if is_primary:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_dir / "train_log.jsonl", "a")

            # Save configs
            with open(self.log_dir / "sae_config.json", "w") as f:
                json.dump(asdict(self.sae_config), f, indent=2)
            with open(self.log_dir / "training_config.json", "w") as f:
                json.dump(
                    {k: v for k, v in asdict(self.train_config).items() if not callable(v)},
                    f, indent=2, default=str,
                )

            print(f"  Data hidden_dim: {self.pipeline.hidden_dim}")
            print(f"{'='*70}\n")

        # Validate hidden_dim match
        if self.pipeline.hidden_dim != self.sae_config.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: SAE config expects {self.sae_config.hidden_dim}, "
                f"but data source provides {self.pipeline.hidden_dim}"
            )

    def _get_dtype(self):
        if self.sae_config.dtype == "bfloat16":
            # CPU doesn't natively support bfloat16 — fall back to float32
            backend = jax.default_backend()
            if backend == "cpu":
                return jnp.float32
            return jnp.bfloat16
        return jnp.float32

    def train(self):
        """Main training loop."""
        is_primary = self.host_info["is_primary"]
        cfg = self.train_config
        start_step = int(self.state.step)
        step = start_step

        if is_primary:
            print(f"Starting training from step {start_step} to {cfg.num_steps}")

        batch_iter = self.pipeline.iter_batches()
        t0 = time.time()
        tokens_this_log = 0

        for batch in batch_iter:
            if step >= cfg.num_steps:
                break

            # Shard batch across devices (multi-host aware)
            with self.mesh:
                batch = shard_batch(batch, self.mesh, num_hosts=self.host_info["num_hosts"])

            # Train step (uses architecture-specific loss via compute_loss)
            self.state, loss_dict = self._jit_train_step(self.state, batch)

            # Decoder norm constraint
            if self.sae_config.normalize_decoder:
                self.state = self._normalize_decoder(self.state)

            # Update dead neuron tracking
            self.state = self._update_dead_tracking(self.state, loss_dict)

            step += 1
            tokens_this_log += cfg.batch_size

            # Logging
            if is_primary and step % cfg.log_every == 0:
                elapsed = time.time() - t0
                tps = tokens_this_log / max(elapsed, 1e-6)
                self._log_step(step, loss_dict, tps)
                t0 = time.time()
                tokens_this_log = 0

            # Evaluation (all hosts must participate for sharded batch)
            if step % cfg.eval_every == 0:
                self._evaluate(step, batch, log=is_primary)

            # Checkpointing
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=cfg.checkpoint_dir,
                    step=step,
                    params=self.state.params,
                    opt_state=self.state.opt_state,
                    dead_neuron_steps=self.state.dead_neuron_steps,
                    total_tokens=int(self.state.total_tokens),
                    sae_config=self.sae_config,
                    training_config=self.train_config,
                    keep_last_n=cfg.keep_last_n_checkpoints,
                )
                if cfg.upload_checkpoints_to_gcs:
                    upload_checkpoint_to_gcs(
                        cfg.checkpoint_dir, step, cfg.gcs_bucket, cfg.gcs_prefix,
                    )

            # Dead neuron resampling (with optional cutoff)
            if (
                cfg.dead_neuron_resample_steps > 0
                and step % cfg.dead_neuron_resample_steps == 0
                and step > 0
                and (cfg.dead_neuron_resample_until == 0 or step <= cfg.dead_neuron_resample_until)
            ):
                self._resample_dead_neurons(batch)

        # Warn if data was exhausted before num_steps
        if is_primary and step < cfg.num_steps:
            print(
                f"\nWARNING: Data source exhausted at step {step}/{cfg.num_steps}. "
                f"Consider using MultiEpochBuffer or providing more data."
            )

        # Final checkpoint
        if is_primary:
            save_checkpoint(
                checkpoint_dir=cfg.checkpoint_dir,
                step=step,
                params=self.state.params,
                opt_state=self.state.opt_state,
                dead_neuron_steps=self.state.dead_neuron_steps,
                total_tokens=int(self.state.total_tokens),
                sae_config=self.sae_config,
                training_config=self.train_config,
                keep_last_n=cfg.keep_last_n_checkpoints,
            )
            if cfg.upload_checkpoints_to_gcs:
                upload_checkpoint_to_gcs(
                    cfg.checkpoint_dir, step, cfg.gcs_bucket, cfg.gcs_prefix,
                )
            if self.log_file:
                self.log_file.close()
            print(f"\nTraining complete at step {step}.")

    @staticmethod
    def _make_train_step(model):
        """Create a JIT-compiled train step that uses the model's compute_loss.

        The model is captured in the closure so that architecture-specific
        loss functions (L1, TopK aux, JumpReLU STE, etc.) are used correctly.
        """

        @partial(jax.jit, donate_argnums=(0,))
        def train_step(state: SAETrainState, batch: jnp.ndarray):
            def loss_fn(params):
                total_loss, loss_dict = model.apply(
                    {"params": params}, batch, method=model.compute_loss
                )
                return total_loss, loss_dict

            (loss, loss_dict), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)

            state = state.apply_gradients(grads=grads)
            state = state.replace(
                total_tokens=state.total_tokens + batch.shape[0]
            )

            return state, loss_dict

        return train_step

    @staticmethod
    @jax.jit
    def _normalize_decoder(state: SAETrainState) -> SAETrainState:
        """Project decoder columns to unit norm."""
        W_dec = state.params["W_dec"]
        norms = jnp.linalg.norm(W_dec, axis=-1, keepdims=True)
        W_dec_normed = W_dec / jnp.maximum(norms, 1e-8)
        new_params = {**state.params, "W_dec": W_dec_normed}
        return state.replace(params=new_params)

    @staticmethod
    @jax.jit
    def _update_dead_tracking(
        state: SAETrainState, loss_dict: dict
    ) -> SAETrainState:
        """Update dead neuron step counter using per-feature activation info."""
        if state.dead_neuron_steps is None:
            return state

        # Use _z from loss_dict to check which features fired
        z = loss_dict.get("_z")
        if z is not None:
            # Features that activated in this batch get reset to 0
            fired = jnp.any(z != 0, axis=0)  # [dict_size]
            new_steps = jnp.where(fired, 0, state.dead_neuron_steps + 1)
        else:
            new_steps = state.dead_neuron_steps + 1

        return state.replace(dead_neuron_steps=new_steps)

    def _resample_dead_neurons(self, batch: jnp.ndarray):
        """Resample dead neurons using high-loss examples."""
        is_primary = self.host_info["is_primary"]
        window = self.train_config.dead_neuron_window
        dead_info = compute_dead_neurons(self.state.dead_neuron_steps, window)

        if is_primary:
            print(
                f"  Dead neuron resampling: {dead_info['dead_count']} dead "
                f"({dead_info['dead_frac']:.1%})"
            )

        if dead_info["dead_count"] == 0:
            return

        dead_mask = self.state.dead_neuron_steps >= window

        # Get high-loss examples for reinitialization
        x_hat, z, _ = self.model.apply({"params": self.state.params}, batch)
        per_sample_loss = jnp.mean((batch - x_hat) ** 2, axis=-1)

        # Use top-loss examples as new encoder directions
        num_dead = int(dead_mask.sum())
        top_indices = jnp.argsort(per_sample_loss)[-num_dead:]
        new_directions = batch[top_indices]  # [num_dead, hidden_dim]

        # Normalize
        norms = jnp.linalg.norm(new_directions, axis=-1, keepdims=True)
        new_directions = new_directions / jnp.maximum(norms, 1e-8)

        # Update encoder weights for dead neurons
        W_enc = self.state.params["W_enc"]  # [hidden_dim, dict_size]
        dead_indices = jnp.where(dead_mask, size=num_dead)[0]

        # Reinitialize encoder columns for dead neurons
        avg_enc_norm = jnp.mean(jnp.linalg.norm(W_enc, axis=0))
        new_enc = new_directions.T * avg_enc_norm * 0.2  # [hidden_dim, num_dead]
        W_enc_new = W_enc.at[:, dead_indices].set(new_enc)

        # Reset decoder rows
        W_dec = self.state.params["W_dec"]  # [dict_size, hidden_dim]
        W_dec_new = W_dec.at[dead_indices, :].set(new_directions)

        # Reset encoder bias for dead neurons
        b_enc = self.state.params["b_enc"]
        b_enc_new = b_enc.at[dead_indices].set(0.0)

        new_params = {
            **self.state.params,
            "W_enc": W_enc_new,
            "W_dec": W_dec_new,
            "b_enc": b_enc_new,
        }

        # Reset dead neuron counters
        dead_steps_new = self.state.dead_neuron_steps.at[dead_indices].set(0)

        self.state = self.state.replace(
            params=new_params,
            dead_neuron_steps=dead_steps_new,
        )

    def _evaluate(self, step: int, batch: jnp.ndarray, log: bool = True):
        """Run evaluation metrics on a batch.

        All hosts must call this (model.apply on sharded batch is collective).
        Only the primary host logs results.
        """
        x_hat, z, _ = self.model.apply({"params": self.state.params}, batch)
        metrics = compute_metrics(batch, x_hat, z)
        dead_info = compute_dead_neurons(
            self.state.dead_neuron_steps, self.train_config.dead_neuron_window
        )

        if log:
            print(f"  [Eval step {step}]")
            print(f"    MSE: {metrics['mse']:.6f}")
            print(f"    Explained variance: {metrics['explained_variance']:.4f}")
            print(f"    L0: {metrics['l0']:.1f}")
            print(f"    Dead neurons: {dead_info['dead_frac']:.1%}")

            if self.log_file:
                log_entry = {"step": step, "type": "eval", **metrics, **dead_info}
                self.log_file.write(json.dumps(log_entry, default=float) + "\n")
                self.log_file.flush()

    def _log_step(self, step: int, loss_dict: dict, tokens_per_sec: float):
        """Log training metrics."""
        loss_vals = {
            k: float(v) for k, v in loss_dict.items()
            if not k.startswith("_")  # skip internal arrays like _z
        }

        # Build metric string with architecture-appropriate fields
        parts = [f"Step {step:>7d}"]
        parts.append(f"loss {loss_vals.get('total', 0):.6f}")
        parts.append(f"MSE {loss_vals.get('mse', 0):.6f}")
        # Show the sparsity term relevant to this architecture
        if "l1" in loss_vals:
            parts.append(f"L1 {loss_vals['l1']:.6f}")
        if "gate_l1" in loss_vals:
            parts.append(f"gate_L1 {loss_vals['gate_l1']:.6f}")
        if "aux_loss" in loss_vals:
            parts.append(f"aux {loss_vals['aux_loss']:.6f}")
        if "l0_penalty" in loss_vals:
            parts.append(f"L0_pen {loss_vals['l0_penalty']:.6f}")
        parts.append(f"L0 {loss_vals.get('l0', 0):.1f}")
        parts.append(f"{tokens_per_sec:.0f} tok/s")
        print("  " + " | ".join(parts))

        if self.log_file:
            log_entry = {
                "step": step,
                "type": "train",
                "tokens_per_sec": tokens_per_sec,
                **loss_vals,
            }
            self.log_file.write(json.dumps(log_entry, default=float) + "\n")
            self.log_file.flush()
