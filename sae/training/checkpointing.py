"""Checkpoint management using orbax."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..configs.base import SAEConfig
from ..configs.training import TrainingConfig


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    params: Dict,
    opt_state: Any,
    dead_neuron_steps: jnp.ndarray,
    total_tokens: int,
    sae_config: SAEConfig,
    training_config: TrainingConfig,
    keep_last_n: int = 3,
):
    """Save a training checkpoint.

    Saves params, optimizer state, and metadata as numpy arrays + JSON.
    Simple file-based approach that works across single/multi-host.

    Args:
        checkpoint_dir: Base directory for checkpoints.
        step: Current training step.
        params: Model parameters (pytree of arrays).
        opt_state: Optimizer state.
        dead_neuron_steps: Dead neuron tracking array.
        total_tokens: Total tokens seen so far.
        sae_config: Model config to save.
        training_config: Training config to save.
        keep_last_n: Number of recent checkpoints to keep.
    """
    # Only save on primary host
    if jax.process_index() != 0:
        return

    ckpt_dir = Path(checkpoint_dir) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save params as flat numpy arrays
    flat_params = jax.tree.leaves(params)
    param_keys = _get_param_keys(params)

    params_dir = ckpt_dir / "params"
    params_dir.mkdir(exist_ok=True)
    for key, arr in zip(param_keys, flat_params):
        np.save(params_dir / f"{key}.npy", np.array(arr))

    # Save optimizer state as flat numpy arrays
    opt_dir = ckpt_dir / "opt_state"
    opt_dir.mkdir(exist_ok=True)
    flat_opt = jax.tree.leaves(opt_state)
    for i, arr in enumerate(flat_opt):
        np.save(opt_dir / f"opt_{i:04d}.npy", np.array(arr))

    # Save dead neuron tracking
    np.save(ckpt_dir / "dead_neuron_steps.npy", np.array(dead_neuron_steps))

    # Save metadata
    from dataclasses import asdict

    metadata = {
        "step": step,
        "total_tokens": int(total_tokens),
        "param_keys": param_keys,
        "sae_config": asdict(sae_config),
        "training_config": {
            k: v
            for k, v in asdict(training_config).items()
            if not callable(v)
        },
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Cleanup old checkpoints
    _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    print(f"  Checkpoint saved: {ckpt_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
) -> Optional[Dict]:
    """Load a checkpoint.

    Args:
        checkpoint_dir: Base directory for checkpoints.
        step: Specific step to load, or None for latest.

    Returns:
        Dict with 'params_flat', 'param_keys', 'dead_neuron_steps',
        'metadata', or None if no checkpoint found.
    """
    base = Path(checkpoint_dir)
    if not base.exists():
        return None

    # Find checkpoint directory
    if step is not None:
        ckpt_dir = base / f"step_{step:08d}"
    else:
        ckpt_dir = _find_latest_checkpoint(base)

    if ckpt_dir is None or not ckpt_dir.exists():
        return None

    # Load metadata
    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load params
    param_keys = metadata["param_keys"]
    params_dir = ckpt_dir / "params"
    params_flat = [np.load(params_dir / f"{key}.npy") for key in param_keys]

    # Load optimizer state
    opt_dir = ckpt_dir / "opt_state"
    opt_flat = None
    if opt_dir.exists():
        opt_files = sorted(opt_dir.glob("opt_*.npy"))
        opt_flat = [np.load(f) for f in opt_files]

    # Load dead neuron tracking
    dead_path = ckpt_dir / "dead_neuron_steps.npy"
    dead_neuron_steps = np.load(dead_path) if dead_path.exists() else None

    print(f"  Checkpoint loaded: {ckpt_dir} (step {metadata['step']})")

    return {
        "params_flat": params_flat,
        "param_keys": param_keys,
        "opt_state_flat": opt_flat,
        "dead_neuron_steps": dead_neuron_steps,
        "metadata": metadata,
    }


def restore_params(checkpoint_data: Dict, params_template: Dict) -> Dict:
    """Restore params pytree structure from flat arrays.

    Args:
        checkpoint_data: Output of load_checkpoint().
        params_template: A params pytree with the right structure (e.g., from model.init).

    Returns:
        Restored params pytree.
    """
    flat_arrays = checkpoint_data["params_flat"]
    tree_def = jax.tree.structure(params_template)
    return jax.tree.unflatten(tree_def, [jnp.array(a) for a in flat_arrays])


def restore_opt_state(checkpoint_data: Dict, opt_state_template: Any) -> Any:
    """Restore optimizer state pytree from flat arrays.

    Args:
        checkpoint_data: Output of load_checkpoint().
        opt_state_template: An opt_state pytree with the right structure.

    Returns:
        Restored opt_state pytree, or None if no opt_state was saved.
    """
    opt_flat = checkpoint_data.get("opt_state_flat")
    if opt_flat is None:
        return None
    tree_def = jax.tree.structure(opt_state_template)
    return jax.tree.unflatten(tree_def, [jnp.array(a) for a in opt_flat])


def _get_param_keys(params: Dict) -> list:
    """Get flat key names for a params pytree."""
    leaves, treedef = jax.tree.flatten_with_path(params)
    keys = []
    for path, _ in leaves:
        key_parts = []
        for p in path:
            if hasattr(p, "key"):
                key_parts.append(str(p.key))
            else:
                key_parts.append(str(p))
        keys.append("_".join(key_parts))
    return keys


def _find_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint directory."""
    ckpt_dirs = sorted(base_dir.glob("step_*"))
    return ckpt_dirs[-1] if ckpt_dirs else None


def _cleanup_old_checkpoints(checkpoint_dir: str, keep: int):
    """Remove old checkpoints, keeping only the most recent `keep`."""
    import shutil

    base = Path(checkpoint_dir)
    ckpt_dirs = sorted(base.glob("step_*"))

    if len(ckpt_dirs) <= keep:
        return

    for old_dir in ckpt_dirs[:-keep]:
        shutil.rmtree(old_dir)
