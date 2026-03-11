"""Shared fixtures for SAE tests."""

import tempfile
import shutil
import gzip
import pickle

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sae.configs.base import SAEConfig
from sae.configs.training import TrainingConfig


# ---------------------------------------------------------------------------
# Small configs for fast tests
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
DICT_SIZE = 64
BATCH_SIZE = 16


@pytest.fixture
def hidden_dim():
    return HIDDEN_DIM


@pytest.fixture
def dict_size():
    return DICT_SIZE


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def random_batch(rng):
    """Random input batch [BATCH_SIZE, HIDDEN_DIM]."""
    return jax.random.normal(rng, (BATCH_SIZE, HIDDEN_DIM))


@pytest.fixture
def vanilla_config():
    return SAEConfig(
        hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
        architecture="vanilla", dtype="float32", l1_coeff=1e-2,
    )


@pytest.fixture
def topk_config():
    return SAEConfig(
        hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
        architecture="topk", dtype="float32", k=8,
    )


@pytest.fixture
def gated_config():
    return SAEConfig(
        hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
        architecture="gated", dtype="float32", l1_coeff=1e-2,
    )


@pytest.fixture
def jumprelu_config():
    return SAEConfig(
        hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE,
        architecture="jumprelu", dtype="float32", l1_coeff=1e-2,
        jumprelu_bandwidth=0.01,
    )


@pytest.fixture(params=["vanilla", "topk", "gated", "jumprelu"])
def any_config(request, vanilla_config, topk_config, gated_config, jumprelu_config):
    """Parametrized fixture: runs the test once per architecture."""
    return {
        "vanilla": vanilla_config,
        "topk": topk_config,
        "gated": gated_config,
        "jumprelu": jumprelu_config,
    }[request.param]


# ---------------------------------------------------------------------------
# Temporary data directories
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def numpy_data_path(tmp_dir):
    """Create a .npy file with random activations."""
    data = np.random.randn(500, HIDDEN_DIM).astype(np.float32)
    path = f"{tmp_dir}/activations.npy"
    np.save(path, data)
    return path


@pytest.fixture
def numpy_3d_data_path(tmp_dir):
    """Create a 3D .npy file [N, seq_len, hidden_dim]."""
    data = np.random.randn(50, 10, HIDDEN_DIM).astype(np.float32)
    path = f"{tmp_dir}/activations_3d.npy"
    np.save(path, data)
    return path


@pytest.fixture
def numpy_dir(tmp_dir):
    """Create a directory with multiple .npy files."""
    d = f"{tmp_dir}/npy_dir"
    import os
    os.makedirs(d)
    for i in range(3):
        data = np.random.randn(100, HIDDEN_DIM).astype(np.float32)
        np.save(f"{d}/shard_{i:04d}.npy", data)
    return d


@pytest.fixture
def npz_data_path(tmp_dir):
    """Create a .npz file."""
    data = np.random.randn(200, HIDDEN_DIM).astype(np.float32)
    path = f"{tmp_dir}/activations.npz"
    np.savez(path, acts=data)
    return path


@pytest.fixture
def pickle_shard_dir(tmp_dir):
    """Create a directory of gzipped pickle shards in activation-extract format."""
    d = f"{tmp_dir}/pickle_shards"
    import os
    os.makedirs(d)
    layer_idx = 5
    for shard_num in range(3):
        shard_data = {
            layer_idx: [
                {
                    "sample_idx": shard_num * 10 + i,
                    "activation": np.random.randn(8, HIDDEN_DIM).astype(np.float32),
                    "shape": (8, HIDDEN_DIM),
                    "text_preview": f"sample {i}",
                }
                for i in range(10)
            ]
        }
        path = f"{d}/shard_{shard_num:04d}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(shard_data, f)
    return d, layer_idx


@pytest.fixture
def small_training_config(tmp_dir):
    """TrainingConfig suitable for fast integration tests."""
    return TrainingConfig(
        batch_size=BATCH_SIZE,
        num_steps=10,
        seed=42,
        learning_rate=1e-3,
        lr_warmup_steps=2,
        lr_decay="constant",
        shuffle_buffer_size=128,
        log_every=5,
        eval_every=5,
        checkpoint_every=5,
        checkpoint_dir=f"{tmp_dir}/ckpt",
        log_dir=f"{tmp_dir}/logs",
        keep_last_n_checkpoints=2,
        dead_neuron_resample_steps=0,
    )
