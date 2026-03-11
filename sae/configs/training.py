"""Training configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    # -- Data --
    source_type: str = "pickle"  # "pickle", "numpy", "safetensors", "hf_dataset"
    source_kwargs: Dict[str, Any] = field(default_factory=dict)
    layer_index: int = 12

    # -- Training --
    batch_size: int = 4096
    num_steps: int = 100_000
    seed: int = 42

    # -- Optimizer --
    optimizer: str = "adam"  # "adam" or "adamw"
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 1000
    lr_decay: str = "cosine"  # "cosine", "constant", "linear"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0

    # -- Shuffle buffer --
    shuffle_buffer_size: int = 262_144  # 256k vectors

    # -- Dead neuron resampling --
    dead_neuron_resample_steps: int = 25_000
    dead_neuron_window: int = 10_000

    # -- Logging --
    log_every: int = 100
    eval_every: int = 1000
    log_dir: str = "./sae_logs"

    # -- Checkpointing --
    checkpoint_dir: str = "./sae_checkpoints"
    checkpoint_every: int = 5000
    keep_last_n_checkpoints: int = 3

    # -- GCS --
    upload_checkpoints_to_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "sae_checkpoints"

    # -- Distributed --
    mesh_type: str = "auto"  # "auto", "1d", "data_parallel"

    # -- Evaluation --
    eval_batch_size: int = 8192
    eval_batches: int = 10
