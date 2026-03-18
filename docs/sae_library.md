# SAE Training Library Reference

The `sae/` package is a generic Sparse Autoencoder training library optimized for JAX/TPU. It trains SAEs on pre-extracted activations from the activation extraction pipeline.

Source location: `/home/kathirks_gc/sae-worktree/sae/` (deployed to workers via `scripts/launch_sae_training_v6e.sh`)

---

## Architecture Overview

```
sae/
  configs/       # SAEConfig, TrainingConfig, presets
  data/          # Pluggable activation sources + shuffle buffer + pipeline
  models/        # SAE architectures (Vanilla, TopK, Gated, JumpReLU)
  training/      # Trainer, distributed, checkpointing, LR schedules
  evaluation/    # Metrics (MSE, explained variance, dead neurons)
  scripts/       # CLI entry point (train.py)
```

### Data Flow

```
Pickle Shards (from extraction) -> ActivationSource -> ShuffleBuffer -> ActivationPipeline
    -> SAETrainer.train() -> JIT train_step(state, batch) -> loss + grads -> state update
    -> Checkpoints (local + GCS) + JSONL logs
```

---

## Configs

### SAEConfig (`sae/configs/base.py`)

```python
@dataclass
class SAEConfig:
    hidden_dim: int = 896          # Must match source model's hidden layer
    dict_size: int = 896 * 16     # SAE features (expansion ratio: 8x, 16x, 32x, 64x)
    architecture: str = "vanilla"  # "vanilla", "topk", "gated", "jumprelu"
    l1_coeff: float = 5e-3        # L1 penalty (Vanilla/Gated)
    k: int = 64                   # Top-k value (TopK)
    topk_aux_coeff: float = 1/32  # Auxiliary loss coefficient (TopK)
    jumprelu_init_threshold: float = 0.001
    jumprelu_bandwidth: float = 0.001
    gated_bandwidth: float = 0.001
    dec_init_norm: float = 0.1
    normalize_decoder: bool = True
    dtype: str = "bfloat16"
```

### TrainingConfig (`sae/configs/training.py`)

```python
@dataclass
class TrainingConfig:
    # Data
    source_type: str = "pickle"    # "pickle", "numpy", "safetensors", "hf_dataset"
    source_kwargs: Dict = field(default_factory=dict)
    layer_index: int = 12

    # Training
    batch_size: int = 4096         # Global batch size
    num_steps: int = 100_000
    seed: int = 42

    # Optimizer
    optimizer: str = "adam"         # "adam" or "adamw"
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 1000
    lr_decay: str = "cosine"       # "cosine", "constant", "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Buffering
    shuffle_buffer_size: int = 262_144  # 256k vectors (~915 MB for 896-dim float32)

    # Dead neurons
    dead_neuron_resample_steps: int = 25_000
    dead_neuron_window: int = 10_000

    # Logging & checkpointing
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 5000
    keep_last_n_checkpoints: int = 3

    # GCS (preemption recovery)
    upload_checkpoints_to_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "sae_checkpoints"

    # Distributed
    mesh_type: str = "auto"        # "auto", "1d", "data_parallel"
```

### Presets (`sae/configs/presets.py`)

```python
PRESETS = {
    "qwen-0.5b-vanilla": qwen_0_5b_vanilla,   # hidden=896, expansion=16x, l1=5e-3
    "qwen-0.5b-topk": qwen_0_5b_topk,         # hidden=896, expansion=16x, k=64
}
```

Each preset returns `(SAEConfig, TrainingConfig)`.

---

## SAE Architectures

All architectures share a common encoder/decoder structure (`BaseSAE`). They differ only in `apply_sparsity()` and `loss()`.

### Common Structure (BaseSAE, `sae/models/base.py`)

```
Parameters:
  W_enc: [hidden_dim, dict_size]     # Encoder weights
  b_enc: [dict_size]                 # Encoder bias
  W_dec: [dict_size, hidden_dim]     # Decoder weights
  b_dec: [hidden_dim]                # Decoder bias (also used as data mean)

Forward pass:
  x_centered = x - b_dec
  z_pre = x_centered @ W_enc + b_enc
  z, aux = apply_sparsity(z_pre)     # Architecture-specific
  x_hat = z @ W_dec + b_dec
```

### Vanilla SAE (`sae/models/vanilla.py`)

Standard ReLU + L1 sparsity penalty.

```
Sparsity:  z = ReLU(z_pre)
Loss:      MSE(x, x_hat) + l1_coeff * mean(sum(|z|))
```

### TopK SAE (`sae/models/topk.py`)

Hard constraint: keep only top-k activations per sample.

```
Sparsity:  z = zeros; z[top_k_indices] = ReLU(z_pre[top_k_indices])
Loss:      MSE(x, x_hat) + topk_aux_coeff * MSE(x, x_hat_aux)
           where x_hat_aux uses ReLU(z_pre) for ALL features (gradient signal to non-top-k)
```

Guarantees exactly k active features. Auxiliary loss prevents dead neurons.

### Gated SAE (`sae/models/gated.py`)

Separate gating and magnitude pathways (DeepMind 2024).

```
Extra params: W_gate [hidden_dim, dict_size], b_gate [dict_size], r_mag [dict_size]
Sparsity:  gate = heaviside_ste(x_centered @ W_gate + b_gate)
           magnitude = ReLU(z_pre) * exp(r_mag)
           z = gate * magnitude
Loss:      MSE(x, x_hat) + l1_coeff * mean(sum(ReLU(gate_pre)))
```

### JumpReLU SAE (`sae/models/jumprelu.py`)

Learnable per-feature thresholds (Anthropic 2024).

```
Extra params: log_threshold [dict_size]
Sparsity:  threshold = exp(log_threshold)
           z = where(z_pre > threshold, z_pre, 0)
Loss:      MSE(x, x_hat) + l1_coeff * mean(sum(heaviside_ste(z_pre - threshold)))
```

Uses `@jax.custom_jvp` for straight-through estimators on discrete operations (JumpReLU, Heaviside). Rectangular kernel approximation enables gradient flow through thresholds.

### Custom Gradient Functions (`sae/models/losses.py`)

```python
jumprelu(z_pre, threshold, bandwidth=0.001)      # Forward: z_pre * (z_pre > threshold)
                                                   # Backward: rectangular kernel for threshold grad
heaviside_ste(x, bandwidth=0.001)                 # Forward: step function
                                                   # Backward: rectangular kernel
```

### Registry (`sae/models/registry.py`)

```python
create_sae(config: SAEConfig) -> BaseSAE           # Factory
register_sae(name: str, cls: type)                  # Add custom architecture
```

---

## Data Sources

All sources implement the `ActivationSource` abstract base class.

### Abstract Interface (`sae/data/base.py`)

```python
class ActivationSource(ABC):
    @abstractmethod
    def iter_vectors(self) -> Iterator[np.ndarray]    # Yield [hidden_dim] vectors
    @property
    @abstractmethod
    def hidden_dim(self) -> int

    def iter_batches(self, batch_size) -> Iterator[np.ndarray]  # Default: collect from iter_vectors
```

### PickleShardSource (`sae/data/pickle_source.py`)

Loads gzipped pickle shards from the activation extraction pipeline.

```python
PickleShardSource(
    shard_dir: str,              # Local path to shard files
    layer_index: int,            # Which layer to extract
    compressed: bool = True,
    shuffle_shards: bool = True,
    seed: int = 42,
    gcs_path: Optional[str] = None,   # gs:// path (overrides shard_dir)
    host_id: int = 0,            # For multi-host shard distribution
    num_hosts: int = 1,
)
```

Supports multi-host directories (e.g., `gs://bucket/host_00/`, `gs://bucket/host_01/`). Uses deterministic round-robin for shard claiming across hosts.

Shard format: `{layer_idx: [{'sample_idx': int, 'activation': np.ndarray[seq_len, hidden_dim], ...}, ...]}`

For 3D activations `[seq_len, hidden_dim]`, flattens to individual `[hidden_dim]` vectors.

### NumpySource (`sae/data/numpy_source.py`)

```python
NumpySource(path: str, key: Optional[str] = None, flatten_sequences: bool = True)
```

Loads `.npy`, `.npz`, or directories of `.npy` files. Handles 2D `[N, hidden_dim]` and 3D `[N, seq_len, hidden_dim]` arrays.

### SafetensorsSource (`sae/data/safetensors_source.py`)

```python
SafetensorsSource(path: str, tensor_name: str = "activations", flatten_sequences: bool = True)
```

Zero-copy loading via `safetensors.numpy.load_file`.

### HFDatasetSource (`sae/data/hf_dataset_source.py`)

```python
HFDatasetSource(dataset_name: str, column: str = "activations", split: str = "train",
                streaming: bool = True, flatten_sequences: bool = True)
```

### Registry (`sae/data/registry.py`)

```python
create_source(source_type: str, **kwargs) -> ActivationSource
register_source(name: str, cls)
```

Built-in: `"pickle"`, `"numpy"`, `"safetensors"`, `"hf_dataset"`

---

## Data Pipeline

### ShuffleBuffer (`sae/data/buffer.py`)

In-memory shuffle buffer to decorrelate sequential shard data.

```python
ShuffleBuffer(source: ActivationSource, buffer_size: int, seed: int = 42)
    .iter_batches(batch_size) -> Iterator[jnp.ndarray]
```

Memory: 256k vectors * 896 dim * 4 bytes = ~915 MB.

Algorithm: fill buffer from source, shuffle, yield complete batches, refill.

### MultiEpochBuffer

```python
MultiEpochBuffer(source_factory: Callable, buffer_size: int, seed: int = 42)
    .iter_batches(batch_size, num_epochs=1)
```

Creates a fresh `ActivationSource` each epoch with varied seed.

### ActivationPipeline (`sae/data/pipeline.py`)

End-to-end pipeline: source -> shuffle buffer -> batch -> JAX array.

```python
ActivationPipeline(config: TrainingConfig, source=None, host_id=0, num_hosts=1)
    .iter_batches() -> Iterator[jnp.ndarray]      # Infinite, yields [per_host_batch, hidden_dim]
    .iter_batches_unbuffered() -> Iterator          # No shuffle (for eval)
```

In multi-host mode: `per_host_batch = global_batch_size // num_hosts`. Each host gets its own subset of shards.

---

## Training

### SAETrainer (`sae/training/trainer.py`)

Main training orchestrator.

```python
class SAETrainer:
    def __init__(self, sae_config: SAEConfig, training_config: TrainingConfig,
                 source: Optional[ActivationSource] = None)
    def setup(resume: bool = True)
    def train()
```

#### setup() Steps

1. Initialize distributed (auto-detect single vs multi-host)
2. Create device mesh (data parallel)
3. Initialize model parameters (Flax init)
4. Create optimizer with LR schedule
5. Create JIT-compiled train step
6. Create `SAETrainState` (extends Flax TrainState with dead neuron tracking)
7. Resume from checkpoint (GCS -> local if available)
8. Replicate params across all devices
9. Setup `ActivationPipeline` (per-host data sharding)
10. Validate hidden_dim matches data source

#### train() Loop

```
for step in range(num_steps):
    batch = next(pipeline)
    batch = shard_batch(batch, mesh, num_hosts)
    state, loss_dict = jit_train_step(state, batch)
    normalize_decoder(state)           # if config.normalize_decoder
    update_dead_tracking(state, z)
    if step % log_every == 0: log_metrics()
    if step % eval_every == 0: evaluate()
    if step % checkpoint_every == 0: save_checkpoint()
    if step % dead_neuron_resample_steps == 0: resample_dead()
```

#### JIT Train Step

```python
@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        total_loss, loss_dict = model.apply(
            {"params": params}, batch, method=model.compute_loss
        )
        return total_loss, loss_dict
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_dict
```

`donate_argnums=(0,)` allows JAX to reuse state memory buffers.

#### Dead Neuron Resampling

Tracks per-feature step counters. Features that haven't activated within `dead_neuron_window` steps are resampled: encoder/decoder rows are re-initialized using high-loss examples from the current batch.

### Distributed (`sae/training/distributed.py`)

```python
create_sae_mesh(mesh_type="auto") -> Mesh
```
Creates a 1D mesh with `(data,)` axis. All devices on the data axis for pure batch parallelism. SAE params (~50MB) are replicated.

```python
shard_batch(batch, mesh, num_hosts=1) -> global_batch
```
Single-host: `jax.device_put(batch, P("data", None))`.
Multi-host: splits per-host batch across local devices, assembles global array via `jax.make_array_from_single_device_arrays`.

```python
replicate_params(params, mesh) -> replicated_params
```
Replicates all parameters with `P()` (empty partition spec = replicate everywhere).

```python
initialize_distributed(verbose=False) -> Dict
```
On TPU pods: auto-detects topology by accessing `jax.devices()`.
On standalone VMs: explicit `jax.distributed.initialize()` with coordinator from env vars.

### LR Schedules (`sae/training/lr_schedule.py`)

```python
create_lr_schedule(config) -> optax.Schedule
create_optimizer(config) -> optax.GradientTransformation
```

Schedules: cosine (warmup + decay), linear (warmup + decay), constant (optional warmup).
Optimizer chain: `clip_by_global_norm` -> Adam/AdamW with schedule.

### Checkpointing (`sae/training/checkpointing.py`)

```python
save_checkpoint(checkpoint_dir, step, params, opt_state, dead_neuron_steps, ...)
upload_checkpoint_to_gcs(checkpoint_dir, step, gcs_bucket, gcs_prefix)
download_checkpoint_from_gcs(checkpoint_dir, gcs_bucket, gcs_prefix) -> Optional[int]
load_checkpoint(checkpoint_dir, step=None) -> Optional[Dict]
restore_params(checkpoint_data, params_template) -> params_pytree
restore_opt_state(checkpoint_data, opt_state_template) -> opt_state_pytree
```

Format: `checkpoint_dir/step_XXXXXXXX/params/*.npy + opt_state/*.npy + metadata.json`

GCS marker: `gs://bucket/prefix/latest_step.json` with `{"step": step}`.

Only primary host (process_index == 0) saves/uploads. Old checkpoints pruned to `keep_last_n`.

### SAETrainState (`sae/training/train_state.py`)

Extends Flax `TrainState` with:
- `dead_neuron_steps: jnp.ndarray` - `[dict_size]` step counters
- `total_tokens: int` - Total vectors processed

---

## Evaluation

### Metrics (`sae/evaluation/metrics.py`)

```python
compute_metrics(x, x_hat, z) -> Dict[str, float]
```

Returns: `mse`, `explained_variance`, `normalized_mse`, `l0`, `l0_frac`, `dead_neuron_frac`, `mean_feature_density`, `max_feature_density`, `min_feature_density`.

```python
compute_dead_neurons(dead_neuron_steps, window) -> Dict[str, float]
```

Returns: `dead_count`, `dead_frac`, `max_inactive_steps`.

---

## CLI Entry Point

### sae/scripts/train.py

```bash
python -m sae.scripts.train \
    --architecture topk \
    --hidden_dim 896 \
    --dict_size 7168 \
    --k 32 \
    --dtype bfloat16 \
    --source_type pickle \
    --data_dir /tmp/unused \
    --gcs_path 'gs://bucket/activations/layer12_v1_50k' \
    --layer_index 12 \
    --batch_size 4096 \
    --num_steps 200000 \
    --learning_rate 3e-4 \
    --lr_warmup_steps 1000 \
    --lr_decay cosine \
    --checkpoint_every 5000 \
    --upload_checkpoints_to_gcs \
    --checkpoint_gcs_bucket bucket-name \
    --checkpoint_gcs_prefix sae_checkpoints/run_001 \
    --mesh_type data_parallel \
    --enable_barrier_sync \
    --barrier_controller_host WORKER_0_IP
```

Supports `--preset qwen-0.5b-vanilla` to use pre-built configs.

Barrier sync runs BEFORE JAX init to coordinate multihost SSH stagger.

---

## Launch Script

### scripts/launch_sae_training_v6e.sh

Resilient launcher for v6e-64 TPU pod with preemption recovery.

```bash
nohup bash scripts/launch_sae_training_v6e.sh > sae_training.log 2>&1 &
```

Current config:
- Architecture: TopK (k=32)
- Hidden dim: 896, Dict size: 7168 (8x expansion)
- Batch size: 4096 global
- 200K steps, cosine LR decay
- Source: `gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64`

Features:
- Deploys code to all 16 workers via tarball
- Monitors TPU status every 5 minutes
- On preemption: waits for TPU recreation, redeploys, relaunches
- Training resumes from GCS checkpoint automatically
