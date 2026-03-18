# Core Modules Reference

This document covers every module in the `core/` package and the root-level extraction/model files.

---

## core/jax_utils.py

JAX utilities for multi-host TPU processing: device mesh creation, parameter sharding, and multihost coordination.

### Initialization

```python
initialize_multihost(coordinator_address, num_hosts, host_id, verbose=False)
```
Explicit JAX distributed initialization for multihost setups.

```python
initialize_multihost_auto(verbose=False)
```
Auto-detect multihost config from environment variables (`TPU_WORKER_HOSTNAMES`, `CLOUD_TPU_TASK_ID`) or GCE metadata.

### Device Mesh

```python
create_device_mesh(mesh_type='auto', verbose=False, fsdp_size=None) -> Mesh
```
Creates a JAX device mesh. Mesh types:
- `'auto'`: 1D `(model,)` for single-host, 3D `(data, fsdp, model)` for multi-host
- `'1d'`: Single axis `(model,)`
- `'2d'`: Two axes `(data, model)`
- `'3d'`: Three axes `(data, fsdp, model)` for FSDP

For multi-host auto mode, `fsdp_size` defaults to `min(2, num_local_devices)` and `model_size = num_local_devices // fsdp_size`.

**Important:** On v5litepod-64 (16 hosts x 4 chips), auto mode creates a 3D mesh `(data=16, fsdp=2, model=2)`. The `model` axis splits the hidden dimension (896/2=448 per shard). Activations must be all-gathered before extraction.

```python
create_sharding_strategy(mesh) -> Dict[str, NamedSharding]
```
Returns named shardings for: `weights`, `embed`, `bias`, `layernorm`, `activations`, `replicated`.

### Sharding & Computation

```python
shard_params(params, sharding_strategy) -> params
```
Recursively applies sharding to model parameters based on parameter name patterns.

```python
extract_activations_sharded(model, params, input_ids) -> activations
```
JIT-compiled forward pass that returns intermediate layer activations. Decorated with `@jax.jit(static_argnums=(0,))`.

```python
pad_sequences(sequences, pad_token_id=0, fixed_length=None) -> np.ndarray
```
Pads variable-length token lists to uniform length.

### Multihost Utilities

```python
get_host_info() -> Dict                          # host_id, num_hosts, is_primary, etc.
distribute_data_across_hosts(data, host_id, num_hosts) -> data_slice  # Round-robin
gather_activations_to_primary(local_activations, host_id) -> gathered  # process_allgather
sync_hosts(tag, timeout_seconds=300)              # Barrier via multihost_utils
is_primary_host() -> bool                         # jax.process_index() == 0
get_per_host_batch_indices(total, batch_size, host_id, num_hosts) -> indices
```

---

## core/mesh_configs.py

Pre-defined TPU topology configurations for common pod sizes. Provides a simpler 2D mesh alternative to the 3D FSDP mesh from `jax_utils.py`.

### TopologyConfig

```python
@dataclass
class TopologyConfig:
    name: str
    hosts: int
    chips_per_host: int
    mesh_shape: Tuple[int, int]       # (data_parallel, model_parallel)
    axis_names: Tuple[str, str]       # ('data', 'model')
    recommended_batch_size: int
    recommended_seq_length: int

    # Properties
    total_chips -> int                # hosts * chips_per_host
    data_parallel_size -> int         # mesh_shape[0]
    model_parallel_size -> int        # mesh_shape[1]
```

### Supported Topologies

| Name | Hosts | Chips/Host | Mesh | Total |
|------|-------|------------|------|-------|
| v5e-8 | 1 | 8 | (1, 8) | 8 |
| v5litepod-64 | 16 | 4 | (16, 4) | 64 |
| v5e-64 | 4 | 8 | (4, 8) | 32 |
| v5e-128 | 8 | 8 | (8, 8) | 64 |
| v5e-256 | 16 | 8 | (16, 8) | 128 |
| v6e-8 | 1 | 8 | (1, 8) | 8 |
| v6e-64 | 16 | 4 | (16, 4) | 64 |
| v6e-128 | 16 | 8 | (16, 8) | 128 |
| v6e-256 | 32 | 8 | (32, 8) | 256 |
| v4-32 | 4 | 4 | (4, 4) | 16 |
| v4-64 | 8 | 4 | (8, 4) | 32 |

### Functions

```python
get_topology_config(topology: str) -> TopologyConfig
detect_topology() -> Optional[str]                    # Auto-detect from JAX devices
create_mesh_for_topology(topology, verbose=False) -> Mesh
create_sharding_specs(mesh, config) -> Dict[str, NamedSharding]
get_per_host_batch_size(topology) -> int
validate_batch_size(batch_size, topology) -> bool
```

---

## core/barrier_sync.py

TCP socket-based barrier synchronization for coordinating multiple TPU workers. Solves "unexpected peer in launch group" JAX errors caused by staggered SSH connections.

### Architecture

Worker 0 runs a `BarrierServer`. All workers (including worker 0) connect as `BarrierClient`s. At each barrier point, workers block until ALL workers arrive, then the server releases them simultaneously.

### BarrierServer

```python
class BarrierServer:
    def __init__(self, num_workers: int, port: int = 5555, host: str = '0.0.0.0')
    def start()                    # Blocking
    def start_background(wait_ready=True, ready_timeout=10)  # Daemon thread
    def stop()                     # Graceful shutdown
```

Message protocol: `BARRIER:<barrier_name>:<worker_id>\n`

### BarrierClient

```python
class BarrierClient:
    def __init__(self, controller_host: str, worker_id: int, port: int = 5555)
    def wait_at_barrier(barrier_name: str, timeout: int = 300) -> bool
```

### Auto-Detection Helpers

```python
get_worker0_internal_ip() -> str   # From BARRIER_CONTROLLER_HOST, TPU_WORKER_HOSTNAMES, or GCE metadata
get_worker_id() -> int             # From CLOUD_TPU_TASK_ID or TPU_WORKER_ID
get_num_workers() -> int           # From TPU_WORKER_COUNT or GCE metadata
```

### Convenience API

```python
init_barrier_sync(num_workers, controller_host, port, worker_id) -> (server_or_None, client)
barrier(name, timeout=300)         # Global barrier wait
shutdown_barrier_sync()            # Cleanup
```

### Barrier Points in multihost_extract.py

1. `pre_jax_init` - Before JAX distributed initialization
2. `post_jax_init` - After JAX sees all devices
3. `model_loaded` - After model weights loaded
4. `sharding_complete` - After parameters sharded across mesh
5. `extraction_start` - Before processing batches (timeout: 1800s for grid chunking)
6. `final_checkpoint` - After last batch
7. `complete` - Before shutdown

---

## core/activation_storage.py

Handles saving activations to disk with automatic sharding and GCS upload.

### ActivationStorage

```python
class ActivationStorage:
    def __init__(
        self,
        output_dir: str,
        upload_to_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = 'activations',
        shard_size_gb: float = 1.0,         # Auto-shard threshold
        compress_shards: bool = True,        # gzip compression
        delete_local_after_upload: bool = False,
        resume_from_shard: int = 0,
        resume_from_activations: int = 0,
    )

    def add_activation(layer_idx, activation: np.ndarray, sample_idx: int, text_preview: str)
    def finalize()                          # Flush buffer + write metadata.json
```

**Behavior:** Buffers activations in memory, tracking size in bytes. When buffer exceeds `shard_size_bytes`, auto-saves a shard as `shard_NNNN.pkl.gz`. Uses fsspec for GCS uploads.

### Output Format

```
output_dir/
  shard_0001.pkl.gz    # gzipped pickle: {layer_idx: [{sample_idx, activation, shape, text_preview}, ...]}
  shard_0002.pkl.gz
  metadata.json        # Summary: total_shards, total_activations, per-shard info
```

### Loading

```python
load_activation_shard(shard_path: str, compressed: bool = True) -> Dict
```

---

## core/dataset_utils.py

Loading ARC datasets from JSONL with machine-based sharding.

### Functions

```python
load_arc_dataset_jsonl(
    dataset_path: str,               # Local path or gs:// URI
    max_tasks: int = None,
    machine_id: int = 0,             # Round-robin sharding
    total_machines: int = 1,
    verbose: bool = False,
) -> Dict[str, Dict]                 # {task_id: {train: [...], test: [...]}}
```

```python
load_arc_dataset_from_shard(
    sharded_dataset_dir: str,
    worker_id: int,
    preferred_shard_id: int = None,
    verbose: bool = False,
) -> Tuple[Dict, int, ShardManager]  # (tasks, shard_id, manager)
```

```python
create_prompts_from_dataset(
    tasks: Dict,
    grid_encoder,
    tokenizer,
    prompt_version: str = 'default',
    predictions_per_task: int = 8,
    random_seed: int = None,
    verbose: bool = False,
) -> List[Dict]                      # [{task_id, prompt, metadata}, ...]
```

Uses `arc24.data_augmentation` for horizontal flips, rotations, and color remapping.

---

## core/dynamic_batching.py

Length-bucketed batching with adaptive batch sizes to maintain roughly constant memory usage across variable-length sequences.

### Constants

```python
DEFAULT_LENGTH_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_BATCH_SIZES = {512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 8, 16384: 4, 32768: 2}
```

Calibrated for v5litepod-64 with Qwen 2.5-0.5B (896 hidden_dim, bfloat16).

### DynamicBatch

```python
@dataclass
class DynamicBatch:
    sequences: List[List[int]]       # Token ID lists (unpadded)
    original_indices: List[int]
    actual_lengths: List[int]
    bucket_size: int                 # Padded length for this batch
    batch_size: int                  # Global batch size
```

### Functions

```python
create_dynamic_batches(
    sequences, prompts_data,
    max_seq_length=32768,
    length_buckets=None, batch_sizes=None,
    num_hosts=1, verbose=False,
) -> Tuple[List[DynamicBatch], List, List]
```

Filters sequences exceeding max_seq_length, sorts by length, groups into buckets. Batch sizes are rounded to be divisible by num_hosts.

```python
pad_batch_to_bucket(sequences, bucket_size, batch_size, pad_token_id=0)
-> Tuple[List[List[int]], int]       # (padded_sequences, actual_count)
```

---

## core/grid_chunking.py

Continuous grid chunking pipeline for SAE activation extraction. Strips all prompt text, converts only grid data to tokens, concatenates into a continuous stream, and splits into fixed-size chunks.

### Pipeline

```
Tasks -> Grid Encoder -> Token Stream -> Fixed-Size Chunks
```

1. For each task (with data augmentation): encode train inputs/outputs and test inputs as grid text
2. Tokenize each grid, append to continuous stream with `\n` separators
3. Split stream into fixed-size chunks (default 5120 tokens)

### ChunkMetadata

```python
@dataclass
class ChunkMetadata:
    chunk_idx: int
    num_tokens: int              # Actual tokens before padding
    is_last: bool
    task_boundaries: List[int]   # Token offsets where new tasks start
```

### Functions

```python
# End-to-end pipeline
create_grid_chunks_from_dataset(
    tasks, grid_encoder, tokenizer,
    chunk_size=2048, predictions_per_task=8,
    random_seed=None, verbose=False,
) -> Tuple[List[List[int]], List[ChunkMetadata], List[Dict]]

# Individual steps
create_grid_token_stream(tasks, grid_encoder, tokenizer, ...) -> (token_stream, stream_metadata)
chunk_token_stream(token_stream, chunk_size, pad_token_id, ...) -> (chunks, metadata)

# Caching (saves ~2 hours on restart)
get_chunk_cache_path(gcs_bucket, gcs_prefix, task_ids, chunk_size, ...) -> str
save_chunks_cache(chunks, chunk_metadata, stream_metadata, cache_path)
load_chunks_cache(cache_path) -> Optional[Tuple]
```

Cache key is a SHA256 hash of sorted task IDs + pipeline parameters. Any change invalidates the cache.

---

## core/stream_manager.py

GCS-backed JSON manifest for managing multi-stream extraction workloads. Tracks which dataset streams are pending, in-progress, or completed.

### StreamManager

```python
class StreamManager:
    def __init__(self, manifest_path: str, verbose=True)
    # manifest_path: local or gs:// URI

    def create_manifest(total_streams, dataset_dir, overwrite=False) -> Dict
    def claim_next_stream(pod_id, stream_range=None) -> Optional[Dict]
    def mark_stream_complete(stream_id: int)
    def get_status_summary() -> Dict    # total, completed, in_progress, pending, pct_complete
    def get_stream_info(stream_id) -> Optional[Dict]
```

Each TPU pod is assigned a fixed range of stream IDs (no race conditions on GCS). Supports resume: if a pod's in-progress stream matches, it re-claims it.

---

## qwen2_jax.py

JAX/Flax implementation of Qwen 2.5 0.5B for TPU inference.

### QwenConfig

```python
@struct.dataclass
class QwenConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    dtype: str = 'bfloat16'
```

### Model Components

- `RMSNorm` - Layer normalization
- `QwenMLP` - Gate/up/down projections with SiLU
- `QwenAttention` - Grouped Query Attention (GQA) with RoPE
- `QwenDecoderLayer` - Self-attention + MLP + residual connections
- `QwenModel` - Full model (embedding + layers + final norm + LM head)

### Weight Conversion

```python
convert_hf_to_jax_weights(hf_model) -> Dict
```
Converts HuggingFace PyTorch Qwen checkpoint to JAX parameter dict.

---

## qwen2_jax_with_hooks.py

Extended Qwen model with activation extraction hooks for capturing intermediate layer outputs during forward passes.

### Activation Types

- `'residual'` (default): Final layer output after both residual connections
- `'mlp'`: MLP output before residual connection
- `'attn'`: Attention output before residual connection

### QwenModelWithActivations

```python
class QwenModelWithActivations(nn.Module):
    config: QwenConfig
    layers_to_extract: Optional[Tuple[int, ...]] = None
    activation_type: str = 'residual'

    def __call__(self, input_ids, attention_mask=None, kv_caches=None,
                 position_offset=0, return_activations=False)
    -> (logits, kv_caches) | (logits, kv_caches, activations_dict)
```

`layers_to_extract` is a static arg for JIT compilation — only specified layers capture activations, minimizing overhead.

```python
create_model_with_hooks(config, layers_to_extract, activation_type) -> QwenModelWithActivations
```

---

## multihost_extract.py

Main extraction script for TPU pod slices. Orchestrates multihost coordination, model loading, data pipeline, forward passes, and GCS upload.

### MultihostExtractionConfig

Covers all CLI arguments: topology, model path, dataset, prompt pipeline, extraction layers, batch size, GCS output, checkpointing, barrier sync.

### Key Pipelines

**Prompt pipeline** (`--pipeline prompt`): Standard prompts with data augmentation, dynamic length batching.

**Grid chunking pipeline** (`--pipeline grid_chunking`): Continuous token stream from grids only, fixed-size chunks. ~2 hours to build from 50K tasks.

### Activation Extraction (FSDP-aware)

The 3D mesh `(data, fsdp, model)` splits hidden_dim across the model axis. After forward pass:

1. Re-shard activations to `P('data', None, None)` — triggers all-gather on model axis to recover full hidden_dim
2. Deduplicate addressable shards (FSDP replicas share identical data)
3. Map local shards to global batch positions using shard index tuples (not host_id)

### Checkpoint/Resume

- Checkpoints saved after each shard upload to `gs://bucket/checkpoints/`
- On restart, skips already-processed samples
- Chunk cache saves the 2-hour grid chunking result to GCS

### Usage

```bash
python3 multihost_extract.py \
    --topology v5litepod-64 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://bucket/dataset.jsonl \
    --pipeline grid_chunking \
    --layers_to_extract 19 \
    --activation_type residual \
    --batch_size 16 \
    --max_seq_length 5120 \
    --upload_to_gcs \
    --gcs_bucket bucket-name \
    --gcs_prefix activations/run_001 \
    --enable_barrier_sync \
    --barrier_controller_host WORKER_0_IP
```

---

## extract_activations.py

Single-host extraction script. Simpler alternative to `multihost_extract.py` for v5e-8 or individual workers.

### ExtractionConfig

Worker ID auto-detected from `TPU_WORKER_ID` env var. Supports direct JSONL or sharded dataset with automatic shard claiming.

### Usage

```bash
export TPU_WORKER_ID=0
python extract_activations.py \
    --dataset_path data/stream_0.jsonl \
    --model_path Qwen/Qwen2.5-0.5B \
    --batch_size 4 \
    --layers_to_extract 0 1 2 12 19 \
    --activation_type residual \
    --gcs_bucket your-bucket \
    --upload_to_gcs
```

---

## convert_hf_to_arc_format.py

Converts HuggingFace datasets to ARC-format JSONL for extraction.

### Usage

```bash
python convert_hf_to_arc_format.py \
    --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
    --column_name examples \
    --output_file combined_50k.jsonl \
    --max_tasks 50000
```

### Output Format

```json
{"task_id": "task_00000001", "train": [{"input": [[...]], "output": [[...]]}], "test": [{"input": [[...]]}]}
```

Streams dataset from HuggingFace (memory-efficient). Splits each task's examples into train pairs (all but last) and test pair (last). Skips invalid tasks.
