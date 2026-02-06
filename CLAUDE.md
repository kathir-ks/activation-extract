# Activation Extraction Project

## Purpose

This project extracts neural network activations from Qwen 2.5 models for training Sparse Autoencoders (SAEs). The extracted activations capture intermediate representations from transformer layers, which can be used to study model internals and train interpretability tools.

## Key Design Decisions

### Forward-Pass Only (No Generation)
The main optimization is using **single forward passes** instead of autoregressive generation:
- 10-100x faster than generation-based extraction
- No KV caching needed during extraction
- True batching across sequences
- Parallel processing across TPU cores

### JAX/Flax Implementation
The model is re-implemented in JAX/Flax for TPU optimization:
- [qwen2_jax.py](qwen2_jax.py) - Core Qwen 2.5 model implementation
- [qwen2_jax_with_hooks.py](qwen2_jax_with_hooks.py) - Model with activation extraction hooks
- Weights converted from HuggingFace PyTorch checkpoints

### Activation Types
Three extraction modes available via `--activation_type`:
- `residual` (default): Final layer output after both residual connections
- `mlp`: MLP output before residual connection
- `attn`: Attention output before residual connection

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TPU Workers                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Worker 0│  │ Worker 1│  │ Worker 2│  │ Worker N│  ...       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
│       ▼            ▼            ▼            ▼                   │
│  ┌─────────────────────────────────────────────────┐            │
│  │         extract_activations.py                   │            │
│  │  - Load dataset stream (JSONL)                   │            │
│  │  - Create prompts with data augmentation         │            │
│  │  - Forward pass through JAX model                │            │
│  │  - Extract layer activations                     │            │
│  │  - Save shards to local / upload to GCS          │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Google Cloud   │
                    │    Storage      │
                    │                 │
                    │ gs://bucket/    │
                    │   activations/  │
                    │     tpu_0/      │
                    │     tpu_1/      │
                    │     ...         │
                    └─────────────────┘
```

## Data Flow

1. **Dataset Loading** ([core/dataset_utils.py](core/dataset_utils.py))
   - Load ARC-format JSONL files
   - Support for sharded datasets with automatic shard claiming
   - Machine-based round-robin sharding for multi-worker setups

2. **Prompt Creation** (uses [arc24/](arc24/) modules)
   - Apply data augmentation (rotations, flips, color maps)
   - Create prompts using grid encoders
   - Multiple predictions per task for diversity

3. **Model Forward Pass** ([qwen2_jax_with_hooks.py](qwen2_jax_with_hooks.py))
   - Tokenize prompts
   - Batch sequences with padding
   - Single forward pass extracts activations from specified layers
   - JIT-compiled for performance

4. **Storage** ([core/activation_storage.py](core/activation_storage.py))
   - Buffer activations in memory
   - Auto-shard when buffer exceeds size threshold (default 1GB)
   - Compress with gzip
   - Upload to GCS with fsspec

## Key Files

| File | Purpose |
|------|---------|
| [extract_activations.py](extract_activations.py) | Single-host extraction script |
| [multihost_extract.py](multihost_extract.py) | **Multi-host TPU extraction (v5litepod-64)** |
| [qwen2_jax.py](qwen2_jax.py) | JAX implementation of Qwen 2.5 |
| [qwen2_jax_with_hooks.py](qwen2_jax_with_hooks.py) | Model with activation hooks |
| [core/activation_storage.py](core/activation_storage.py) | Shard storage and GCS upload |
| [core/dataset_utils.py](core/dataset_utils.py) | Dataset loading utilities |
| [core/jax_utils.py](core/jax_utils.py) | JAX/TPU utilities (mesh, sharding) |
| [core/barrier_sync.py](core/barrier_sync.py) | **Socket-based barrier synchronization** |
| [scripts/manage_tpus.sh](scripts/manage_tpus.sh) | TPU lifecycle management |
| [scripts/launch_multihost_sync.sh](scripts/launch_multihost_sync.sh) | Launch multihost extraction with barrier sync |
| [create_dataset_streams.py](create_dataset_streams.py) | Create per-worker dataset streams |

## Multihost TPU Architecture (v5litepod-64)

The multihost extraction system handles TPU pods with 16 workers (4 TPU chips each):

```
┌─────────────────────────────────────────────────────────────────┐
│                    TPU Pod (v5litepod-64)                        │
│                                                                  │
│  ┌─────────┐  ┌─────────┐      ┌─────────┐  ┌─────────┐        │
│  │Worker 0 │  │Worker 1 │ ...  │Worker 14│  │Worker 15│        │
│  │ 4 chips │  │ 4 chips │      │ 4 chips │  │ 4 chips │        │
│  └────┬────┘  └────┬────┘      └────┬────┘  └────┬────┘        │
│       │            │                 │            │              │
│       ▼            ▼                 ▼            ▼              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │            Barrier Sync Server (Worker 0)             │       │
│  │   Coordinates all workers at synchronization points   │       │
│  └──────────────────────────────────────────────────────┘       │
│                              │                                   │
│       ┌──────────────────────┼──────────────────────┐           │
│       ▼                      ▼                      ▼           │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐       │
│  │  init   │──────────▶│ model   │──────────▶│  start  │       │
│  │ barrier │           │ loaded  │           │ extract │       │
│  └─────────┘           └─────────┘           └─────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Socket Barrier Synchronization

The barrier sync system ([core/barrier_sync.py](core/barrier_sync.py)) solves the problem of staggered SSH connections causing "unexpected peer in launch group" JAX errors:

```python
# Barrier sync points in multihost_extract.py:
# 1. init             - After JAX initialization  
# 2. model_loaded     - After model weights loaded
# 3. sharding_complete - After parameters sharded
# 4. extraction_start  - Before processing batches
# 5. final_checkpoint  - After last batch
# 6. complete          - Before shutdown
```

**How it works:**
1. Worker 0 starts a TCP barrier server
2. All workers connect as clients
3. At each barrier point, workers wait until ALL workers arrive
4. Server releases all workers simultaneously

### Running Multihost Extraction

```bash
# From control machine, SSH to all workers:
gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=all \
    --command="cd ~/activation-extract && python3 multihost_extract.py \
        --topology v5litepod-64 \
        --dataset_path gs://bucket/datasets/stream.jsonl \
        --gcs_bucket bucket \
        --gcs_prefix activations/run_001 \
        --batch_size 32 \
        --upload_to_gcs \
        --enable_barrier_sync \
        --barrier_controller_host WORKER_0_IP \
        --barrier_port 5555"
```

### Current Status (Feb 2026)

**Working:**
- Single-host extraction on v5-8, v5e-8
- Model loading and weight conversion
- Activation extraction and GCS upload
- Checkpoint/resume system

**In Progress:**
- Multihost extraction on v5litepod-64
- Socket barrier synchronization to fix JAX peer sync issues
- Issue: Workers detect host_id from JAX process_index() after JAX init, but barrier sync needs to know which worker is "worker 0" to start the server

**Known Issues:**
- JAX's process_index() not available before distributed initialization
- SSH timing causes workers to start at different times
- Need to determine worker 0 IP before JAX init for barrier server

## Running Extraction

### Single Worker (Local or Single TPU)
```bash
python extract_activations.py \
    --dataset_path data/stream_0.jsonl \
    --model_path Qwen/Qwen2.5-0.5B \
    --batch_size 4 \
    --layers_to_extract 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
    --activation_type residual \
    --gcs_bucket your-bucket \
    --upload_to_gcs
```

### Parallel Independent Workers (Recommended)
Each worker processes its own dataset stream independently:
```bash
# On TPU worker 0
export TPU_WORKER_ID=0
python extract_activations.py \
    --dataset_path data/stream_0.jsonl \
    --gcs_bucket your-bucket \
    --upload_to_gcs

# On TPU worker 1
export TPU_WORKER_ID=1
python extract_activations.py \
    --dataset_path data/stream_1.jsonl \
    --gcs_bucket your-bucket \
    --upload_to_gcs
```

### Multi-Host TPU (v5e-64)
```bash
python extract_activations.py \
    --use_sharded_dataset \
    --sharded_dataset_dir gs://bucket/sharded_dataset \
    --multihost --num_hosts 4 --host_id 0 \
    --coordinator_address "10.0.0.1:8476" \
    --gcs_bucket your-bucket
```

## TPU Management

The project includes scripts for managing preemptible TPUs:

```bash
# Create TPUs across zones
./scripts/manage_tpus.sh create --zones us-central1-a,us-central1-b --workers_per_zone 4

# Check status
./scripts/manage_tpus.sh status --zones us-central1-a,us-central1-b

# Recreate preempted TPUs
./scripts/manage_tpus.sh recreate-preempted --zones us-central1-a --workers_per_zone 4
```

## Checkpoint/Resume

The system supports checkpoint/resume for handling TPU preemption:
- Checkpoints saved after each shard upload
- Resume from last completed sample on restart
- Checkpoint files stored in `./checkpoints/checkpoint_worker_{id}.json`

## Output Format

Activations are saved as gzipped pickle shards:
```
gs://bucket/activations/tpu_0/
├── shard_0001.pkl.gz
├── shard_0002.pkl.gz
├── ...
└── metadata.json
```

Each shard contains:
```python
{
    layer_idx: [
        {
            'sample_idx': int,
            'activation': np.ndarray,  # [seq_len, hidden_dim]
            'shape': tuple,
            'text_preview': str
        },
        ...
    ]
}
```

## Archived Code

The `archived_old/` directory contains previous iterations:
- `extraction_scripts/` - Earlier extraction implementations
- `test_files/` - Test scripts
- `verification_scripts/` - Model verification tools
- `shell_scripts/` - Legacy deployment scripts
