# Activation Extraction for SAE Training

Extract layer activations from Qwen 2.5 models on TPU pods for Sparse Autoencoder (SAE) training. Designed for ARC-AGI grid data on preemptible TPU pods with automatic recovery.

## Current Setup

| Component | Value |
|-----------|-------|
| Model | Qwen 2.5 0.5B (24 layers, hidden_size=896) |
| TPU | v5litepod-64 (16 workers x 4 chips = 64 devices) |
| Pipeline | Grid chunking (grids-only, no prompt text) |
| Chunk size | 5120 tokens |
| Batch size | 16 (global across all hosts) |
| Dataset | 50K ARC tasks (JSONL on GCS) |
| Output | Gzipped pickle shards uploaded to GCS |

## Quick Start

See **[RUNBOOK.md](RUNBOOK.md)** for step-by-step instructions to launch an extraction run.

## How It Works

### Grid Chunking Pipeline

Instead of full prompts (system prompt + instructions + grids), the grid chunking pipeline:

1. Strips all text/instructions
2. Encodes only grid data (train inputs, train outputs, test inputs) using `GridShapeEncoder`
3. Applies data augmentation (rotations, flips, color maps) for diversity
4. Concatenates all grid tokens into a single continuous stream
5. Splits the stream into fixed 5120-token chunks

This maximizes token utilization for SAE training -- every token is actual grid content.

### Multihost TPU Architecture

The v5litepod-64 has 16 SSH-accessible workers, each with 4 TPU chips. JAX SPMD coordinates all 64 devices as a single mesh:

```
Control Machine (you)
  |
  |  nohup scripts/launch_extraction.sh
  |
  v
TPU Pod (v5litepod-64) -- 16 workers, 64 chips
  |
  |-- Worker 0  (barrier sync server)
  |-- Worker 1  (JAX primary, process_index=0)
  |-- Worker 2
  |-- ...
  |-- Worker 15
  |
  |-- All workers run multihost_extract.py independently
  |-- JAX SPMD shards batches across all 64 devices
  |-- Socket barrier sync coordinates startup
  |
  v
GCS: gs://bucket/activations/host_*/shard_*.pkl.gz
GCS: gs://bucket/checkpoints/*.json (survive preemption)
GCS: gs://bucket/checkpoints/grid_chunks_*.pkl.gz (chunk cache)
```

### Preemption Recovery

The resilient launcher (`scripts/launch_extraction.sh`) runs on the control machine under `nohup` and handles the full lifecycle:

1. Deploys code + deps to all 16 workers
2. Launches `multihost_extract.py` on all workers
3. Polls every 5 minutes for TPU health
4. On preemption: waits for TPU recreation, re-deploys, relaunches
5. Extraction resumes from GCS checkpoint automatically
6. **Chunk cache** skips the ~2 hour data pipeline on restart

### Chunk Caching

Building the grid token stream from 50K tasks takes ~2 hours. On the first run, the computed chunks are saved to GCS as a gzipped pickle. On restart (after preemption), all workers load from cache in seconds instead of recomputing.

Cache is keyed on: sorted task IDs + chunk_size + predictions_per_task + random_seed. Any config change invalidates the cache automatically.

## Key Files

| File | Purpose |
|------|---------|
| `multihost_extract.py` | Main extraction script for TPU pods |
| `extract_activations.py` | Single-host extraction (testing/small runs) |
| `scripts/launch_extraction.sh` | Resilient launcher with preemption recovery |
| `core/grid_chunking.py` | Grid chunking pipeline + chunk caching |
| `core/jax_utils.py` | JAX/TPU mesh, sharding, multihost init |
| `core/barrier_sync.py` | Socket-based barrier for worker synchronization |
| `core/activation_storage.py` | Activation buffering, sharding, GCS upload |
| `core/dataset_utils.py` | Dataset loading and prompt creation |
| `qwen2_jax.py` | Qwen 2.5 JAX/Flax implementation |
| `qwen2_jax_with_hooks.py` | Model with activation extraction hooks |

## Extraction Parameters

### `multihost_extract.py` key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--topology` | `v5e-64` | TPU pod topology |
| `--model_path` | `Qwen/Qwen2.5-0.5B` | HuggingFace model ID |
| `--dataset_path` | | Path to JSONL dataset (local or gs://) |
| `--max_tasks` | all | Limit number of tasks to process |
| `--pipeline` | `prompt` | `prompt` or `grid_chunking` |
| `--layers_to_extract` | all | Space-separated layer indices |
| `--activation_type` | `residual` | `residual`, `mlp`, or `attn` |
| `--batch_size` | 32 | Global batch size across all hosts |
| `--max_seq_length` | 5120 | Fixed chunk/sequence length |
| `--gcs_bucket` | | GCS bucket for activation uploads |
| `--gcs_prefix` | `activations/multihost` | GCS path prefix |
| `--checkpoint_gcs_prefix` | `checkpoints` | GCS prefix for checkpoints + chunk cache |
| `--barrier_controller_host` | auto | Worker 0 IP for barrier sync |

### Activation Types

- `residual` (default): Final layer output after both attention and MLP residual connections
- `mlp`: MLP output before residual connection
- `attn`: Attention output before residual connection

## Output Format

Activations are saved as gzipped pickle shards (~1 GB each):

```
gs://bucket/activations/layer19_gridchunk_50k_v5litepod-64/
  host_00/shard_0001.pkl.gz
  host_00/shard_0002.pkl.gz
  ...
  host_15/shard_0001.pkl.gz
  ...
```

Each shard:

```python
import pickle, gzip

with gzip.open('shard_0001.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# data = {
#     19: [  # Layer index
#         {
#             'sample_idx': 0,
#             'activation': np.ndarray,  # shape: (5120, 896)
#             'shape': (5120, 896),
#             'text_preview': 'grid_chunk_0'
#         },
#         ...
#     ]
# }
```

## Monitoring

```bash
# Watch launcher log
tail -f launch.log

# Check worker logs (SSH into any worker)
gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=0 \
    --command="tail -50 ~/activation-extract/extraction.log"

# Check GCS output
gsutil ls gs://bucket/activations/layer19_gridchunk_50k_v5litepod-64/host_00/

# Count shards across all hosts
gsutil ls gs://bucket/activations/layer19_gridchunk_50k_v5litepod-64/host_*/*.pkl.gz | wc -l

# Check checkpoint status
gsutil cat gs://bucket/checkpoints/gridchunk_layer19/checkpoint_v5litepod-64_host_00.json | python3 -m json.tool
```

## Archived

Old implementations (independent parallel workers, old deployment scripts, old docs) are in `archived_old/`. The current approach uses multihost JAX SPMD on a single TPU pod instead.
