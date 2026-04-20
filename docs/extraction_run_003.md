# Extraction Run 003: Layers 15 + 22, Grid-Chunk, Batch 2

## Run Summary

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-20 (planned) |
| Model | `KathirKs/qwen-2.5-0.5b` (Qwen 2.5 0.5B, fine-tuned on ARC) |
| TPU | `node-v5e-64-europe-west4-b` (v5litepod-64, 16 workers x 4 chips) |
| Layers | **15, 22** (residual activations) |
| Pipeline | `grid_chunking` (chunk_size=5120) |
| Dataset | Tasks 50001-100000 from `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` |
| Predictions per task | 8 |
| Total sequences | ~400,000 (50K tasks x 8 predictions) |
| Batch size | 16 (global across 16 hosts) |
| Max seq length | 5120 (fixed chunks) |
| FSDP size | 4 (model_size=1, no hidden_dim splitting) |
| Status | **Planned** |

## Motivation

Previous runs extracted layers 12 and 19. This run adds coverage at two complementary positions:

- **Layer 15** (62% depth): Mid-late representations between layer 12 (early-mid) and layer 19 (late). Captures intermediate reasoning features before the model commits to output representations.
- **Layer 22** (92% depth): Near-output layer, two layers before the final. Captures the model's decision-making features, complementing layer 19's representations.

Using a fresh dataset (batch 2: tasks 50001-100000) ensures the extracted features generalize beyond the first 50K tasks used in prior runs.

## Dataset

| Parameter | Value |
|-----------|-------|
| Source | `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` |
| Slice | Tasks 50001-100000 (indices 50000-99999) |
| GCS location | `gs://arc-data-europe-west4/dataset_streams/combined_50k_batch2.jsonl` |
| Format | JSONL (ARC format: task_id, train, test) |

The first 50K tasks were used in Run 001 (layer 12) and Run 002 (layer 19). This batch uses the next 50K from the same 200K source.

**Dataset creation:**
```bash
bash scripts/create_dataset_batch2.sh
```

## Output

| Parameter | Value |
|-----------|-------|
| GCS location | `gs://arc-data-europe-west4/activations/layer15_22_gridchunk_50k_batch2/` |
| Checkpoint location | `gs://arc-data-europe-west4/checkpoints/gridchunk_layer15_22_batch2/` |
| Expected shards/host | ~85 (2 layers x ~43 shards per layer) |
| Expected total shards | ~1,360 |
| Expected total size | ~250 GB (compressed) |
| Shard size | 1 GB (uncompressed) |

## Memory Optimization

| Setting | Value | Why |
|---------|-------|-----|
| `fsdp_size=4` | All 4 local devices on FSDP axis | `model_size=1` means full 896-dim on each host — no hidden_dim splitting, no all-gather needed |
| `batch_size=16` | 1 sample per host | 2 layers x 5120 tokens x 896 dim x 2 bytes (bf16) = ~18 MB per sample per layer — fits comfortably in 16 GB HBM per chip |
| `shard_size_gb=1.0` | Flush to GCS at 1 GB | Keeps peak memory bounded; ~835 activations per shard |
| `delete_local_after_upload` | Free disk after upload | Workers have limited SSD |
| `grid_chunking` | Fixed 5120-token chunks | No variable-length padding waste; every token is grid content |

## Configuration

```bash
python3 multihost_extract.py \
    --topology v5litepod-64 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://arc-data-europe-west4/dataset_streams/combined_50k_batch2.jsonl \
    --max_tasks 50000 \
    --pipeline grid_chunking \
    --predictions_per_task 8 \
    --layers_to_extract 15 22 \
    --activation_type residual \
    --batch_size 16 \
    --max_seq_length 5120 \
    --fsdp_size 4 \
    --upload_to_gcs \
    --gcs_bucket arc-data-europe-west4 \
    --gcs_prefix activations/layer15_22_gridchunk_50k_batch2 \
    --shard_size_gb 1.0 \
    --delete_local_after_upload \
    --enable_barrier_sync \
    --barrier_controller_host <worker_0_ip> \
    --barrier_port 5555 \
    --checkpoint_gcs_prefix checkpoints/gridchunk_layer15_22_batch2 \
    --verbose
```

## Running

**Step 1: Create and upload dataset (run once)**
```bash
bash scripts/create_dataset_batch2.sh
```

**Step 2: Launch extraction (with preemption recovery)**
```bash
nohup bash scripts/launch_extraction_run003.sh > launch_run003.log 2>&1 &
```

**Monitor:**
```bash
tail -f launch_run003.log
```

## Shard Format

Same as Run 002. Each shard is a gzipped pickle containing:
```python
{
    15: [  # layer 15
        {
            'sample_idx': int,
            'activation': np.ndarray,  # [seq_len, 896], bfloat16
            'shape': tuple,
            'text_preview': str
        },
        ...
    ],
    22: [  # layer 22
        { ... },
        ...
    ]
}
```

## Comparison with Previous Runs

| | Run 001 | Run 002 | **Run 003** |
|---|---------|---------|-------------|
| Layers | 12 | 19 | **15, 22** |
| Dataset | Batch 1 (0-50K) | Batch 1 (0-50K) | **Batch 2 (50K-100K)** |
| Pipeline | prompt (v1, truncated) | grid_chunking | **grid_chunking** |
| Max seq len | 2048 | 5120 | **5120** |
| FSDP size | N/A | 4 | **4** |
| Hidden dim | 896 | 896 (after merge) | **896 (native)** |
