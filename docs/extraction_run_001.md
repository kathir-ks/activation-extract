# Extraction Run 001: Layer 12, v1 Prompting, 50K Tasks

## Run Summary

| Parameter | Value |
|-----------|-------|
| Date | 2026-03-08 |
| Model | `KathirKs/qwen-2.5-0.5b` (Qwen 2.5 0.5B, fine-tuned on ARC) |
| TPU | `node-v5e-64-europe-west4-b` (v5litepod-64, 16 workers x 4 chips) |
| Layer | 12 (residual activations) |
| Prompt version | `output-from-examples-v1` |
| Dataset | `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` (first 50K tasks) |
| Predictions per task | 8 |
| Total sequences | 400,000 (50K tasks x 8 predictions) |
| Batch size | 32 (global across 16 hosts) |
| Max seq length | 2048 |
| Dtype | bfloat16 |
| Status | **Completed** |

## Output

| Metric | Value |
|--------|-------|
| GCS location | `gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/` |
| Shards per host | 43 |
| Total shards | 688 (43 x 16 hosts) |
| Total size | ~530 GB (compressed gzip) |
| Activations per host | 25,000 |
| Total activations | 400,000 |

## Dataset

- Source: `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` from HuggingFace
- Sliced to first 50,000 tasks
- Uploaded to: `gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl` (1.26 GB)
- Each task augmented with random flips, rotations, and color maps
- 8 predictions per task (across augmentation combinations)

## Prompting

Used `output-from-examples-v1` which has a minimal system prompt:
- System: "You are a helpful assistant." (~8 tokens)
- User: Task description + training examples + test input
- Assistant: "### Output\n\n```grid" (start of answer, ~10 tokens)

Note: `is_train_prompt=False` means sequences contain **only the prompt** (no generated answer). All extracted activations are from the model processing the prompt context.

## Configuration

```bash
python3 multihost_extract.py \
  --topology v5litepod-64 \
  --model_path KathirKs/qwen-2.5-0.5b \
  --dataset_path gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl \
  --max_tasks 50000 \
  --prompt_version output-from-examples-v1 \
  --predictions_per_task 8 \
  --layers_to_extract 12 \
  --activation_type residual \
  --batch_size 32 \
  --max_seq_length 2048 \
  --upload_to_gcs \
  --gcs_bucket arc-data-europe-west4 \
  --gcs_prefix activations/layer12_v1_50k \
  --shard_size_gb 1.0 \
  --delete_local_after_upload \
  --enable_barrier_sync \
  --barrier_controller_host 10.164.0.54 \
  --barrier_port 5555
```

## Known Issue: Sequence Truncation

The `max_seq_length=2048` setting causes right-truncation (`seq[:2048]`) for prompts exceeding 2048 tokens. The model supports up to 32,768 tokens (`max_position_embeddings`), so this is an artificial limit.

**Impact**: For long ARC prompts (large grids, many training examples), the test input at the end of the user message may be truncated. This means the model never sees what it's supposed to solve, and the extracted activations represent processing incomplete context rather than task-solving.

This issue is tracked in `docs/issue_dynamic_seq_length.md` and will be addressed in the `fix/dynamic-seq-length` branch.

## Shard Format

Each shard is a gzipped pickle file containing:
```python
{
    12: [  # layer index
        {
            'sample_idx': int,
            'activation': np.ndarray,  # [seq_len, hidden_dim] where hidden_dim=896
            'shape': tuple,
            'text_preview': str
        },
        ...  # ~585 samples per shard
    ]
}
```

## Checkpoint

GCS checkpoints at: `gs://arc-data-europe-west4/checkpoints/checkpoint_v5litepod-64_host_XX.json`

Example checkpoint (host_07):
```json
{
  "topology": "v5litepod-64",
  "host_id": 7,
  "last_processed_sample_idx": 399999,
  "total_samples_processed": 400000,
  "total_shards": 43,
  "total_activations": 25000,
  "dataset_path": "gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl",
  "model_path": "KathirKs/qwen-2.5-0.5b",
  "status": "completed"
}
```
