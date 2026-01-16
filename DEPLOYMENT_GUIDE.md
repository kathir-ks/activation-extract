# Activation Extraction Deployment Guide

## Overview

This guide covers deploying the activation extraction system to multiple TPU workers for large-scale parallel processing.

## System Architecture

```
HuggingFace Dataset
       ↓
create_dataset_streams.py → Split into N independent streams
       ↓
GCS: gs://bucket/datasets/stream_000.jsonl, stream_001.jsonl, ...
       ↓
       ├→ TPU Worker 0 → extract_activations.py → gs://bucket/activations/tpu_0/
       ├→ TPU Worker 1 → extract_activations.py → gs://bucket/activations/tpu_1/
       ├→ TPU Worker 2 → extract_activations.py → gs://bucket/activations/tpu_2/
       └→ ... (embarrassingly parallel, no communication)
```

## Prerequisites

1. **GCP Project with TPU access**
   - TPU quota allocated (v2-8, v3-8, v4-8, or v5e-8)
   - Preemptible TPUs recommended for cost savings

2. **GCS Bucket**
   - Create bucket in same region as TPUs
   - Example: `gsutil mb -l us-central1 gs://my-activations-bucket`

3. **Local Setup**
   - Python 3.10+
   - Dependencies installed: `pip install -r requirements.txt`
   - GCloud SDK configured: `gcloud auth login`

## Verification (REQUIRED BEFORE DEPLOYMENT)

### 1. Verify Tokenization Pipeline

Review the formatted prompts and tokenization:

```bash
# Check tokenization output
cat tokenization_verification.txt

# Verify:
# - Prompt structure follows ARC format
# - Grid encoding is correct (```grid format)
# - System/user/assistant roles are proper
# - Special tokens handled correctly
```

### 2. Verify Model Implementation

Review the model generation outputs:

```bash
# Check model generation
cat model_generation_verification.txt

# Verify:
# - Model generates coherent grid outputs
# - Output format matches expectations (```grid format)
# - No NaN or numerical errors
# - Generated grids follow ARC patterns
```

**IMPORTANT:** Review both files before proceeding to production deployment. If you see any issues with tokenization or model generation, stop and fix them first.

## Quick Start Deployment

### Option 1: Automated Full Deployment (Recommended)

Use the all-in-one deployment script:

```bash
# Deploy to 8 TPU workers
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --num_workers 8 \
  --model Qwen/Qwen2.5-0.5B

# Test with limited data first
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --num_workers 2 \
  --max_samples 100
```

This script handles:
- ✓ Dataset stream creation and upload
- ✓ Code deployment to TPU VMs
- ✓ Extraction launch on all workers
- ✓ Automatic checkpoint/resume

### Option 2: Manual Step-by-Step

For more control, follow these steps:

#### Step 1: Prepare Dataset Streams

```bash
# Create N independent dataset streams
python create_dataset_streams.py \
  --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
  --num_streams 8 \
  --output_dir ./dataset_streams \
  --verbose

# Upload to GCS
./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-us-central1 \
  --prefix datasets \
  --local_dir ./dataset_streams
```

#### Step 2: Deploy Code to TPU Workers

```bash
# Assuming TPUs named tpu-worker-0, tpu-worker-1, ...
for i in {0..7}; do
  gcloud compute scp --recurse \
    ./*.py ./arc24 ./core ./scripts \
    tpu-worker-$i:~/qwen/ \
    --zone=us-central1-a &
done
wait
```

#### Step 3: Launch Extraction on Each Worker

```bash
# Launch on each TPU
for i in {0..7}; do
  gcloud compute ssh tpu-worker-$i --zone=us-central1-a --command="
    cd ~/qwen
    nohup bash scripts/run_extraction_worker.sh \
      --gcs_bucket fineweb-data-us-central1 \
      --dataset_stream gs://fineweb-data-us-central1/datasets/stream_$(printf '%03d' $i).jsonl \
      --model Qwen/Qwen2.5-0.5B \
      --gcs_prefix activations \
      > extraction.log 2>&1 &
  " &
  sleep 2  # Stagger starts
done
```

## Configuration Options

### Model Selection

```bash
# Qwen 2.5 models
--model Qwen/Qwen2.5-0.5B    # 0.5B parameters (default, fast)
--model Qwen/Qwen2.5-1.5B    # 1.5B parameters
--model Qwen/Qwen2.5-7B      # 7B parameters
```

### Layer Selection

```bash
# Extract specific layers (default: all layers)
--layers "0 1 2 3 4 5"              # First 6 layers
--layers "18 19 20 21 22 23"        # Last 6 layers
--layers "0 5 10 15 20 23"          # Sparse sampling
```

### Performance Tuning

```bash
# Batch size (affects memory and speed)
--batch_size 4       # Default, ~6GB memory
--batch_size 8       # Higher throughput, ~10GB memory
--batch_size 1       # Minimal memory

# Shard size (affects GCS object count)
--shard_size_gb 1.0  # Default, ~1GB per shard
--shard_size_gb 2.0  # Fewer, larger files

# Sequence length
--max_seq_length 2048   # Default
--max_seq_length 4096   # For longer prompts
```

## Monitoring

### Check Worker Status

```bash
# View real-time logs
gcloud compute ssh tpu-worker-0 --zone=us-central1-a \
  --command='tail -f ~/qwen/extraction.log'

# Check checkpoint status
gcloud compute ssh tpu-worker-0 --zone=us-central1-a \
  --command='cat ~/qwen/checkpoints/checkpoint_worker_0.json'
```

### Monitor GCS Uploads

```bash
# List all worker outputs
gsutil ls gs://fineweb-data-us-central1/activations/

# Count files per worker
gsutil ls gs://fineweb-data-us-central1/activations/tpu_0/ | wc -l

# Check total size
gsutil du -sh gs://fineweb-data-us-central1/activations/
```

### Check All Workers

```bash
# Check all checkpoint statuses
for i in {0..7}; do
  echo "=== Worker $i ==="
  gcloud compute ssh tpu-worker-$i --zone=us-central1-a \
    --command='cat ~/qwen/checkpoints/checkpoint_worker_*.json' 2>/dev/null || echo "No checkpoint"
done
```

## Fault Tolerance

### Preemption Handling

The system automatically handles TPU preemptions:

1. **Checkpoint saved** every N samples (configurable)
2. **On preemption**, checkpoint contains:
   - `last_processed_sample_idx`: Resume point
   - `total_samples_processed`: Progress counter
   - `total_shards`: Number of shards created
   - `status`: "in_progress" or "completed"

3. **On restart**, extraction resumes from checkpoint automatically

### Manual Resume

If a worker stops:

```bash
# Re-run the same command
gcloud compute ssh tpu-worker-0 --zone=us-central1-a --command="
  cd ~/qwen
  nohup bash scripts/run_extraction_worker.sh \
    --gcs_bucket fineweb-data-us-central1 \
    --dataset_stream gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
    --model Qwen/Qwen2.5-0.5B \
    > extraction.log 2>&1 &
"
```

The script will detect the checkpoint and resume.

## Output Structure

```
gs://bucket/activations/
├── tpu_0/
│   ├── shard_0001.pkl.gz  (~1GB, layers 0-1)
│   ├── shard_0002.pkl.gz  (~1GB, layers 2-3)
│   ├── shard_0003.pkl.gz  (~1GB, layers 4-5)
│   └── ...
├── tpu_1/
│   ├── shard_0001.pkl.gz
│   └── ...
└── tpu_N/
    └── ...
```

### Shard Format

Each shard is a pickled dictionary:

```python
{
  0: [  # Layer 0
    {
      'sample_idx': 0,
      'activation': np.ndarray,  # shape: [seq_len, hidden_size]
      'shape': (seq_len, hidden_size),
      'text_preview': 'Task: task_00000000, Prompt: ...'
    },
    ...
  ],
  1: [  # Layer 1
    ...
  ]
}
```

## Performance Estimates

Based on Qwen 2.5-0.5B with 2048 max sequence length:

| Workers | Samples/Worker | Total Samples | Est. Time | Est. Storage |
|---------|----------------|---------------|-----------|--------------|
| 8       | 25,000        | 200,000       | ~4 days   | ~3.2 TB      |
| 16      | 12,500        | 200,000       | ~2 days   | ~3.2 TB      |
| 32      | 6,250         | 200,000       | ~1 day    | ~3.2 TB      |

Assumptions:
- ~15 seconds per sample (including JIT)
- ~16MB per sample activation data (all 24 layers)
- No downtime from preemptions

## Cost Estimation

Using preemptible TPU v3-8 in us-central1:

- **Compute**: ~$1.35/hour per TPU × 8 TPUs × 96 hours = **~$1,040**
- **Storage**: ~3.2 TB × $0.020/GB/month = **~$64/month**
- **Network**: Negligible (within same region)

**Total cost**: ~$1,040 compute + ~$64/month storage

💡 **Tip**: Use preemptible TPUs to save ~70% on compute costs

## Troubleshooting

### Worker Not Processing

```bash
# Check if extraction is running
gcloud compute ssh tpu-worker-0 --zone=us-central1-a \
  --command='ps aux | grep extract_activations'

# Check for errors in log
gcloud compute ssh tpu-worker-0 --zone=us-central1-a \
  --command='tail -100 ~/qwen/extraction.log'
```

### Out of Memory Errors

Reduce batch size:
```bash
--batch_size 2   # or even 1
```

### Slow Processing

- Check if HuggingFace dataset download is slow
- Verify TPU is not preempting frequently
- Consider using local SSD for temporary storage

### GCS Upload Failures

```bash
# Check GCS permissions
gsutil ls gs://fineweb-data-us-central1/

# Retry with manual upload
gcloud compute ssh tpu-worker-0 --zone=us-central1-a --command="
  gsutil -m cp -r ~/qwen/activations/*.pkl.gz \
    gs://fineweb-data-us-central1/activations/tpu_0/
"
```

## Best Practices

1. **Always test with small dataset first**
   ```bash
   --max_samples 100
   ```

2. **Monitor first worker before scaling**
   - Verify correct output format
   - Check GCS uploads working
   - Confirm checkpoint/resume works

3. **Use separate GCS prefix per experiment**
   ```bash
   --gcs_prefix "experiment_2024_01_15"
   ```

4. **Keep logs**
   - Save extraction logs for debugging
   - Monitor Cloud Logging for TPU events

5. **Verify outputs periodically**
   ```bash
   # Download and inspect a shard
   gsutil cp gs://bucket/activations/tpu_0/shard_0001.pkl.gz ./
   python -c "
   import gzip, pickle
   with gzip.open('shard_0001.pkl.gz', 'rb') as f:
       data = pickle.load(f)
       print(f'Layers: {list(data.keys())}')
       print(f'Samples in layer 0: {len(data[0])}')
       print(f'Activation shape: {data[0][0][\"shape\"]}')
   "
   ```

## Support

For issues or questions:
1. Check logs: `extraction.log` on TPU worker
2. Review checkpoint: `checkpoints/checkpoint_worker_N.json`
3. Verify GCS uploads: `gsutil ls gs://bucket/activations/tpu_N/`
4. Test locally first with `--max_samples 10`
