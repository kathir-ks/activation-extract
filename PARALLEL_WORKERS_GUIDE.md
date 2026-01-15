# Parallel Independent Workers Guide

This guide explains how to use the massively parallel, independent single-host worker architecture for activation extraction on pre-emptible TPUs.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  barc0/200k_HEAVY dataset               │
│  Split into N streams (32-64)           │
└──────────────┬──────────────────────────┘
               │
       ┌───────┼───────┬─────────┬────────┐
       │       │       │         │        │
    Stream0 Stream1 Stream2 ... StreamN
       │       │       │         │        │
       ▼       ▼       ▼         ▼        ▼
   ┌─────┐ ┌─────┐ ┌─────┐   ┌─────┐
   │TPU 0│ │TPU 1│ │TPU 2│...│TPU N│
   │(v4-8)│(v4-8)│(v4-8)│   │(v4-8)│
   └──┬──┘ └──┬──┘ └──┬──┘   └──┬──┘
      │       │       │         │
      │ Periodic GCS Upload     │
      │ + Checkpoint saving     │
      ▼       ▼       ▼         ▼
   gs://bucket/activations/
      ├── tpu_0/
      │   ├── shard_0001.pkl.gz
      │   ├── shard_0002.pkl.gz
      │   └── metadata.json
      ├── tpu_1/
      │   ├── shard_0001.pkl.gz
      │   └── ...
      └── ...

   ./checkpoints/
      ├── checkpoint_worker_0.json
      ├── checkpoint_worker_1.json
      └── ...
```

## Key Features

✅ **Independent Workers** - No coordination needed between TPUs
✅ **Checkpoint/Resume** - Automatic recovery from pre-emption
✅ **Per-TPU GCS Folders** - No conflicts, organized storage
✅ **Periodic Upload** - Upload every ~1GB to avoid data loss
✅ **All Layers Extraction** - Extracts from all 24 layers by default
✅ **Single-Host Only** - No complex multi-host coordination

## Quick Start

### Step 1: Create Dataset Streams

Split the HuggingFace dataset into N independent streams:

```bash
# For 32 TPUs (processing 200k samples)
python create_dataset_streams.py \
    --num_streams 32 \
    --output_dir ./data/streams \
    --max_samples 200000

# For 64 TPUs
python create_dataset_streams.py \
    --num_streams 64 \
    --output_dir ./data/streams \
    --max_samples 200000

# This creates:
# ./data/streams/stream_000.jsonl
# ./data/streams/stream_001.jsonl
# ...
# ./data/streams/stream_031.jsonl
```

**Note:** This only needs to be run **once**. All workers will read from these pre-created files.

### Step 2: Launch Workers

On each TPU VM:

```bash
# Set worker ID (0 to N-1)
export TPU_WORKER_ID=5

# Set GCS bucket for upload
export GCS_BUCKET=your-bucket-name
export UPLOAD_TO_GCS=true

# Launch worker
./launch_worker.sh
```

Or pass worker ID as argument:

```bash
export GCS_BUCKET=your-bucket-name
export UPLOAD_TO_GCS=true
./launch_worker.sh 5
```

### Step 3: Monitor Progress

Each worker will:
- Process its assigned stream independently
- Save checkpoints after each shard (~1GB)
- Upload shards to `gs://bucket/activations/tpu_N/`
- Print progress to stdout

If a worker is pre-empted, simply restart it:

```bash
export TPU_WORKER_ID=5
./launch_worker.sh
# Will resume from checkpoint automatically
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TPU_WORKER_ID` | (required) | Worker ID (0 to N-1) |
| `GCS_BUCKET` | - | GCS bucket for uploads |
| `UPLOAD_TO_GCS` | false | Enable GCS upload |
| `MODEL_PATH` | Qwen/Qwen2.5-0.5B | HuggingFace model path |
| `DATASET_DIR` | ./data/streams | Directory with stream files |
| `OUTPUT_DIR` | ./activations/worker_N | Local output directory |
| `CHECKPOINT_DIR` | ./checkpoints | Checkpoint directory |
| `BATCH_SIZE` | 4 | Batch size per device |
| `SHARD_SIZE_GB` | 1.0 | Target shard size in GB |
| `MAX_SEQ_LENGTH` | 2048 | Maximum sequence length |

### Command-Line Arguments

For advanced usage, call `extract_activations.py` directly:

```bash
python extract_activations.py \
    --worker_id 5 \
    --dataset_path data/streams/stream_005.jsonl \
    --model_path Qwen/Qwen2.5-0.5B \
    --output_dir ./activations \
    --checkpoint_dir ./checkpoints \
    --enable_checkpointing \
    --upload_to_gcs \
    --gcs_bucket your-bucket \
    --batch_size 4 \
    --shard_size_gb 1.0 \
    --max_seq_length 2048 \
    --verbose
```

## Checkpoint & Resume

### How It Works

1. **Automatic Checkpointing**: After each shard is saved to disk and uploaded to GCS, a checkpoint file is saved to `./checkpoints/checkpoint_worker_N.json`

2. **Resume on Restart**: When a worker restarts, it:
   - Checks for existing checkpoint
   - Skips already-processed samples
   - Continues from where it left off

3. **Checkpoint Contents**:
```json
{
  "worker_id": 5,
  "last_processed_sample_idx": 1250,
  "total_samples_processed": 1251,
  "total_shards": 3,
  "dataset_path": "data/streams/stream_005.jsonl",
  "model_path": "Qwen/Qwen2.5-0.5B",
  "status": "in_progress"
}
```

### Handling Pre-emption

When a TPU is pre-empted:

1. Worker dies (may lose data since last checkpoint)
2. Last checkpoint preserved on disk
3. Restart worker with same `TPU_WORKER_ID`
4. Worker resumes from last checkpoint
5. Only re-processes samples since last shard upload

**Data Loss Window**: At most 1 shard worth of data (~1GB or ~100-500 samples depending on sequence length)

### Disabling Checkpointing

If you don't want checkpointing:

```bash
python extract_activations.py \
    --no_checkpointing \
    --dataset_path data/streams/stream_005.jsonl \
    ...
```

## Output Structure

### Local Output

```
./activations/worker_5/
├── shard_0001.pkl.gz    # First shard (~1GB compressed)
├── shard_0002.pkl.gz    # Second shard
├── shard_0003.pkl.gz
└── metadata.json        # Metadata for all shards

./checkpoints/
└── checkpoint_worker_5.json
```

### GCS Output

```
gs://your-bucket/activations/
├── tpu_0/
│   ├── shard_0001.pkl.gz
│   ├── shard_0002.pkl.gz
│   └── metadata.json
├── tpu_1/
│   ├── shard_0001.pkl.gz
│   └── ...
├── tpu_2/
│   └── ...
└── ...
```

Each `tpu_N/` folder contains activations from that specific worker - no conflicts between workers.

## Activation Data Format

Each shard contains activations for all layers:

```python
import pickle
import gzip

# Load a shard
with gzip.open('shard_0001.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# Structure:
# data = {
#     0: [  # Layer 0
#         {'sample_idx': 0, 'activation': np.array(...), 'shape': (seq_len, hidden_size), 'text_preview': '...'},
#         {'sample_idx': 1, 'activation': np.array(...), ...},
#         ...
#     ],
#     1: [  # Layer 1
#         ...
#     ],
#     ...
#     23: [  # Layer 23
#         ...
#     ]
# }
```

## Deployment Strategies

### Strategy 1: Sequential Launch (Simple)

Launch workers one by one manually:

```bash
# On TPU VM 0
export TPU_WORKER_ID=0
./launch_worker.sh

# On TPU VM 1
export TPU_WORKER_ID=1
./launch_worker.sh

# ... etc
```

### Strategy 2: Parallel Launch (Automated)

Use a loop with `gcloud` to launch all workers:

```bash
#!/bin/bash
# launch_all_workers.sh

NUM_WORKERS=32
GCS_BUCKET="your-bucket"

for i in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Launching worker $i on TPU tpu-worker-$i"

    gcloud compute tpus tpu-vm ssh tpu-worker-$i \
        --zone us-central2-b \
        --command "
            export TPU_WORKER_ID=$i
            export GCS_BUCKET=$GCS_BUCKET
            export UPLOAD_TO_GCS=true
            cd /home/user/qwen
            nohup ./launch_worker.sh > logs/worker_$i.log 2>&1 &
        "
done
```

### Strategy 3: Kubernetes/Batch Jobs

Create a Kubernetes Job or GCP Batch job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: activation-extraction-worker-5
spec:
  template:
    spec:
      containers:
      - name: worker
        image: gcr.io/your-project/qwen-extraction:latest
        env:
        - name: TPU_WORKER_ID
          value: "5"
        - name: GCS_BUCKET
          value: "your-bucket"
        - name: UPLOAD_TO_GCS
          value: "true"
        command: ["/app/launch_worker.sh"]
      restartPolicy: OnFailure
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v4-podslice
```

## Monitoring

### Check Worker Progress

```bash
# Check checkpoint
cat checkpoints/checkpoint_worker_5.json

# Check local output
ls -lh activations/worker_5/

# Check GCS output
gsutil ls -lh gs://your-bucket/activations/tpu_5/
```

### Aggregate Progress Across Workers

```bash
#!/bin/bash
# check_all_workers.sh

for i in $(seq 0 31); do
    if [ -f "checkpoints/checkpoint_worker_$i.json" ]; then
        samples=$(jq -r '.total_samples_processed' checkpoints/checkpoint_worker_$i.json)
        shards=$(jq -r '.total_shards' checkpoints/checkpoint_worker_$i.json)
        status=$(jq -r '.status // "in_progress"' checkpoints/checkpoint_worker_$i.json)
        echo "Worker $i: $samples samples, $shards shards, $status"
    else
        echo "Worker $i: Not started"
    fi
done
```

### Monitor GCS Upload

```bash
# Count total shards uploaded
gsutil ls gs://your-bucket/activations/*/*.pkl.gz | wc -l

# Check total size
gsutil du -sh gs://your-bucket/activations/
```

## Troubleshooting

### Worker Not Starting

**Symptom**: Worker exits immediately

**Causes**:
- Stream file not found
- Checkpoint directory not writable
- GCS authentication issues

**Solution**:
```bash
# Check stream exists
ls -l data/streams/stream_$(printf '%03d' $TPU_WORKER_ID).jsonl

# Check GCS auth
gcloud auth list
gsutil ls gs://$GCS_BUCKET/

# Check logs
./launch_worker.sh 2>&1 | tee debug.log
```

### Worker Stuck on JIT Compilation

**Symptom**: Worker hangs at "Processing batches" for >5 minutes

**Cause**: First JIT compilation is slow on TPU

**Solution**: Wait 5-10 minutes for first batch. Subsequent batches will be fast.

### Checkpoint Not Resuming

**Symptom**: Worker starts from beginning despite checkpoint existing

**Causes**:
- Checkpoint file corrupted
- Worker ID mismatch

**Solution**:
```bash
# Verify checkpoint
cat checkpoints/checkpoint_worker_$TPU_WORKER_ID.json

# Ensure TPU_WORKER_ID is set correctly
echo $TPU_WORKER_ID

# Manually delete checkpoint to start fresh
rm checkpoints/checkpoint_worker_$TPU_WORKER_ID.json
```

### GCS Upload Failing

**Symptom**: "Failed to upload to GCS" errors

**Causes**:
- Authentication issues
- Bucket doesn't exist
- Permissions issues

**Solution**:
```bash
# Re-authenticate
gcloud auth application-default login

# Create bucket if needed
gsutil mb -l us-central1 gs://$GCS_BUCKET/

# Test write access
echo "test" | gsutil cp - gs://$GCS_BUCKET/test.txt
gsutil rm gs://$GCS_BUCKET/test.txt
```

## Performance Tips

1. **Batch Size**: Start with 4, increase if you have memory headroom
2. **Shard Size**: 1GB is a good default (saves checkpoint every ~10 minutes)
3. **Sequence Length**: Keep at 2048 unless you need longer contexts
4. **Local Storage**: Use fast local SSD if available (enable with `delete_local_after_upload`)
5. **GCS Region**: Use same region as TPUs for faster upload

## Cost Optimization

### Pre-emptible TPUs

Pre-emptible TPUs are ~70% cheaper but can be interrupted. This architecture is **designed for pre-emptible TPUs**:

- Checkpoint/resume handles interruptions automatically
- Data loss is minimal (at most 1 shard worth)
- No coordination needed - workers restart independently

### Recommended Setup

- Use pre-emptible v4-8 TPUs ($1.35/hour vs $4.50/hour)
- Enable GCS upload + local deletion to save disk space
- Set `shard_size_gb=1.0` for frequent checkpoints
- Run 32-64 workers in parallel for fast completion

**Example cost for 200k samples:**
- 32 pre-emptible v4-8 TPUs @ $1.35/hour
- ~2-4 hours total (with pre-emptions)
- Cost: ~$100-200 total vs $400-600 with on-demand

## Advanced: Custom Worker Configuration

### Different Layers Per Worker

Extract different layers on different workers:

```bash
# Worker 0: Layers 0-11
export TPU_WORKER_ID=0
python extract_activations.py \
    --dataset_path data/streams/stream_000.jsonl \
    --layers_to_extract 0 1 2 3 4 5 6 7 8 9 10 11 \
    ...

# Worker 1: Layers 12-23
export TPU_WORKER_ID=1
python extract_activations.py \
    --dataset_path data/streams/stream_001.jsonl \
    --layers_to_extract 12 13 14 15 16 17 18 19 20 21 22 23 \
    ...
```

### Custom Stream Creation

Create streams with specific sample ranges:

```python
from convert_hf_to_arc_format import convert_hf_dataset_to_arc_format

# Worker 0: Samples 0-5000
convert_hf_dataset_to_arc_format(
    dataset_name="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
    column_name="examples",
    output_filename="data/custom_stream_0.jsonl",
    start_index=0,
    end_index=5000
)

# Worker 1: Samples 5000-10000
convert_hf_dataset_to_arc_format(
    dataset_name="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
    column_name="examples",
    output_filename="data/custom_stream_1.jsonl",
    start_index=5000,
    end_index=10000
)
```

## Summary

This parallel worker architecture provides:

✅ **Fault-tolerant** - Automatic checkpoint/resume
✅ **Scalable** - Add more workers to process faster
✅ **Cost-effective** - Designed for pre-emptible TPUs
✅ **Simple** - No coordination between workers
✅ **Organized** - Per-TPU folders in GCS

For questions or issues, refer to the main README or open an issue on the repository.
