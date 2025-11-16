# Dataset Sharding System

## Overview

Implemented a complete sharding system for distributed activation extraction across multiple machines/workers with automatic workload coordination.

---

## Features

✅ **Automatic Shard Claiming** - Workers automatically claim available shards without manual coordination
✅ **Metadata Tracking** - Track shard status (available/in_use/completed) with atomic updates
✅ **100MB Chunks** - Split each shard into ~100MB files for efficient loading
✅ **8-Way Sharding** - Default 8 shards for distributed processing
✅ **GCS Support** - Works with both local and Google Cloud Storage
✅ **Failure Recovery** - Reset failed shards to make them available again
✅ **Worker Identification** - Unique worker IDs track which machine is processing which shard

---

## New Files

### 1. `create_sharded_dataset.py`

Creates sharded dataset from a single JSONL file.

**Usage:**
```bash
python create_sharded_dataset.py \
  --input_file dataset.jsonl \
  --output_dir gs://bucket/sharded_dataset \
  --num_shards 8 \
  --chunk_size_mb 100 \
  --verbose
```

**Output Structure:**
```
sharded_dataset/
├── master_metadata.json          # Overall dataset info
├── shard_000/
│   ├── metadata.json             # Shard status and info
│   ├── chunk_0000.jsonl          # ~100MB
│   ├── chunk_0001.jsonl
│   └── ...
├── shard_001/
│   ├── metadata.json
│   └── chunk_*.jsonl
└── ... (6 more shards)
```

**Features:**
- Counts total tasks first
- Distributes tasks evenly across shards
- Creates ~100MB chunks within each shard
- Generates metadata for tracking
- Works with local files or GCS

---

### 2. `shard_manager.py`

Manages shard allocation and tracking.

**Python API:**
```python
from shard_manager import ShardManager

# Initialize manager
manager = ShardManager("gs://bucket/sharded_dataset", worker_id="worker_0")

# Claim a shard
shard = manager.claim_shard()  # Auto-selects available shard
# OR
shard = manager.claim_shard(preferred_shard_id=3)  # Try shard 3 first

# Get chunk files
chunk_files = shard["chunks"]  # List of JSONL file paths

# Mark as completed
manager.mark_completed(shard["shard_id"])

# Mark as failed (makes it available again)
manager.mark_failed(shard["shard_id"])

# Get status of all shards
statuses = manager.get_shard_status()
```

**CLI:**
```bash
# Claim a shard
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action claim \
  --worker_id worker_0

# Check status
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action status

# Mark completed
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action complete \
  --shard_id 0

# Reset failed shard
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action reset \
  --shard_id 5
```

---

### 3. Integration with `extract_activations_arc_v5e64.py`

**New Arguments:**
```bash
--use_sharded_dataset              # Enable sharded dataset mode
--sharded_dataset_dir DIR          # Path to sharded dataset
--preferred_shard_id ID            # Optional preferred shard
```

**Example:**
```bash
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://bucket/sharded_dataset \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --gcs_bucket bucket \
  --upload_to_gcs \
  --verbose
```

**What Happens:**
1. Script generates unique worker ID: `machine{X}_host{Y}`
2. Attempts to claim an available shard
3. Loads all tasks from claimed shard's chunks
4. Processes tasks as normal
5. Marks shard as completed when done

---

## Workflow

### 1. Create and Upload Sharded Dataset

```bash
# Convert from HuggingFace
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_file dataset_full.jsonl \
  --max_tasks 10000 \
  --verbose

# Shard into 8 parts
python create_sharded_dataset.py \
  --input_file dataset_full.jsonl \
  --output_dir sharded_dataset \
  --num_shards 8 \
  --chunk_size_mb 100 \
  --verbose

# Upload to GCS
gsutil -m cp -r sharded_dataset gs://your-bucket/
```

### 2. Deploy to Multiple Machines

**Simple Approach (8 separate machines):**
```bash
# Machine 0
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://bucket/sharded_dataset \
  ...

# Machine 1 (different machine, runs in parallel)
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://bucket/sharded_dataset \
  ...

# ... Machines 2-7 (each auto-claims different shard)
```

**TPU v5e-64 Approach (4 hosts, 2 shards each):**
```bash
# Worker 0 claims shard 0
# Worker 1 claims shard 1
# Worker 2 claims shard 2
# Worker 3 claims shard 3

for WORKER in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=${WORKER} --command="
    sudo docker run -d --name extraction --net=host --privileged \
      -v ~/.config/gcloud:/root/.config/gcloud:ro \
      gcr.io/${PROJECT_ID}/activation-extraction:latest \
      -c 'python /workspace/extract_activations_arc_v5e64.py \
        --use_sharded_dataset \
        --sharded_dataset_dir gs://${BUCKET}/sharded_dataset \
        --host_id ${WORKER} --num_hosts 4 --multihost \
        --coordinator_address ${COORDINATOR_IP}:8476 \
        --model_path KathirKs/qwen-2.5-0.5b \
        --batch_size 4 \
        --gcs_bucket ${BUCKET} --upload_to_gcs \
        --verbose'
  " &
done
wait
```

### 3. Monitor Progress

```bash
# Check which shards are being processed
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action status

# Expected output:
#   Shard 0: in_use (assigned to: machine0_host0)
#   Shard 1: in_use (assigned to: machine0_host1)
#   Shard 2: completed
#   Shard 3: completed
#   Shard 4: available
#   ...
```

### 4. Handle Failures

If a machine fails midway:

```bash
# Check status to find failed shard
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action status

# Shard 5 is stuck "in_use" but machine crashed
# Reset it to make available
python shard_manager.py \
  --dataset_dir gs://bucket/sharded_dataset \
  --action reset \
  --shard_id 5

# Now any machine can claim shard 5
```

---

## Metadata Format

### Master Metadata (`master_metadata.json`)

```json
{
  "total_tasks": 10000,
  "num_shards": 8,
  "chunk_size_mb": 100.0,
  "shards": [
    {
      "shard_id": 0,
      "directory": "shard_000",
      "status": "available"
    },
    ...
  ]
}
```

### Shard Metadata (`shard_XXX/metadata.json`)

```json
{
  "shard_id": 0,
  "total_shards": 8,
  "status": "in_use",
  "total_tasks": 1250,
  "num_chunks": 3,
  "chunks": [
    {
      "chunk_id": 0,
      "file": "chunk_0000.jsonl",
      "num_tasks": 450,
      "size_mb": 98.5
    },
    {
      "chunk_id": 1,
      "file": "chunk_0001.jsonl",
      "num_tasks": 450,
      "size_mb": 99.2
    },
    {
      "chunk_id": 2,
      "file": "chunk_0002.jsonl",
      "num_tasks": 350,
      "size_mb": 76.8
    }
  ],
  "assigned_to": "machine0_host0",
  "started_at": "2025-11-13T20:00:00.000000",
  "completed_at": null
}
```

**Status Values:**
- `available` - Shard is ready to be claimed
- `in_use` - Shard is currently being processed
- `completed` - Shard has been fully processed

---

## Testing

Run the test script to verify sharding works:

```bash
bash test_sharding.sh
```

**What it tests:**
1. Creates 100 test tasks
2. Shards into 8 parts with 0.1MB chunks
3. Tests shard claiming
4. Verifies metadata tracking
5. Shows directory structure

---

## Benefits vs. Manual Sharding

### Before (Manual Sharding)

```bash
# Had to manually specify which machine processes which tasks
python extract_activations.py --dataset_path data.jsonl --machine_id 0 --total_machines 8
python extract_activations.py --dataset_path data.jsonl --machine_id 1 --total_machines 8
# ... etc, risk of duplicate work or missed tasks
```

**Problems:**
- ❌ Had to coordinate machine IDs manually
- ❌ If a machine failed, had to re-run entire shard
- ❌ No visibility into which machine is processing what
- ❌ Loading entire dataset file on each machine

### After (Automatic Sharding)

```bash
# All machines run the same command
python extract_activations.py --use_sharded_dataset --sharded_dataset_dir gs://bucket/sharded_dataset
```

**Benefits:**
- ✅ Automatic shard claiming (no coordination needed)
- ✅ Metadata tracks status (easy monitoring)
- ✅ Failed shards can be reset and retried
- ✅ Only loads relevant shard's chunks (efficient)
- ✅ 100MB chunks enable faster loading

---

## Performance Impact

**Dataset Loading:**
- Before: Load entire dataset on each machine
- After: Load only assigned shard's chunks (~12.5% of data for 8 shards)
- Speedup: ~8x faster dataset loading

**Coordination:**
- Before: Manual machine ID assignment
- After: Automatic shard claiming via metadata
- Time saved: Minutes to hours of setup time

**Failure Recovery:**
- Before: Re-run entire machine
- After: Reset shard, any machine can claim it
- Improvement: Much faster recovery

---

## Files Modified

1. `extract_activations_arc_v5e64.py` - Added sharded dataset support
2. `Dockerfile` - Added sharding scripts to image
3. `DATASET_GUIDE.md` - Added sharding documentation

## Files Created

1. `create_sharded_dataset.py` - Shard creation script
2. `shard_manager.py` - Shard management utility
3. `test_sharding.sh` - Test script
4. `SHARDING_SYSTEM.md` - This document

---

## Next Steps

1. ✅ Convert your dataset to ARC format
2. ✅ Create sharded dataset: `python create_sharded_dataset.py`
3. ✅ Upload to GCS: `gsutil -m cp -r sharded_dataset gs://bucket/`
4. ✅ Build and push Docker image (see `QUICK_START.md`)
5. ✅ Deploy to multiple machines with `--use_sharded_dataset`
6. ✅ Monitor with `python shard_manager.py --action status`

---

## Questions?

- See `DATASET_GUIDE.md` for dataset creation
- See `QUICK_START.md` for deployment
- Run `bash test_sharding.sh` to test locally
