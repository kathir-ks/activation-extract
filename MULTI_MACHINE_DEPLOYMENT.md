# Multi-Machine Deployment Guide

## Overview

This guide explains how to run activation extraction across 32 TPU v4-8 machines in parallel, with each machine processing a different subset of the FineWeb-Edu dataset.

## Architecture

```
Machine 0                Machine 1                ...        Machine 31
├── TPU v4-8            ├── TPU v4-8                        ├── TPU v4-8
├── Qwen 7B (sharded)   ├── Qwen 7B (sharded)              ├── Qwen 7B (sharded)
├── Dataset shard 0/32  ├── Dataset shard 1/32             ├── Dataset shard 31/32
└── Output: machine_0/  └── Output: machine_1/             └── Output: machine_31/
```

**Key Features**:
- **Data parallelism**: Each machine processes different data samples
- **Model parallelism**: Each machine shards the 7B model across 4 TPU chips
- **No inter-machine communication**: Machines work independently
- **Distributed storage**: Each machine writes to separate GCS directories

## Prerequisites

### 1. GCS Setup

Create a GCS bucket for storing activations:
```bash
# Create bucket (run once)
gsutil mb -l us-central1 gs://your-activations-bucket

# Create directory structure
gsutil mkdir gs://your-activations-bucket/qwen_7b_fineweb/
```

### 2. SSH Access

Ensure you can SSH to all machines:
```bash
# Test SSH access
for i in {0..31}; do
  echo "Testing machine $i..."
  gcloud compute ssh tpu-vm-$i --zone=us-central1-a --command="echo OK"
done
```

### 3. Environment Setup

On each machine, ensure the following are installed:
- JAX with TPU support
- transformers
- datasets
- fsspec (for GCS)
- gcsfs

```bash
# Install on all machines
for i in {0..31}; do
  gcloud compute ssh tpu-vm-$i --zone=us-central1-a --command="
    pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install transformers datasets fsspec gcsfs
  "
done
```

## Configuration

### Single-Machine Configuration (for testing)

```bash
python extract_activations_fineweb_multihost.py \
  --machine_id 0 \
  --total_machines 1 \
  --model_path "Qwen/Qwen2.5-7B" \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_config "sample-10BT" \
  --dataset_split "train" \
  --layers_to_extract 15 16 17 18 \
  --batch_size 16 \
  --max_seq_length 512 \
  --max_samples 1000 \
  --mesh_type 1d \
  --output_dir "./test_extraction" \
  --shard_size_gb 0.5 \
  --verbose
```

### 32-Machine Configuration (production)

Each machine runs:
```bash
python extract_activations_fineweb_multihost.py \
  --machine_id ${MACHINE_ID} \
  --total_machines 32 \
  --model_path "Qwen/Qwen2.5-7B" \
  --dataset_name "HuggingFaceFW/fineweb-edu" \
  --dataset_config "sample-10BT" \
  --dataset_split "train" \
  --layers_to_extract 15 16 17 18 \
  --batch_size 16 \
  --max_seq_length 512 \
  --max_samples 100000 \
  --mesh_type 1d \
  --upload_to_gcs \
  --gcs_bucket "your-activations-bucket" \
  --gcs_prefix "qwen_7b_fineweb/machine_${MACHINE_ID}" \
  --shard_size_gb 0.5 \
  --compress_shards \
  --delete_local_after_upload \
  --verbose
```

## Deployment Scripts

### 1. Upload Code to All Machines

```bash
#!/bin/bash
# upload_code.sh

ZONE="us-central1-a"
CODE_DIR="/home/kathirks_gc/torch_xla/qwen"

for i in {0..31}; do
  echo "Uploading to machine $i..."
  gcloud compute scp --recurse \
    ${CODE_DIR}/* \
    tpu-vm-$i:~/qwen/ \
    --zone=${ZONE} &
done

wait
echo "Upload complete!"
```

### 2. Launch Extraction on All Machines

```bash
#!/bin/bash
# launch_extraction.sh

ZONE="us-central1-a"
TOTAL_MACHINES=32
MODEL="Qwen/Qwen2.5-7B"
DATASET="HuggingFaceFW/fineweb-edu"
LAYERS="15 16 17 18"
BATCH_SIZE=16
MAX_SEQ_LEN=512
MAX_SAMPLES=100000
GCS_BUCKET="your-activations-bucket"
GCS_PREFIX="qwen_7b_fineweb"

for i in {0..31}; do
  echo "Starting extraction on machine $i..."

  gcloud compute ssh tpu-vm-$i --zone=${ZONE} --command="
    cd ~/qwen
    nohup python extract_activations_fineweb_multihost.py \
      --machine_id $i \
      --total_machines ${TOTAL_MACHINES} \
      --model_path '${MODEL}' \
      --dataset_name '${DATASET}' \
      --dataset_config 'sample-10BT' \
      --dataset_split 'train' \
      --layers_to_extract ${LAYERS} \
      --batch_size ${BATCH_SIZE} \
      --max_seq_length ${MAX_SEQ_LEN} \
      --max_samples ${MAX_SAMPLES} \
      --mesh_type 1d \
      --upload_to_gcs \
      --gcs_bucket '${GCS_BUCKET}' \
      --gcs_prefix '${GCS_PREFIX}/machine_$i' \
      --shard_size_gb 0.5 \
      --compress_shards \
      --delete_local_after_upload \
      --verbose \
      > extraction_$i.log 2>&1 &

    echo 'Started on machine $i'
  " &
done

wait
echo "All machines started!"
```

### 3. Monitor Progress

```bash
#!/bin/bash
# monitor_progress.sh

ZONE="us-central1-a"

while true; do
  clear
  echo "=================================================="
  echo "EXTRACTION PROGRESS ($(date))"
  echo "=================================================="

  for i in {0..31}; do
    status=$(gcloud compute ssh tpu-vm-$i --zone=${ZONE} --command="
      if pgrep -f extract_activations_fineweb_multihost.py > /dev/null; then
        tail -5 ~/qwen/extraction_$i.log 2>/dev/null | grep -oP '\\d+it' | tail -1 || echo 'Running...'
      else
        echo 'Completed or Not Started'
      fi
    " 2>/dev/null)

    printf "Machine %2d: %s\n" $i "$status"
  done

  sleep 30
done
```

### 4. Check Completion

```bash
#!/bin/bash
# check_completion.sh

ZONE="us-central1-a"
GCS_BUCKET="your-activations-bucket"
GCS_PREFIX="qwen_7b_fineweb"

echo "Checking completion status..."
echo ""

for i in {0..31}; do
  # Check if process is still running
  running=$(gcloud compute ssh tpu-vm-$i --zone=${ZONE} --command="
    pgrep -f extract_activations_fineweb_multihost.py > /dev/null && echo 'YES' || echo 'NO'
  " 2>/dev/null)

  # Check GCS files
  file_count=$(gsutil ls gs://${GCS_BUCKET}/${GCS_PREFIX}/machine_$i/ 2>/dev/null | wc -l)

  printf "Machine %2d: Running=%s, Files=%d\n" $i "$running" $file_count
done

echo ""
echo "Total files in GCS:"
gsutil ls -r gs://${GCS_BUCKET}/${GCS_PREFIX}/ | wc -l
```

## Data Distribution

The script automatically distributes data across machines:

```python
# Each machine processes a different shard
samples_per_machine = total_samples // total_machines
start_idx = machine_id * samples_per_machine
end_idx = start_idx + samples_per_machine

# Machine 0: samples 0-3124
# Machine 1: samples 3125-6249
# ...
# Machine 31: samples 96874-99999
```

**Sample IDs are globally unique**:
```python
global_sample_id = machine_id * samples_per_machine + local_idx
```

## Storage Organization

### GCS Directory Structure

```
gs://your-activations-bucket/qwen_7b_fineweb/
├── machine_0/
│   ├── metadata.json
│   ├── shard_0001.pkl.gz
│   ├── shard_0002.pkl.gz
│   └── ...
├── machine_1/
│   ├── metadata.json
│   ├── shard_0001.pkl.gz
│   └── ...
...
└── machine_31/
    ├── metadata.json
    ├── shard_0001.pkl.gz
    └── ...
```

### Merging Metadata (after completion)

```python
#!/usr/bin/env python
# merge_metadata.py

import json
from pathlib import Path
import subprocess

GCS_BUCKET = "your-activations-bucket"
GCS_PREFIX = "qwen_7b_fineweb"
TOTAL_MACHINES = 32

# Download all metadata files
for i in range(TOTAL_MACHINES):
    subprocess.run([
        "gsutil", "cp",
        f"gs://{GCS_BUCKET}/{GCS_PREFIX}/machine_{i}/metadata.json",
        f"metadata_{i}.json"
    ])

# Merge metadata
merged = {
    "total_machines": TOTAL_MACHINES,
    "machines": [],
    "total_samples": 0,
    "total_shards": 0,
    "total_size_bytes": 0
}

for i in range(TOTAL_MACHINES):
    with open(f"metadata_{i}.json") as f:
        machine_meta = json.load(f)
        merged["machines"].append({
            "machine_id": i,
            "metadata": machine_meta
        })
        merged["total_samples"] += machine_meta.get("total_samples", 0)
        merged["total_shards"] += machine_meta.get("total_shards", 0)

# Save merged metadata
with open("merged_metadata.json", "w") as f:
    json.dump(merged, f, indent=2)

# Upload to GCS
subprocess.run([
    "gsutil", "cp",
    "merged_metadata.json",
    f"gs://{GCS_BUCKET}/{GCS_PREFIX}/merged_metadata.json"
])

print(f"Merged metadata:")
print(f"  Total samples: {merged['total_samples']:,}")
print(f"  Total shards: {merged['total_shards']:,}")
```

## Resource Estimates

### Per Machine (TPU v4-8, Qwen 7B)

- **Memory**: 16 GB available per chip after model load
- **Batch size**: 16 (recommended)
- **Throughput**: ~2-3 samples/second
- **Storage**: ~500 MB per shard

### 32 Machines Processing 100K Samples

- **Total samples**: 100,000
- **Samples per machine**: 3,125
- **Time per machine**: ~20-30 minutes
- **Total storage**: ~150-200 GB
- **Cost**: ~$50-100 (depending on TPU pricing)

### Full FineWeb-Edu (10BT, ~96M samples)

- **Samples per machine**: ~3M samples
- **Time per machine**: ~15-20 hours
- **Total storage**: ~45-60 TB
- **Cost**: ~$2,000-3,000

## Troubleshooting

### 1. Out of Memory

**Symptoms**: JAX OOM errors

**Solutions**:
- Reduce `batch_size` (try 8 or 4)
- Reduce `max_seq_length` (try 256)
- Extract fewer layers at once

### 2. Slow Performance

**Symptoms**: <1 sample/second

**Checks**:
```bash
# Check TPU utilization
watch -n 1 'python -c "import jax; print(jax.devices())"'

# Check for JIT recompilation
grep "Compiling" extraction_*.log
```

**Solutions**:
- Ensure `max_seq_length` is fixed (not varying)
- Increase `batch_size` for better TPU utilization
- Check network bandwidth to GCS

### 3. GCS Upload Failures

**Symptoms**: `gsutil` errors in logs

**Solutions**:
```bash
# Test GCS access
gsutil ls gs://your-activations-bucket/

# Check credentials
gcloud auth application-default login

# Increase retry count
export CLOUDSDK_STORAGE_RETRY_LIMIT=10
```

### 4. Machine Crashes

**Symptoms**: Process dies unexpectedly

**Recovery**:
```bash
# Find last processed sample
tail -100 extraction_${MACHINE_ID}.log | grep "sample"

# Restart from checkpoint (if implemented)
# OR restart with reduced max_samples to skip completed portion
```

## Best Practices

1. **Start small**: Test with 1 machine and 100 samples first
2. **Monitor closely**: Watch first few minutes of multi-machine run
3. **Stagger starts**: Don't start all 32 machines simultaneously (can overwhelm HF servers)
4. **Use GCS**: Don't store locally then upload - stream directly to GCS
5. **Checkpoint frequently**: Set `shard_size_gb` to 0.5-1.0 for frequent saves
6. **Compress**: Always use `--compress_shards` to save storage costs
7. **Clean up**: Use `--delete_local_after_upload` to avoid filling local disks

## Example: Quick Start

```bash
# 1. Upload code
./upload_code.sh

# 2. Test on one machine
gcloud compute ssh tpu-vm-0 --zone=us-central1-a --command="
  cd ~/qwen
  python extract_activations_fineweb_multihost.py \
    --machine_id 0 --total_machines 1 \
    --model_path 'Qwen/Qwen2.5-7B' \
    --layers_to_extract 15 16 \
    --batch_size 16 \
    --max_samples 10 \
    --verbose
"

# 3. If successful, launch all machines
./launch_extraction.sh

# 4. Monitor progress
./monitor_progress.sh

# 5. After completion, merge metadata
python merge_metadata.py
```

## Support

For issues, check:
1. Machine logs: `~/qwen/extraction_${MACHINE_ID}.log`
2. GCS upload logs: Look for `gsutil` errors
3. TPU status: `gcloud compute tpus list`
4. JAX devices: `python -c "import jax; print(jax.devices())"`
