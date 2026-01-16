# Quick Reference: Multi-Region Deployment

## Your TPU Quota

```
┌─────────────────────────────────────────────────────────────┐
│  Region            Type    Chips   Hosts   Stream Range     │
├─────────────────────────────────────────────────────────────┤
│  us-central1-a     v5e-8   64      8       0-7              │
│  europe-west4-b    v5e-8   64      8       8-15             │
│  us-east1-d        v6e-8   64      8       16-23            │
│  europe-west4-a    v6e-8   64      8       24-31            │
├─────────────────────────────────────────────────────────────┤
│  TOTAL                     256     32      0-31             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
Dataset: 200K samples
        ↓
Split into 32 streams
        ↓
    ┌───┴───┬───────┬───────┐
    ↓       ↓       ↓       ↓
┌────────────────────────────────────────────────────┐
│ US Bucket: fineweb-data-us-central1               │
├────────────────────────────────────────────────────┤
│ streams 0-7    → us-central1-a (v5e) → tpu_0-7    │
│ streams 16-23  → us-east1-d (v6e)    → tpu_16-23  │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Europe Bucket: fineweb-data-europe-west4          │
├────────────────────────────────────────────────────┤
│ streams 8-15   → europe-west4-b (v5e) → tpu_8-15  │
│ streams 24-31  → europe-west4-a (v6e) → tpu_24-31 │
└────────────────────────────────────────────────────┘
```

## Three-Step Deployment

### Step 1: Setup Streams (Run Once)
```bash
bash configs/setup_streams.sh
```
Creates 32 streams and uploads to both buckets.

### Step 2: Deploy Regions

**Option A: Parallel (Recommended - 4 terminals)**
```bash
# Terminal 1
bash configs/deploy_us_central1_v5e.sh

# Terminal 2
bash configs/deploy_europe_west4_v5e.sh

# Terminal 3
bash configs/deploy_us_east1_v6e.sh

# Terminal 4
bash configs/deploy_europe_west4_v6e.sh
```

**Option B: Sequential (1 terminal)**
```bash
bash configs/deploy_all_regions.sh
```

### Step 3: Monitor
Each deployment shows live dashboard with:
- TPU status (READY, PREEMPTED, etc.)
- Samples processed per worker
- Shards created and uploaded
- Auto-recovery for preemptions

Press Ctrl+C to stop monitoring (workers keep running).

## Verification Commands

### Check Stream Distribution
```bash
# Verify each stream exists in GCS
for i in {0..31}; do
  printf "Stream %02d: " $i
  gsutil -q stat gs://fineweb-data-us-central1/datasets/stream_$(printf "%03d" $i).jsonl 2>/dev/null && echo "✓ US" || \
  gsutil -q stat gs://fineweb-data-europe-west4/datasets/stream_$(printf "%03d" $i).jsonl 2>/dev/null && echo "✓ EU" || \
  echo "✗ MISSING"
done
```

### Check TPU Status
```bash
# All regions at once
gcloud compute tpus tpu-vm list --filter="name:tpu-*" \
  --format="table(name,zone,state,accelerator_type)"
```

### Check Activation Outputs
```bash
# US bucket
gsutil ls gs://fineweb-data-us-central1/activations/

# Europe bucket
gsutil ls gs://fineweb-data-europe-west4/activations/

# Count shards per worker
for i in {0..31}; do
  count=$(gsutil ls gs://fineweb-data-us-central1/activations/tpu_$i/*.pkl.gz 2>/dev/null | wc -l)
  [ $count -eq 0 ] && count=$(gsutil ls gs://fineweb-data-europe-west4/activations/tpu_$i/*.pkl.gz 2>/dev/null | wc -l)
  echo "Worker $i: $count shards"
done
```

## Key Points

✅ **No Duplicates**: Each worker processes unique streams via `--stream_offset`
✅ **Auto-Recovery**: Preempted TPUs automatically recreated and relaunched
✅ **Region Buckets**: US regions use us-central1 bucket, Europe uses europe-west4 bucket
✅ **Checkpoint Resume**: Recovered workers resume from last checkpoint
✅ **Live Monitoring**: Real-time dashboard shows progress across all workers

## Cost Estimate

- **Preemptible v5e-8**: ~$1.35/hour × 16 hosts = **~$21.60/hour**
- **Preemptible v6e-8**: ~$1.35/hour × 16 hosts = **~$21.60/hour**
- **Total**: **~$43.20/hour** for all 32 workers

With automatic preemption recovery, you get ~70% cost savings vs on-demand pricing!

## Troubleshooting

### TPU Creation Fails
```bash
# Check quota
gcloud compute tpus describe-quota --zone=us-central1-a

# Try different zone or contact support
```

### Stream Missing
```bash
# Re-upload specific streams
./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-us-central1 \
  --prefix datasets \
  --local_dir ./dataset_streams
```

### Worker Stalled
```bash
# Check logs
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 \
  --zone=us-central1-a \
  --command='tail -100 ~/activation-extract/extraction.log'

# Check checkpoint
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 \
  --zone=us-central1-a \
  --command='cat ~/activation-extract/checkpoints/checkpoint_worker_*.json'
```

### Manual Recovery
```bash
# If monitoring stopped, manually recreate preempted TPUs
./scripts/manage_tpus.sh recreate-preempted \
  --zones us-central1-a \
  --workers_per_zone 8
```
