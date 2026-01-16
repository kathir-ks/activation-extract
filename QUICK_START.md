# Quick Start Guide - TPU Activation Extraction

## ⚡ Fastest Way to Deploy

### Test Run (2 workers, 100 samples) with Auto-Monitoring

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket YOUR_BUCKET_NAME \
  --zones us-central1-a \
  --workers_per_zone 2 \
  --max_samples 100 \
  --create_tpus \
  --monitor
```

### Production Run (8 workers, full dataset) with Auto-Monitoring

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket YOUR_BUCKET_NAME \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --create_tpus \
  --monitor
```

**Note:** The `--monitor` flag enables:
- Live progress dashboard showing samples processed, shards created, and GCS uploads
- Automatic detection and recovery of preempted TPUs
- Continuous monitoring until you stop it (Ctrl+C)

## 📋 Common Commands

### Create TPUs
```bash
./scripts/manage_tpus.sh create --zones us-central1-a --workers_per_zone 4
```

### Check Status
```bash
./scripts/manage_tpus.sh status --zones us-central1-a
```

### View Worker Log
```bash
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a \
  --command='tail -f ~/activation-extract/extraction.log'
```

### Monitor GCS
```bash
gsutil ls gs://YOUR_BUCKET/activations/
```

### Recreate Preempted TPUs
```bash
./scripts/manage_tpus.sh recreate-preempted --zones us-central1-a --workers_per_zone 4
```

### Delete TPUs
```bash
./scripts/manage_tpus.sh delete --zones us-central1-a --workers_per_zone 4
```

## 📊 Monitoring Mode

The integrated monitoring mode provides:
- **Live Dashboard**: Real-time view of all TPU workers
- **Progress Tracking**: Samples processed, shards created, GCS uploads
- **Auto-Recovery**: Automatically detects and recovers preempted TPUs
- **Status Summary**: Healthy vs needs-recovery counts

### Start Monitoring
```bash
# Add --monitor to any deployment
./scripts/deploy_to_tpus.sh \
  --gcs_bucket YOUR_BUCKET \
  --zones us-central1-a \
  --workers_per_zone 4 \
  --monitor

# Customize check interval (default: 60 seconds)
./scripts/deploy_to_tpus.sh \
  --gcs_bucket YOUR_BUCKET \
  --zones us-central1-a \
  --workers_per_zone 4 \
  --monitor \
  --monitor_interval 120
```

### Dashboard Display
```
╔════════════════════════════════════════════════════════════════╗
║         ACTIVATION EXTRACTION - LIVE MONITORING               ║
╚════════════════════════════════════════════════════════════════╝

GCS Bucket: gs://your-bucket
Total Workers: 4 across 1 zone(s)
Check Interval: 60s

╔════════════════════════════════════════════════════════════════╗
║  TPU NAME                 STATUS      SAMPLES    SHARDS   GCS  ║
╠════════════════════════════════════════════════════════════════╣
║  tpu-us-central1-a-0      READY        1250       150     150  ║
║  tpu-us-central1-a-1      READY        1180       142     142  ║
║  tpu-us-central1-a-2      PREEMPT         0         0       0  ║
║  tpu-us-central1-a-3      READY        1300       156     156  ║
╠════════════════════════════════════════════════════════════════╣
║  TOTALS                                3730       448     448  ║
╚════════════════════════════════════════════════════════════════╝

Status Summary:
  ✓ Healthy: 3
  ⟳ Working: 3
  ✗ Needs Recovery: 1
```

## Full Documentation

See **README_DEPLOYMENT.md** for complete guide.
