# TPU Deployment System - Complete Guide

## Overview

This deployment system handles end-to-end activation extraction across multiple TPU workers with:
- ✅ **Automatic TPU management** (create/destroy/recreate preempted TPUs)
- ✅ **Multi-zone deployment** (distribute workers across regions for better availability)
- ✅ **Zero-setup deployment** (auto-clones repo and installs dependencies on TPU VMs)
- ✅ **Checkpoint/resume** (automatic recovery from preemptions)
- ✅ **GCS integration** (per-worker isolated storage)
- ✅ **Integrated monitoring** (live progress dashboard with automatic preemption recovery)

## Quick Start

### 1. Test Deployment (2 workers, limited data) with Monitoring

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a \
  --workers_per_zone 2 \
  --max_samples 100 \
  --create_tpus \
  --monitor
```

This will:
1. Create 2 preemptible TPUs in us-central1-a
2. Prepare and upload dataset streams
3. Launch extraction on all workers
4. Enter monitoring mode with live progress dashboard
5. Automatically recover any preempted TPUs

### 2. Production Deployment (8 workers across 2 zones) with Monitoring

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --create_tpus \
  --monitor
```

The monitoring dashboard shows:
- Real-time TPU status (READY, PREEMPTED, etc.)
- Samples processed per worker
- Shards created and uploaded to GCS
- Automatic recovery actions

## Architecture

### TPU Naming Convention

TPUs follow the pattern: `tpu-{region}-{zone_letter}-{worker_id}`

**Examples:**
- Zone `us-central1-a`, worker 0 → `tpu-us-central1-a-0`
- Zone `us-central1-b`, worker 3 → `tpu-us-central1-b-3`
- Zone `europe-west4-a`, worker 1 → `tpu-europe-west4-a-1`

This naming enables:
- Easy identification of worker location
- Automated management across zones
- Clear debugging and monitoring

### Directory Structure

```
activation-extract/                    # GitHub repo (cloned on each TPU)
├── requirements.txt                   # Python dependencies (JAX TPU, transformers, etc.)
├── extract_activations.py            # Main extraction script
├── create_dataset_streams.py         # Dataset splitting
├── qwen2_jax.py                      # Model implementation
├── qwen2_jax_with_hooks.py          # Activation extraction model
├── arc24/                            # ARC dataset utilities
├── core/                             # Core utilities
└── scripts/
    ├── setup_tpu_worker.sh          # Clone repo + install deps (runs on TPU)
    ├── run_extraction_worker.sh     # Full worker pipeline (setup + extract)
    ├── manage_tpus.sh               # TPU lifecycle management
    ├── deploy_to_tpus.sh            # End-to-end deployment
    └── upload_dataset_streams_to_gcs.sh
```

## Scripts Reference

### 1. `setup_tpu_worker.sh` - Worker Environment Setup

**Purpose:** Runs on each TPU VM to clone repo and install dependencies

**What it does:**
1. Installs system dependencies (git, python3, pip)
2. Clones https://github.com/kathir-ks/activation-extract
3. Installs Python packages from requirements.txt
4. Verifies JAX TPU installation
5. Creates necessary directories

**Usage:**
```bash
# On TPU VM (automatically called by run_extraction_worker.sh)
./setup_tpu_worker.sh

# Force reinstall
./setup_tpu_worker.sh --force
```

**Output location:** `~/activation-extract/`

---

### 2. `run_extraction_worker.sh` - Complete Worker Pipeline

**Purpose:** Full extraction pipeline for a single TPU worker

**What it does:**
1. **Setup** (if first time): Clones repo and installs deps
2. **Update**: Pulls latest code from GitHub
3. **Download**: Fetches dataset stream from GCS
4. **Extract**: Runs activation extraction with checkpointing
5. **Upload**: Uploads shards to GCS in per-worker folder

**Usage:**
```bash
# Direct invocation (from GitHub, no local files needed!)
bash <(curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/run_extraction_worker.sh) \
  --gcs_bucket fineweb-data-us-central1 \
  --dataset_stream gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
  --model Qwen/Qwen2.5-0.5B

# Or locally on TPU
./scripts/run_extraction_worker.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --dataset_stream gs://fineweb-data-us-central1/datasets/stream_000.jsonl
```

**Key Features:**
- Auto-detects `TPU_WORKER_ID` environment variable
- Resumes from checkpoint if exists
- Logs to `~/activation-extract/extraction.log`

---

### 3. `manage_tpus.sh` - TPU Lifecycle Management

**Purpose:** Create, delete, monitor, and recreate TPUs across multiple zones

**Commands:**

#### Create TPUs
```bash
./scripts/manage_tpus.sh create \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --tpu_type v3-8
```

#### Delete TPUs
```bash
./scripts/manage_tpus.sh delete \
  --zones us-central1-a \
  --workers_per_zone 4
```

#### Check Status
```bash
./scripts/manage_tpus.sh status \
  --zones us-central1-a,us-central1-b
```

#### Recreate Preempted TPUs
```bash
# Automatically detects and recreates only preempted TPUs
./scripts/manage_tpus.sh recreate-preempted \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4
```

#### List All TPUs
```bash
./scripts/manage_tpus.sh list
```

**Naming Convention:**
- Automatically generates names: `tpu-{region}-{zone_letter}-{worker_id}`
- Example: `tpu-us-central1-a-0`, `tpu-europe-west4-a-3`

---

### 4. `deploy_to_tpus.sh` - End-to-End Deployment

**Purpose:** Complete deployment orchestration

**What it does:**
1. **Create TPUs** (optional, with `--create_tpus`)
2. **Prepare datasets**: Split into N independent streams
3. **Upload datasets**: To GCS bucket
4. **Launch extraction**: On all workers via remote SSH + curl

**Usage:**

```bash
# Full deployment with TPU creation
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --create_tpus

# Test with limited data
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a \
  --workers_per_zone 2 \
  --max_samples 100 \
  --create_tpus

# Just launch (TPUs already exist, dataset already uploaded)
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a \
  --workers_per_zone 4 \
  --skip_dataset
```

**Options:**
- `--create_tpus`: Create TPUs before deployment
- `--skip_dataset`: Skip dataset preparation (use existing)
- `--skip_launch`: Skip extraction launch (just prepare)
- `--max_samples N`: Limit samples per stream (for testing)

---

### 5. `upload_dataset_streams_to_gcs.sh` - Dataset Upload

**Purpose:** Upload local JSONL streams to GCS

**Usage:**
```bash
./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-us-central1 \
  --prefix datasets \
  --local_dir ./dataset_streams
```

## Workflow

### Standard Deployment Flow

```
┌─────────────────────────────────────┐
│ 1. Create TPUs (manage_tpus.sh)    │
│    - Creates TPUs in specified zones│
│    - Names: tpu-{region}-{zone}-{id}│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. Prepare Datasets (local)         │
│    - create_dataset_streams.py      │
│    - Split into N independent files │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. Upload to GCS                    │
│    - upload_dataset_streams_to_gcs  │
│    - gs://bucket/datasets/stream_*.jsonl│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. Launch Extraction on Each TPU   │
│    - SSH to TPU                     │
│    - curl run_extraction_worker.sh  │
│    - Auto setup + extract + upload  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ On Each TPU (automatic):            │
│  a. Clone repo (if first time)      │
│  b. Install dependencies            │
│  c. Download dataset stream         │
│  d. Run extraction with checkpoints │
│  e. Upload shards to GCS            │
│     → gs://bucket/activations/tpu_N/│
└─────────────────────────────────────┘
```

### Handling Preemptions

When a TPU gets preempted:

**Option 1: Automatic Recreation**
```bash
# Recreate only preempted TPUs
./scripts/manage_tpus.sh recreate-preempted \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4

# Then relaunch extraction
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --skip_dataset  # Dataset already uploaded
```

**Option 2: Manual Recreation**
```bash
# Delete specific TPU
./scripts/manage_tpus.sh delete \
  --zones us-central1-a \
  --workers_per_zone 1

# Create replacement
./scripts/manage_tpus.sh create \
  --zones us-central1-a \
  --workers_per_zone 1

# Launch extraction (will resume from checkpoint)
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a --command="
  bash <(curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/run_extraction_worker.sh) \
    --gcs_bucket fineweb-data-us-central1 \
    --dataset_stream gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
    > ~/extraction.log 2>&1 &
"
```

## Monitoring

### Integrated Monitoring Mode (Recommended)

The deployment script includes an integrated monitoring mode that provides:
- **Live progress dashboard** with real-time updates
- **Automatic preemption detection** across all TPUs
- **Automatic recovery** (delete → recreate → relaunch)
- **Progress tracking** (samples processed, shards created, GCS uploads)
- **Status summaries** (healthy, working, needs recovery)

**Enable with `--monitor` flag:**

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a,us-central1-b \
  --workers_per_zone 4 \
  --create_tpus \
  --monitor \
  --monitor_interval 60  # Check every 60 seconds (default)
```

**Dashboard Example:**
```
╔════════════════════════════════════════════════════════════════╗
║         ACTIVATION EXTRACTION - LIVE MONITORING               ║
╚════════════════════════════════════════════════════════════════╝

GCS Bucket: gs://fineweb-data-us-central1
Total Workers: 8 across 2 zone(s)
Check Interval: 60s

╔════════════════════════════════════════════════════════════════╗
║  TPU NAME                 STATUS      SAMPLES    SHARDS   GCS  ║
╠════════════════════════════════════════════════════════════════╣
║  tpu-us-central1-a-0      READY        1250       150     150  ║
║  tpu-us-central1-a-1      READY        1180       142     142  ║
║  tpu-us-central1-a-2      PREEMPT         0         0       0  ║
║  tpu-us-central1-a-3      READY        1300       156     156  ║
║  tpu-us-central1-b-0      READY        1220       146     146  ║
║  tpu-us-central1-b-1      READY        1190       143     143  ║
║  tpu-us-central1-b-2      READY        1210       145     145  ║
║  tpu-us-central1-b-3      STARTING        0         0       0  ║
╠════════════════════════════════════════════════════════════════╣
║  TOTALS                                7350       882     882  ║
╚════════════════════════════════════════════════════════════════╝

Status Summary:
  ✓ Healthy: 6
  ⟳ Working: 7
  ✗ Needs Recovery: 1

Last update: 2026-01-16 14:32:15
Press Ctrl+C to stop monitoring
```

**Features:**
- **Auto-detection**: Identifies PREEMPTED, NOT_FOUND, or TERMINATED TPUs
- **Auto-recovery**: Automatically recreates and relaunches on failed TPUs
- **Checkpoint resume**: Recovered workers resume from last checkpoint
- **Real-time updates**: Dashboard refreshes every `monitor_interval` seconds
- **Graceful shutdown**: Press Ctrl+C to stop monitoring

**When to use:**
- ✅ Production runs where preemptions are expected
- ✅ Long-running extraction jobs
- ✅ Multi-zone deployments
- ✅ When you want hands-off operation

**When to use manual monitoring:**
- Small test runs
- Debugging specific workers
- When you need fine-grained control

---

### Manual Monitoring Commands

For manual monitoring and troubleshooting, use these commands:

#### Check Worker Logs

```bash
# View log for specific worker
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 \
  --zone=us-central1-a \
  --command='tail -f ~/activation-extract/extraction.log'
```

### Check Checkpoints

```bash
# View checkpoint status
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 \
  --zone=us-central1-a \
  --command='cat ~/activation-extract/checkpoints/checkpoint_worker_0.json'
```

### Monitor GCS Uploads

```bash
# List all worker outputs
gsutil ls gs://fineweb-data-us-central1/activations/

# Count files for specific worker
gsutil ls gs://fineweb-data-us-central1/activations/tpu_0/ | wc -l

# Check total size
gsutil du -sh gs://fineweb-data-us-central1/activations/
```

### Check TPU Status

```bash
# Status across all zones
./scripts/manage_tpus.sh status --zones us-central1-a,us-central1-b

# List all TPUs
./scripts/manage_tpus.sh list
```

## Multi-Zone Deployment Example

Deploy 16 workers across 4 zones (4 workers per zone):

```bash
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a,us-central1-b,us-central1-c,us-central1-f \
  --workers_per_zone 4 \
  --create_tpus
```

This creates:
- Zone `us-central1-a`: `tpu-us-central1-a-{0,1,2,3}`
- Zone `us-central1-b`: `tpu-us-central1-b-{0,1,2,3}`
- Zone `us-central1-c`: `tpu-us-central1-c-{0,1,2,3}`
- Zone `us-central1-f`: `tpu-us-central1-f-{0,1,2,3}`

Benefits:
- Better availability (zone failures isolated)
- Higher preemption tolerance
- Potential quota distribution across zones

## Key Features

### 1. Zero-Setup Deployment

Workers automatically:
- Clone repo from GitHub
- Install all dependencies (JAX TPU, transformers, etc.)
- Verify installation
- Start extraction

**No manual setup required on TPU VMs!**

### 2. Checkpoint/Resume

Each worker maintains checkpoint:
```json
{
  "worker_id": 0,
  "last_processed_sample_idx": 1250,
  "total_samples_processed": 1251,
  "total_shards": 150,
  "dataset_path": "gs://bucket/datasets/stream_000.jsonl",
  "status": "in_progress"
}
```

On restart, extraction resumes from `last_processed_sample_idx`.

### 3. Per-Worker GCS Storage

Output structure:
```
gs://bucket/activations/
├── tpu_0/
│   ├── shard_0001.pkl.gz
│   ├── shard_0002.pkl.gz
│   └── ...
├── tpu_1/
│   └── ...
└── tpu_N/
    └── ...
```

No conflicts between workers!

### 4. Automatic Preemption Recovery

1. Worker gets preempted
2. Checkpoint saved with last position
3. Run `recreate-preempted` to recreate TPU
4. Relaunch extraction
5. Resumes from checkpoint automatically

## Configuration

### `requirements.txt`

Updated with TPU-specific dependencies:
```txt
# JAX and TPU support
jax[tpu]>=0.4.23
flax>=0.8.0

# HuggingFace ecosystem
transformers>=4.38.0
datasets>=2.17.0

# PyTorch (CPU only for weight conversion)
torch>=2.2.0

# Utilities
jinja2>=3.1.0
termcolor>=2.3.0
numpy>=1.24.0
google-cloud-storage>=2.14.0
```

### Shard Size

Default: **1.0 GB** (configured in `extract_activations.py`)

Reduces number of objects in GCS while maintaining manageable file sizes.

## Troubleshooting

### TPU Creation Fails

```bash
# Check quota
gcloud compute tpus describe-quota --zone=us-central1-a

# Try different zone
./scripts/manage_tpus.sh create --zones us-central1-b --workers_per_zone 4
```

### Worker Setup Fails

```bash
# SSH to worker and check logs
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a

# Manual setup
cd ~/activation-extract
./scripts/setup_tpu_worker.sh --force
```

### Extraction Stalled

```bash
# Check if process running
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a \
  --command='ps aux | grep extract_activations'

# Check logs
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a \
  --command='tail -100 ~/activation-extract/extraction.log'
```

### GCS Upload Issues

```bash
# Test GCS access
gsutil ls gs://fineweb-data-us-central1/

# Manual upload
gcloud compute tpus tpu-vm ssh tpu-us-central1-a-0 --zone=us-central1-a \
  --command='gsutil -m cp ~/activation-extract/activations/*.pkl.gz gs://bucket/activations/tpu_0/'
```

## Cost Optimization

### Use Preemptible TPUs

Preemptible TPUs cost ~70% less:
- Standard v3-8: ~$4.00/hour
- Preemptible v3-8: ~$1.35/hour

With checkpoint/resume, preemptions are handled automatically!

### Multi-Zone for Availability

Distribute workers across zones:
- Reduces impact of zone-specific preemptions
- Better quota distribution
- Higher overall availability

### Monitor and Stop

```bash
# Check progress
./scripts/manage_tpus.sh status --zones us-central1-a

# Delete when done
./scripts/manage_tpus.sh delete --zones us-central1-a --workers_per_zone 4
```

## GitHub Repository

**Repository:** https://github.com/kathir-ks/activation-extract

This repo is automatically cloned on each TPU worker. Push updates to GitHub and workers will pull latest code on next run.

## Summary

✅ **Created:**
- `setup_tpu_worker.sh` - Auto-setup on TPU VMs
- `run_extraction_worker.sh` - Complete worker pipeline
- `manage_tpus.sh` - TPU lifecycle management
- `deploy_to_tpus.sh` - End-to-end deployment
- `requirements.txt` - TPU-optimized dependencies

✅ **Features:**
- Zero-setup deployment (curl + run!)
- Multi-zone support
- Automatic preemption recovery
- Checkpoint/resume
- Per-worker GCS isolation

✅ **Ready for:**
- Small-scale testing (2-4 workers)
- Production deployment (16+ workers)
- Multi-zone deployment (high availability)
