# Extraction Runbook

Step-by-step instructions for launching an activation extraction run on a v5litepod-64 TPU pod.

## Prerequisites

1. **GCP project** with TPU quota for v5litepod-64 (or v5e-64)
2. **GCS bucket** in the same region as the TPU (e.g., `europe-west4`)
3. **Dataset** uploaded to GCS as JSONL (ARC task format)
4. **Control machine** with `gcloud` CLI configured and SSH access to the TPU pod
5. **This repo** cloned on the control machine

## Step 1: Create the TPU Pod

```bash
gcloud compute tpus tpu-vm create YOUR_TPU_NAME \
    --zone=europe-west4-b \
    --accelerator-type=v5litepod-64 \
    --version=tpu-ubuntu2204-base \
    --preemptible
```

Verify it's ready:

```bash
gcloud compute tpus tpu-vm describe YOUR_TPU_NAME \
    --zone=europe-west4-b --format="value(state)"
# Should print: READY
```

## Step 2: Upload Your Dataset to GCS

The dataset must be a JSONL file with ARC task format (each line is a JSON object with `train` and `test` keys containing grid arrays).

```bash
gsutil cp your_dataset.jsonl gs://YOUR_BUCKET/dataset_streams/combined_50k.jsonl
```

## Step 3: Configure the Launcher

Edit `scripts/launch_extraction.sh` and update the variables at the top:

```bash
# TPU identity
TPU_NAME="YOUR_TPU_NAME"
ZONE="europe-west4-b"
NUM_WORKERS=16             # 16 for v5litepod-64

# What to extract
LAYER=19                   # Which layer (0-23 for Qwen 2.5 0.5B)
GCS_PREFIX="activations/layer19_gridchunk_50k"
CHECKPOINT_PREFIX="checkpoints/gridchunk_layer19"
```

Inside the `launch_extraction()` function, update these if needed:

| Parameter | Current Value | When to Change |
|-----------|---------------|----------------|
| `--model_path` | `KathirKs/qwen-2.5-0.5b` | Different model |
| `--dataset_path` | `gs://arc-data-europe-west4/...` | Different dataset |
| `--max_tasks` | `50000` | Process fewer/more tasks |
| `--layers_to_extract` | `$LAYER` | Multiple layers: `0 1 2 3` |
| `--activation_type` | `residual` | `mlp` or `attn` |
| `--batch_size` | `16` | OOM? Try 8. Plenty of memory? Try 32 |
| `--max_seq_length` | `5120` | Must match chunk size |
| `--gcs_bucket` | `arc-data-europe-west4` | Your bucket |
| `--predictions_per_task` | `8` | More augmentation = more data |

### Batch Size Guidelines

With grid chunking, ALL sequences are exactly `max_seq_length` tokens (no short sequences). Memory usage is higher than prompt mode.

| Topology | max_seq_length | Safe batch_size |
|----------|---------------|-----------------|
| v5litepod-64 | 5120 | 16 |
| v5litepod-64 | 2048 | 32 |
| v5e-8 (single) | 5120 | 4 |

If you get OOM (C++ SIGABRT crash with no Python traceback), halve the batch_size.

## Step 4: Launch

From the control machine (where this repo is cloned):

```bash
cd ~/activation-extract
nohup bash scripts/launch_extraction.sh > launch.log 2>&1 &
echo $!  # Save this PID
```

The launcher will:
1. Create a tarball of the code
2. SCP + install deps on all 16 workers (in batches of 4)
3. Detect worker 0 IP for barrier sync
4. SSH to `--worker=all` and launch `multihost_extract.py`
5. Enter monitoring loop (polls every 5 minutes)

## Step 5: Monitor

### Watch the launcher

```bash
tail -f launch.log
```

You should see:
```
[2026-03-12 10:00:00] Resilient Extraction Launcher
[2026-03-12 10:00:00] TPU: node-v5e-64-europe-west4-b (europe-west4-b)
[2026-03-12 10:00:01] Setting up all 16 workers...
[2026-03-12 10:05:00] All workers set up
[2026-03-12 10:05:01] Worker 0 IP (barrier host): 10.164.0.61
[2026-03-12 10:05:02] Extraction launched
[2026-03-12 10:10:02] OK: Extraction running. (recoveries: 0)
```

### Check worker logs

```bash
# Worker 0 log
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=ZONE --worker=0 \
    --command="tail -30 ~/activation-extract/extraction.log"

# Worker 1 (usually the JAX primary -- prints verbose output)
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=ZONE --worker=1 \
    --command="tail -50 ~/activation-extract/extraction.log"
```

### Check GCS output

```bash
# List shards
gsutil ls gs://YOUR_BUCKET/activations/layer19_gridchunk_50k_v5litepod-64/host_00/

# Count total shards
gsutil ls "gs://YOUR_BUCKET/activations/layer19_gridchunk_50k_v5litepod-64/host_*/*.pkl.gz" | wc -l

# Check chunk cache exists (saved after first run)
gsutil ls gs://YOUR_BUCKET/checkpoints/gridchunk_layer19/grid_chunks_*.pkl.gz
```

## Auto-Start After Control Machine Reboot

The launcher can survive control machine reboots via `@reboot` crontab.

### Enable auto-start

```bash
# Install the crontab entry (one-time)
(crontab -l 2>/dev/null; echo "@reboot /home/kathirks_gc/activation-extract/scripts/start_launcher.sh") | crontab -
```

### How it works

`scripts/start_launcher.sh` is a thin wrapper that:
1. Checks if the launcher is already running (via PID file)
2. Starts `launch_extraction.sh` under `nohup` if not
3. Writes PID to `launcher.pid` for tracking

On reboot, cron runs the wrapper automatically. On preemption recovery, the existing launcher handles everything — the wrapper just ensures the launcher itself is alive.

### Manage auto-start

```bash
# Check if enabled
crontab -l

# Disable auto-start
crontab -r

# Start manually (safe to call if already running)
bash scripts/start_launcher.sh

# Check if launcher is running
cat launcher.pid && kill -0 $(cat launcher.pid) 2>/dev/null && echo "Running" || echo "Not running"

# Stop the launcher
kill $(cat launcher.pid)
```

## What Happens on Preemption

1. GCP preempts the TPU pod -- all workers die
2. Launcher detects TPU status != READY within 5 minutes
3. Launcher waits for GCP to auto-recreate the preemptible TPU (up to 30 min)
4. Once READY: re-deploys code, re-installs deps, relaunches extraction
5. Each worker loads its checkpoint from GCS and resumes where it left off
6. **Chunk cache**: all workers load pre-computed chunks from GCS (skips ~2 hour data pipeline)

No manual intervention needed. Check `launch.log` for recovery count.

## What Happens on Completion

The launcher detects `EXTRACTION COMPLETE` in worker logs and exits:

```
[2026-03-12 18:00:00] EXTRACTION COMPLETED SUCCESSFULLY!
[2026-03-12 18:00:00] Total recoveries: 2
```

## Extracting Different Layers

To extract a different layer, update `scripts/launch_extraction.sh`:

```bash
LAYER=12  # or any 0-23
GCS_PREFIX="activations/layer12_gridchunk_50k"
CHECKPOINT_PREFIX="checkpoints/gridchunk_layer12"
```

To extract multiple layers in one run, change the `launch_extraction()` function:

```bash
--layers_to_extract 0 6 12 18 23 \
```

Note: more layers = more memory per batch. You may need to reduce `--batch_size`.

## Extracting with Prompt Pipeline (Instead of Grid Chunking)

For full prompts (system prompt + instructions + grids) instead of grid-only chunks:

```bash
--pipeline prompt \
--max_seq_length 2048 \
--batch_size 32 \
```

Prompt pipeline uses dynamic batching (variable sequence lengths), so batch_size can be higher.

## Troubleshooting

### Workers fail at barrier sync

```
RuntimeError: Failed to synchronize at 'model_loaded' barrier
```

Worker 0 IP may have changed after preemption. The launcher auto-detects this, but if running manually, get the current IP:

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=ZONE --worker=0 \
    --command="hostname -I | awk '{print \$1}'"
```

### OOM crash (no Python traceback, just SIGABRT)

Reduce `--batch_size`. With grid chunking at 5120 seq length, attention matrices are large (batch x 14 heads x 5120 x 5120).

### Launcher reports "OK" but extraction actually crashed

Check if `pgrep` is matching itself. The current launcher uses `pgrep -af | grep -v pgrep` to avoid this. If using an old version, update `scripts/launch_extraction.sh`.

### "EXTRACTION COMPLETE" never appears

Check if workers are stuck or slow:

```bash
# Check if python is still running on worker 0
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=ZONE --worker=0 \
    --command="pgrep -af 'python3.*multihost_extract' | grep -v pgrep"

# Check last lines of log for errors
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone=ZONE --worker=1 \
    --command="tail -100 ~/activation-extract/extraction.log"
```

### Chunk cache not being used

Verify the cache file exists:

```bash
gsutil ls gs://YOUR_BUCKET/checkpoints/YOUR_PREFIX/grid_chunks_*.pkl.gz
```

Cache invalidation happens when task IDs, chunk_size, predictions_per_task, or random_seed change. If you changed any of these, a new cache will be built.
