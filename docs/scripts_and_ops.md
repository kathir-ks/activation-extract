# Scripts & Operations Reference

This document covers the launch scripts, TPU management, and operational procedures.

---

## Launch Scripts

### scripts/launch_extraction.sh

Resilient extraction launcher with preemption recovery for v5litepod-64.

```bash
nohup bash scripts/launch_extraction.sh > launch.log 2>&1 &
```

**Config (top of file):**
```bash
TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
NUM_WORKERS=16
LAYER=19
GCS_PREFIX="activations/layer19_gridchunk_50k"
CHECKPOINT_PREFIX="checkpoints/gridchunk_layer19"
```

**What it does:**
1. Creates a tarball of the repo (excluding .git, activations, logs)
2. SCPs tarball to all 16 workers in batches of 4
3. Installs JAX + dependencies on each worker
4. Detects worker 0's internal IP for barrier sync
5. Launches `multihost_extract.py` on all workers via `--worker=all`
6. Monitors TPU status every 5 minutes
7. On preemption: waits for TPU recreation (up to 30 min), redeploys, relaunches
8. On process death: checks if code exists (fresh VMs?), redeploys if needed, relaunches

**Extraction command flags:**
- `--topology v5litepod-64` (16 hosts x 4 chips)
- `--pipeline grid_chunking` (continuous token stream, chunk_size=5120)
- `--batch_size 16` (minimum for 16 hosts)
- `--max_seq_length 5120`
- `--shard_size_gb 1.0`
- `--enable_barrier_sync`

**Log rotation:** On each launch, `extraction.log` is renamed to `extraction.log.prev` to preserve crash logs.

### scripts/launch_sae_training_v6e.sh

SAE training launcher for v6e-64 TPU pod. Same resilient pattern as extraction launcher.

```bash
nohup bash scripts/launch_sae_training_v6e.sh > sae_training.log 2>&1 &
```

**Config:**
```bash
TPU_NAME="node-v6e-64-europe-west4-a"
ZONE="europe-west4-a"
NUM_WORKERS=16
ARCHITECTURE="topk"
HIDDEN_DIM=896
DICT_SIZE=7168        # 8x expansion
K=32
BATCH_SIZE=4096
NUM_STEPS=200000
```

**Differences from extraction launcher:**
- Syncs `sae/` directory from sae-worktree before creating tarball
- Installs `optax` in addition to standard deps
- Launches `python3 -u -m sae.scripts.train`
- Monitors for `Training complete` in logs (vs `EXTRACTION COMPLETE`)
- Shows latest training metrics in monitor output

### scripts/start_launcher.sh

Thin wrapper for `@reboot` crontab to auto-start the extraction launcher after control machine reboot.

```bash
#!/bin/bash
REPO_DIR="/home/kathirks_gc/activation-extract"
PIDFILE="$REPO_DIR/launcher.pid"
LOGFILE="$REPO_DIR/launch.log"

# Prevent duplicates
if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Launcher already running (PID $pid)"
        exit 0
    fi
fi

cd "$REPO_DIR"
nohup bash scripts/launch_extraction.sh >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "Launcher started (PID $!), log: $LOGFILE"
```

**Crontab entry:**
```
@reboot /home/kathirks_gc/activation-extract/scripts/start_launcher.sh
```

### scripts/manage_tpus.sh

TPU lifecycle management (create, status, recreate preempted).

```bash
./scripts/manage_tpus.sh create --zones us-central1-a,us-central1-b --workers_per_zone 4
./scripts/manage_tpus.sh status --zones us-central1-a,us-central1-b
./scripts/manage_tpus.sh recreate-preempted --zones us-central1-a --workers_per_zone 4
```

---

## Operational Procedures

### Starting Extraction from Scratch

1. Ensure dataset exists in GCS:
   ```bash
   gcloud storage ls gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl
   ```

2. If dataset missing, regenerate:
   ```bash
   python3 convert_hf_to_arc_format.py \
       --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
       --column_name examples \
       --output_file /tmp/combined_50k.jsonl \
       --max_tasks 50000
   gcloud storage cp /tmp/combined_50k.jsonl gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl
   ```

3. Clean old extraction data:
   ```bash
   gcloud storage rm -r gs://arc-data-europe-west4/activations/layer19_gridchunk_50k/
   gcloud storage rm -r gs://arc-data-europe-west4/checkpoints/gridchunk_layer19/
   ```

4. Start launcher:
   ```bash
   cd /home/kathirks_gc/activation-extract
   bash scripts/start_launcher.sh
   ```

5. Monitor:
   ```bash
   tail -f launch.log
   # Or check worker logs:
   gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=0 \
       --command="tail -50 ~/activation-extract/extraction.log"
   ```

### Checking Extraction Status

```bash
# Launcher status
cat launcher.pid && kill -0 $(cat launcher.pid) 2>/dev/null && echo "Running" || echo "Stopped"

# TPU status
gcloud compute tpus tpu-vm describe TPU_NAME --zone=ZONE --format="value(state)"

# GCS output
gcloud storage ls gs://arc-data-europe-west4/activations/layer19_gridchunk_50k/ | head -20
gcloud storage ls gs://arc-data-europe-west4/activations/layer19_gridchunk_50k/ | wc -l

# Worker logs
gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=0 \
    --command="tail -20 ~/activation-extract/extraction.log"
```

### Stopping Extraction

```bash
# Kill launcher
kill $(cat launcher.pid)

# Kill extraction on all workers
gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=all \
    --command="pkill -f multihost_extract || true"
```

### Enabling/Disabling Auto-Start

```bash
# Enable (already installed)
crontab -l   # Verify @reboot entry exists

# Disable
crontab -l | grep -v start_launcher.sh | crontab -

# Re-enable
(crontab -l 2>/dev/null; echo "@reboot /home/kathirks_gc/activation-extract/scripts/start_launcher.sh") | crontab -
```

### Starting SAE Training

1. Ensure activations exist:
   ```bash
   gcloud storage ls gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/ | wc -l
   ```

2. Start training:
   ```bash
   nohup bash scripts/launch_sae_training_v6e.sh > sae_training.log 2>&1 &
   ```

3. Monitor:
   ```bash
   tail -f sae_training.log
   # Or check training metrics on worker:
   gcloud compute tpus tpu-vm ssh TPU_NAME --zone=ZONE --worker=0 \
       --command="tail -20 ~/activation-extract/training.log"
   ```

---

## GCS Data Layout

```
gs://arc-data-europe-west4/
  dataset_streams/
    combined_50k.jsonl                    # 50K ARC tasks from HuggingFace

  activations/
    layer19_gridchunk_50k/
      host_00/shard_0001.pkl.gz           # Per-host activation shards
      host_00/shard_0002.pkl.gz
      host_00/metadata.json
      host_01/...
      ...
      host_15/...
      grid_chunks_<hash>.pkl.gz           # Cached grid chunks (~2h build)

    layer12_v1_50k_v5litepod-64/
      host_00/...                         # Layer 12 activations (for SAE training)

  checkpoints/
    gridchunk_layer19/
      checkpoint_host_00.json             # Per-host extraction progress
      ...

  sae_checkpoints/
    layer12_topk_v6e64/
      step_00005000/                      # SAE training checkpoints
        params/W_enc.npy
        params/W_dec.npy
        params/b_enc.npy
        params/b_dec.npy
        opt_state/...
        metadata.json
      latest_step.json                    # {"step": 5000}
```

---

## Known Issues & Workarounds

### FSDP Hidden Dim Splitting

The 3D mesh `(data=16, fsdp=2, model=2)` on v5litepod-64 splits hidden_dim (896) across the model axis, giving 448 per shard. `multihost_extract.py` handles this by re-sharding activations to `P('data', None, None)` which triggers an all-gather on the model axis to recover the full 896-dim.

### Grid Chunking Build Time

Building the continuous token stream from 50K tasks takes ~2 hours. The chunk cache (`grid_chunks_<hash>.pkl.gz`) saves this to GCS so restarts skip the rebuild. If the cache doesn't exist (first run or parameter change), all 16 hosts redundantly build the same chunks independently.

### Barrier Sync Timeouts

The `extraction_start` barrier has a 1800s timeout (30 min) to accommodate the ~2 hour grid chunking pipeline running before extraction starts. Other barriers use the default 300s. If the grid chunking takes longer than expected, increase this timeout in `multihost_extract.py`.

### gsutil OpenSSL Error

`gsutil` fails on this machine due to an OpenSSL compatibility issue. Use `gcloud storage` instead:
```bash
# Instead of: gsutil cp file gs://bucket/path
gcloud storage cp file gs://bucket/path

# Instead of: gsutil ls gs://bucket/
gcloud storage ls gs://bucket/
```
