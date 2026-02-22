# Deployment Guide - Production FSDP Extraction

## Quick Start (Smoke Test)

### 1. Push code to GitHub
```bash
git push origin main
```

### 2. Get your TPU pod name and zone
```bash
# List TPUs
gcloud compute tpus tpu-vm list --zone=us-central1-a

# Example TPU name
TPU_NAME="node-v5e-64-us-central1-a"
ZONE="us-central1-a"
```

### 3. Run a small smoke test (50 samples)
```bash
./scripts/auto_recover.sh \
    --tpu_name "$TPU_NAME" \
    --zone "$ZONE" \
    --dataset_path gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
    --gcs_bucket fineweb-data-us-central1 \
    --gcs_prefix activations/smoke_test \
    --batch_size 32 \
    --layers "0 5 10 15 20 23" \
    --max_tasks 50 \
    --topology v5litepod-64 \
    --max_retries 3
```

This will:
- Run on all 16 workers simultaneously
- Process 50 samples (takes ~2-3 minutes)
- Upload to GCS at `gs://fineweb-data-us-central1/activations/smoke_test_v5litepod-64/host_*/`
- Retry up to 3 times on preemption
- Use GCS checkpoints for resume

### 4. Verify output
```bash
# Check GCS output
gsutil ls -lh gs://fineweb-data-us-central1/activations/smoke_test_v5litepod-64/

# Should see 16 host directories (host_00/ through host_15/)
gsutil ls gs://fineweb-data-us-central1/activations/smoke_test_v5litepod-64/host_*/

# Check one host's output
gsutil ls -lh gs://fineweb-data-us-central1/activations/smoke_test_v5litepod-64/host_00/
```

### 5. Manual resume test (optional)
To test the checkpoint/resume system:

```bash
# Start extraction (no max_tasks, so it runs forever)
./scripts/auto_recover.sh \
    --tpu_name "$TPU_NAME" \
    --zone "$ZONE" \
    --dataset_path gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
    --gcs_bucket fineweb-data-us-central1 \
    --gcs_prefix activations/resume_test \
    --batch_size 32 \
    --layers "0 5 10 15 20 23" \
    --topology v5litepod-64 \
    --max_retries 5

# After 1 minute, press Ctrl+C to stop
# Then re-run the same command
# It should resume from where it left off
```

Check the logs for:
```
📌 RESUMING FROM CHECKPOINT (Host 0, source: gcs)
  Last processed sample: 159
  Starting from sample: 160
```

---

## Production Run (1B-4B tokens)

### 6-layer extraction on single v5litepod-64

```bash
./scripts/auto_recover.sh \
    --tpu_name "$TPU_NAME" \
    --zone "$ZONE" \
    --dataset_path gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
    --gcs_bucket fineweb-data-us-central1 \
    --gcs_prefix activations/prod_6layer_1B_tokens \
    --batch_size 32 \
    --layers "0 5 10 15 20 23" \
    --topology v5litepod-64 \
    --model_path Qwen/Qwen2.5-0.5B \
    --max_retries 50 \
    --checkpoint_gcs_bucket fineweb-data-us-central1
```

**Scale estimates:**
- **1B tokens** (~488K samples): ~8-9 hours, ~$85 spot cost, ~2.6 TB compressed output
- **4B tokens** (~1.95M samples): ~34 hours, ~$330 spot cost, ~10.5 TB compressed output

The script will:
- Auto-recover from preemptions (up to 50 retries)
- Resume from GCS checkpoints after each preemption
- Store checkpoints at `gs://fineweb-data-us-central1/checkpoints/checkpoint_v5litepod-64_host_*.json`
- Output to `gs://fineweb-data-us-central1/activations/prod_6layer_1B_tokens_v5litepod-64/host_*/`

---

## Monitoring

### Real-time logs
The `auto_recover.sh` script runs locally and shows all worker output. You can safely Ctrl+C and re-run — it will resume from the GCS checkpoint.

### Check extraction progress
```bash
# Check how many shards each host has uploaded
for i in {00..15}; do
    echo -n "Host $i: "
    gsutil ls gs://fineweb-data-us-central1/activations/prod_6layer_1B_tokens_v5litepod-64/host_$i/ | wc -l
done

# Check checkpoint state
gsutil cat gs://fineweb-data-us-central1/checkpoints/checkpoint_v5litepod-64_host_00.json | jq
```

### Check TPU state
```bash
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='get(state)'
```

### SSH to a worker (for debugging)
```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0
```

---

## Troubleshooting

### "Checkpoint mismatch detected"
```
RuntimeError: Checkpoint mismatch detected! This host wants to resume from
sample 16000, but other hosts have a different resume point.
Fix: delete local checkpoints (rm -rf ./checkpoints) on all hosts
```

**Solution:**
```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
    --command="rm -rf ~/activation-extract/checkpoints/*"
```

Then re-run the extraction — all hosts will load from GCS.

### Auto-recovery not working
If `auto_recover.sh` exits before max retries, check:
1. TPU state: `gcloud compute tpus tpu-vm describe ... --format='get(state)'`
2. If state is `NOT_FOUND`, you may need to manually recreate the TPU
3. Check the error logs above the exit message

### SSH hangs during setup
If `setup_worker.sh` hangs during `gcloud tpu-vm ssh --worker=all`, one worker may be stuck. SSH to each worker individually to diagnose:
```bash
for i in {0..15}; do
    echo "=== Worker $i ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=$i \
        --command="hostname && uptime"
done
```

### Out of disk space
Each host stores activations locally before uploading. If you see `No space left on device`:
1. Add `--delete_local_after_upload` to the extraction args (in `auto_recover.sh`, line ~210)
2. Or use a smaller `--shard_size_gb` (default 1.0)

---

## Manual TPU Setup (if needed)

If the TPU was deleted or needs recreation:

```bash
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type=v5litepod-64 \
    --version=tpu-ubuntu2204-base \
    --preemptible
```

Then run `auto_recover.sh` — it will set up all workers automatically.

---

## Next Steps After Smoke Test

1. **Verify smoke test output** (see step 4 above)
2. **Test manual resume** (see step 5 above)
3. **Launch production run** (see "Production Run" section)
4. **Monitor progress** (see "Monitoring" section)
5. **Celebrate when it completes!** 🎉
