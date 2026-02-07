# TPU Pod Verification Guide

This guide explains how to test the updated multihost barrier synchronization on an actual TPU pod.

## What Changed

The multihost extraction system now **automatically detects which worker is worker 0** using the `CLOUD_TPU_TASK_ID` environment variable that Google Cloud sets when you SSH with `--worker=all`.

**Before**: Had to manually specify `--is_barrier_server` flag differently for each worker (which was impossible with `--worker=all`)

**After**: Worker 0 is automatically detected and starts the barrier server

## Quick Test

Run the existing test script - it should now work without modifications:

```bash
./test_multihost_v5e64.sh
```

## Expected Behavior

When you run multihost extraction, you should see:

### On Worker 0 (automatically detected):
```
📋 Worker ID: 0
   Is barrier server: True

🚀 Starting barrier server on port 5555...
✓ Barrier server ready on port 5555
⏳ Waiting at 'pre_jax_init' barrier...
```

### On Workers 1-15:
```
📋 Worker ID: 1  (or 2, 3, ... 15)
   Is barrier server: False

⏳ Waiting at 'pre_jax_init' barrier...
```

### All Workers (after synchronization):
```
✓ Synchronized! Starting JAX init...
```

## Verification Checklist

- [ ] All workers print their detected Worker ID (0-15)
- [ ] Worker 0 shows `Is barrier server: True`
- [ ] Workers 1-15 show `Is barrier server: False`  
- [ ] Barrier server starts successfully (Worker 0 only)
- [ ] All workers wait at `pre_jax_init` barrier
- [ ] All workers pass the barrier simultaneously
- [ ] JAX initialization completes without "unexpected peer" errors
- [ ] Model loads on all workers
- [ ] Extraction proceeds normally

## Debugging

If you see issues:

### Problem: "Connection refused" to barrier server

**Check**: Worker 0 IP detection
```bash
# On worker 0, check internal IP:
hostname -I | awk '{print $1}'
```

The script should auto-detect this, but you can override:
```bash
--barrier_controller_host 10.0.0.1  # Replace with actual Worker 0 IP
```

### Problem: Wrong worker detected as worker 0

**Check**: Environment variables
```bash
# On each worker:
echo $CLOUD_TPU_TASK_ID
```

Should be 0, 1, 2, ... 15 for a 16-worker pod.

### Problem: Barrier timeout

**Possible causes**:
- Not all workers reached the barrier (some crashed early)
- Network connectivity issues between workers
- Barrier controller host IP incorrect

**Check logs** from all workers to identify which worker didn't reach the barrier.

## Manual Override (if needed)

If auto-detection fails, you can still manually specify:

```bash
# Worker 0 only:
python3 multihost_extract.py --is_barrier_server ...

# All other workers - don't add the flag
```

But this defeats the purpose of using `--worker=all`, so auto-detection should be preferred.

## Running the Full Extraction

Once verified, run your normal multihost extraction:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=all \
    --command="cd ~/activation-extract-multihost && \
        python3 multihost_extract.py \
            --topology v5litepod-64 \
            --dataset_path gs://bucket/data.jsonl \
            --gcs_bucket my-bucket \
            --batch_size 64 \
            --enable_barrier_sync \
            --verbose"
```

No need to specify `--barrier_controller_host` or `--is_barrier_server` - they're auto-detected!
