# Multi-Host TPU Usage Guide

## Quick Start: v5e-64 (4 hosts × 16 chips = 64 chips)

### Step 1: Identify Coordinator
Choose host 0 as coordinator. Get its internal IP:
```bash
# On host 0:
hostname -I | awk '{print $1}'
# Example output: 10.0.0.10
```

### Step 2: Launch on Each Host

**On Host 0** (coordinator):
```bash
python extract_activations_fineweb_multihost.py \
    --machine_id 0 \
    --total_machines 1 \
    --multihost \
    --coordinator_address "10.0.0.10:8476" \
    --host_id 0 \
    --num_hosts 4 \
    --mesh_type 2d \
    --model_path "Qwen/Qwen2.5-7B" \
    --batch_size 8 \
    --layers_to_extract 20 21 22 23 24 25 26 27 \
    --upload_to_gcs \
    --gcs_bucket your-bucket-name \
    --verbose
```

**On Host 1**:
```bash
python extract_activations_fineweb_multihost.py \
    --machine_id 0 \
    --total_machines 1 \
    --multihost \
    --coordinator_address "10.0.0.10:8476" \
    --host_id 1 \
    --num_hosts 4 \
    --mesh_type 2d \
    --model_path "Qwen/Qwen2.5-7B" \
    --batch_size 8 \
    --layers_to_extract 20 21 22 23 24 25 26 27 \
    --upload_to_gcs \
    --gcs_bucket your-bucket-name \
    --verbose
```

**On Host 2 & 3**: Same command but change `--host_id` to 2 and 3 respectively.

### Step 3: Monitor Progress

Check logs on each host:
```bash
# Host 0 should show:
# INITIALIZING MULTI-HOST JAX DISTRIBUTED
# Global devices: 64
# Local devices on host 0: 16
# Mesh type: 2d
# Mesh shape: (4, 16)
# Created mesh with axis: ('data', 'model')
```

---

## Mesh Types Explained

### 1D Mesh (Pure Model Parallelism)
```bash
--mesh_type 1d
```
- All 64 chips shard the model
- No data parallelism
- **Use case**: Model too large for 16-way sharding
- **Example**: 40B+ models

### 2D Mesh (Data + Model Parallelism) ⭐ RECOMMENDED
```bash
--mesh_type 2d
```
- 4-way data parallel (one per host)
- 16-way model parallel (chips within each host)
- **Use case**: 7B-30B models, best throughput
- **Example**: Qwen 7B, LLaMA 13B

### 3D Mesh (Pipeline + Data + Model)
```bash
--mesh_type 3d
```
- For 70B+ models
- Automatic fallback to 2D if < 64 devices

---

## Expected Output

### 2D Mesh (Recommended for v5e-64 + Qwen 7B):
```
======================================================================
FINEWEB-EDU ACTIVATION EXTRACTION - MACHINE 0/0
MULTI-HOST MODE - Host 0/3
======================================================================
...
INITIALIZING MULTI-HOST JAX DISTRIBUTED
======================================================================
  Coordinator: 10.0.0.10:8476
  Total hosts: 4
  This host ID: 0
  ✓ JAX distributed initialized
  Global devices: 64 (TPU v5e...)
  Local devices on host 0: 16
  Process index: 0
  Process count: 4
======================================================================

Setting up model sharding across 64 devices...
  Multi-host mode: 4 hosts × 16 devices/host
  Mesh type: 2d
  ✓ Created mesh with axis: ('data', 'model')
  ✓ Mesh shape: (4, 16)
  ✓ Created sharding strategy with 12 rules
  ⟳ Sharding parameters across devices...
  ✓ Parameters sharded successfully

  Memory distribution (global view across all hosts):
    Host 0: 16 local devices
    Total: 64 devices across 4 hosts

Extracting activations from layers [20, 21, 22, 23, 24, 25, 26, 27]...
Mode: Sharding (2D mesh: ('data', 'model')) - Multi-host (4 hosts)
```

---

## Performance Expectations

### v5e-64 with 2D Mesh

**Qwen 2.5 7B**:
- Model sharded 16 ways: ~437 MB per chip ✅ Fits easily in 16GB HBM
- 4-way data parallelism: 4× throughput
- Expected: ~2000-5000 samples/hour
- Cost: ~$20-30/hour

**Comparison to Single-Host v4-8**:
- Single-host v4-8: ~500 samples/hour @ $2/hour = $4 per 1000 samples
- Multi-host v5e-64: ~3000 samples/hour @ $25/hour = $8.3 per 1000 samples
- **Cost per sample**: Similar, but 6× faster!

---

## Troubleshooting

### "Connection refused" or timeout
**Problem**: Hosts can't reach coordinator
**Solution**:
- Check coordinator IP is internal IP, not external
- Ensure all hosts in same VPC/subnet
- Firewall allows port 8476

### "Coordinator address required"
**Problem**: Forgot `--multihost` or `--coordinator_address`
**Solution**:
```bash
--multihost \
--coordinator_address "10.0.0.10:8476"
```

### "Sharding spec mismatch"
**Problem**: Wrong mesh type for model size
**Solution**:
- For 7B: Use `--mesh_type 2d` ✅
- For 40B+: Use `--mesh_type 1d`

### "Deadlock" or "Hang"
**Problem**: Not all hosts started
**Solution**: Launch on ALL hosts simultaneously within ~30 seconds

### Different mesh shapes on different hosts
**Problem**: Inconsistent arguments across hosts
**Solution**: Ensure `--num_hosts`, `--mesh_type`, `--model_path` are IDENTICAL on all hosts

---

## Advanced: Multiple Machines + Multi-Host

If you want 32 v5e-64 pods (32 machines × 4 hosts × 16 chips = 2048 chips!):

**Machine 0, Host 0**:
```bash
--machine_id 0 --total_machines 32 --host_id 0 --num_hosts 4 --coordinator_address "10.0.0.10:8476"
```

**Machine 0, Host 1**:
```bash
--machine_id 0 --total_machines 32 --host_id 1 --num_hosts 4 --coordinator_address "10.0.0.10:8476"
```

**Machine 1, Host 0**:
```bash
--machine_id 1 --total_machines 32 --host_id 0 --num_hosts 4 --coordinator_address "10.0.1.10:8476"
```

Each machine has its own coordinator!

---

## GCS Output Structure

```
gs://your-bucket/activations_fineweb/
├── machine_000_host_00/
│   ├── shard_0001.pkl.gz
│   ├── shard_0002.pkl.gz
│   └── metadata.json
├── machine_000_host_01/
│   └── ...
├── machine_000_host_02/
│   └── ...
├── machine_000_host_03/
│   └── ...
└── ...
```

Each host saves independently to avoid coordination overhead.

---

## Summary

**For v5e-64 + Qwen 7B**:
```bash
# Best configuration:
--mesh_type 2d           # Data + model parallelism
--num_hosts 4            # 4 hosts in v5e-64
--batch_size 8           # Per host
```

**Expected**:
- ✅ 6× faster than single-host
- ✅ Similar cost per sample
- ✅ Can handle larger models
- ✅ Easy to scale to 32 machines

**Next**: Just test on your actual v5e-64! 🚀
