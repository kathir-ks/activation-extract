# Data Storage Architecture for Distributed TPU v5e-64

## Overview

This document explains how data (dataset, activations, and model) is stored and accessed in the distributed TPU v5e-64 architecture.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Google Cloud Storage (GCS)                  │
│  ┌────────────────────┐  ┌────────────────────────────────────┐    │
│  │ Input Dataset      │  │ Output Activations                 │    │
│  │ gs://bucket/       │  │ gs://bucket/activations_arc_v5e64/ │    │
│  │ datasets/          │  │                                     │    │
│  │ arc_...jsonl       │  │ ├── machine_000_host_00/           │    │
│  │                    │  │ │   ├── shard_0001.pkl.gz          │    │
│  │                    │  │ │   ├── shard_0002.pkl.gz          │    │
│  │                    │  │ │   └── metadata.json              │    │
│  │                    │  │ ├── machine_000_host_01/           │    │
│  │                    │  │ ├── machine_000_host_02/           │    │
│  │                    │  │ └── machine_000_host_03/           │    │
│  └────────────────────┘  └────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↕ Download/Upload
┌─────────────────────────────────────────────────────────────────────┐
│                         TPU v5e-64 Pod (1 Machine)                   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Host 0 (Coordinator)                    8 TPU chips           │  │
│  │ ┌────────────────┐ ┌──────────────────────────────────────┐ │  │
│  │ │ Local Storage  │ │ Memory (128 GB HBM)                  │ │  │
│  │ │ ~/activation-  │ │ ┌─────────────┐  ┌────────────────┐ │ │  │
│  │ │ extraction/    │ │ │ Model Params│  │  Activations   │ │ │  │
│  │ │ ├── dataset    │ │ │ (Sharded)   │  │   (Buffer)     │ │ │  │
│  │ │ ├── code       │ │ │   Chip 0    │  │                │ │ │  │
│  │ │ └── logs       │ │ │   Chip 1    │  │                │ │ │  │
│  │ └────────────────┘ │ │   ...       │  │                │ │ │  │
│  │                    │ │   Chip 7    │  │                │ │ │  │
│  │                    │ └─────────────┘  └────────────────┘ │ │  │
│  └────────────────────┘ └──────────────────────────────────────┘ │  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Host 1                                  8 TPU chips           │  │
│  │ (Same structure as Host 0)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Host 2                                  8 TPU chips           │  │
│  │ (Same structure as Host 0)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Host 3                                  8 TPU chips           │  │
│  │ (Same structure as Host 0)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Types and Storage Locations

### 1. Input Dataset

#### Storage Location
```
Primary: Google Cloud Storage (GCS)
└── gs://YOUR_BUCKET/datasets/arc_formatted_challenges.jsonl

Replicated to Each Host:
└── Host 0: ~/activation-extraction/arc_formatted_challenges.jsonl
└── Host 1: ~/activation-extraction/arc_formatted_challenges.jsonl
└── Host 2: ~/activation-extraction/arc_formatted_challenges.jsonl
└── Host 3: ~/activation-extraction/arc_formatted_challenges.jsonl
```

#### Key Points
- **Source**: JSONL file in GCS (single copy)
- **Replication**: Downloaded to ALL hosts (4 copies total)
- **Why replicate?**: Each host needs local access to read dataset
- **Size**: ~45 MB for 10,000 tasks (negligible)

#### How It's Loaded
```python
# Each host loads the SAME dataset file
dataset = load_arc_dataset_jsonl(
    dataset_path='arc_formatted_challenges.jsonl',
    machine_id=0,        # All hosts use same machine_id (single v5e-64)
    total_machines=1     # Only 1 v5e-64 machine
)

# Dataset is sharded IN MEMORY per host (not on disk)
# Each host processes different subset of tasks
```

**Important**: The dataset is **NOT sharded on disk**. All hosts have the same file, but each host processes different tasks.

### 2. Model Parameters

#### Storage Location
```
Primary: HuggingFace Hub
└── KathirKs/qwen-2.5-7b (remote)

Temporary Download (Each Host):
└── ~/.cache/huggingface/hub/models--KathirKs--qwen-2.5-7b/

In-Memory (Sharded Across ALL 32 TPU Chips):
┌──────────────────────────────────────────────┐
│  Chip 0 (Host 0): Shard 0 of model params    │
│  Chip 1 (Host 0): Shard 1 of model params    │
│  ...                                          │
│  Chip 7 (Host 0): Shard 7 of model params    │
│  Chip 8 (Host 1): Shard 8 of model params    │
│  ...                                          │
│  Chip 31 (Host 3): Shard 31 of model params  │
└──────────────────────────────────────────────┘
```

#### Key Points
- **Source**: Downloaded from HuggingFace (Qwen 2.5 7B = ~14 GB)
- **Caching**: Cached on each host (~14 GB × 4 hosts = ~56 GB total)
- **In-Memory Sharding**: Model is **sharded across all 32 chips**
  - Each chip holds 1/32 of the model parameters
  - 7B model @ float32 = ~28 GB / 32 chips = ~875 MB per chip
- **Why shard?**: Model doesn't fit on single chip (16 GB HBM)

#### Sharding Example (2D Mesh)
```python
# 2D Mesh: 4 hosts × 8 chips = (4, 8) = (data_axis, model_axis)
mesh = Mesh(devices, axis_names=('data', 'model'))

# Model parameters are sharded along 'model' axis
# Example for a weight matrix [hidden_size, vocab_size]:
embedding = jax.device_put(
    embedding_weights,
    NamedSharding(mesh, PartitionSpec('model', None))
)
# Result: Embedding sharded across 8 chips per host (model axis)
```

### 3. Activations (Output)

#### Storage Flow
```
1. In-Memory Buffer (Each Host):
   Host 0: activations_buffer = {layer_10: [...], layer_11: [...], ...}
   Host 1: activations_buffer = {layer_10: [...], layer_11: [...], ...}
   Host 2: activations_buffer = {layer_10: [...], layer_11: [...], ...}
   Host 3: activations_buffer = {layer_10: [...], layer_11: [...], ...}

2. When Buffer Full (reaches shard_size_gb):
   → Save to local disk:
     Host 0: ~/activation-extraction/activations_arc_v5e64/.../shard_0001.pkl.gz
     Host 1: ~/activation-extraction/activations_arc_v5e64/.../shard_0001.pkl.gz
     Host 2: ~/activation-extraction/activations_arc_v5e64/.../shard_0001.pkl.gz
     Host 3: ~/activation-extraction/activations_arc_v5e64/.../shard_0001.pkl.gz

3. Upload to GCS:
   → Host 0: gs://bucket/activations_arc_v5e64/machine_000_host_00/shard_0001.pkl.gz
   → Host 1: gs://bucket/activations_arc_v5e64/machine_000_host_01/shard_0001.pkl.gz
   → Host 2: gs://bucket/activations_arc_v5e64/machine_000_host_02/shard_0001.pkl.gz
   → Host 3: gs://bucket/activations_arc_v5e64/machine_000_host_03/shard_0001.pkl.gz

4. (Optional) Delete Local Copy:
   → Saves disk space on TPU hosts
```

#### Key Points
- **In-Memory Buffer**: Each host has independent buffer (~100-500 MB)
- **Local Disk**: Temporary storage before GCS upload
- **GCS (Final)**: Permanent storage with host-specific paths
- **Independence**: Each host's data is COMPLETELY SEPARATE

#### GCS Directory Structure
```
gs://YOUR_BUCKET/activations_arc_v5e64/
├── machine_000_host_00/          # Host 0's data
│   ├── shard_0001.pkl.gz         # 1 GB compressed
│   ├── shard_0002.pkl.gz
│   ├── shard_0003.pkl.gz
│   └── metadata.json             # Metadata for all shards
├── machine_000_host_01/          # Host 1's data
│   ├── shard_0001.pkl.gz
│   ├── shard_0002.pkl.gz
│   └── metadata.json
├── machine_000_host_02/          # Host 2's data
│   └── ...
└── machine_000_host_03/          # Host 3's data
    └── ...
```

#### Shard Content Format
```python
# Each shard is a pickle file with this structure:
{
    layer_10: [
        {
            'sample_idx': 0,
            'activation': np.array([seq_len, hidden_dim]),
            'shape': (2048, 3584),
            'text_preview': 'Task: task_00000001, Prompt: ...'
        },
        {
            'sample_idx': 1,
            'activation': np.array([seq_len, hidden_dim]),
            ...
        },
        ...
    ],
    layer_11: [...],
    layer_12: [...],
    ...
}
```

## Data Distribution Strategy

### Single v5e-64 Machine (4 hosts)

```
Dataset: 10,000 tasks
├── Host 0 processes: Tasks 0, 4, 8, 12, ... (2,500 tasks)
├── Host 1 processes: Tasks 1, 5, 9, 13, ... (2,500 tasks)
├── Host 2 processes: Tasks 2, 6, 10, 14, ... (2,500 tasks)
└── Host 3 processes: Tasks 3, 7, 11, 15, ... (2,500 tasks)
```

**How it works**:
```python
# In load_arc_dataset_jsonl():
for line_idx, line in enumerate(dataset_file):
    # Round-robin distribution based on HOST_ID
    if line_idx % 4 != host_id:  # 4 = num_hosts
        continue  # Skip this task

    # This host processes this task
    process_task(line)
```

### Multiple v5e-64 Machines (e.g., 4 machines = 16 hosts total)

```
Dataset: 100,000 tasks

Machine 0 (4 hosts):
├── Host 0: Tasks 0, 16, 32, 48, ... (6,250 tasks)
├── Host 1: Tasks 1, 17, 33, 49, ...
├── Host 2: Tasks 2, 18, 34, 50, ...
└── Host 3: Tasks 3, 19, 35, 51, ...

Machine 1 (4 hosts):
├── Host 0: Tasks 4, 20, 36, 52, ...
├── Host 1: Tasks 5, 21, 37, 53, ...
├── Host 2: Tasks 6, 22, 38, 54, ...
└── Host 3: Tasks 7, 23, 39, 55, ...

Machine 2 (4 hosts):
├── Host 0: Tasks 8, 24, 40, 56, ...
├── Host 1: Tasks 9, 25, 41, 57, ...
├── Host 2: Tasks 10, 26, 42, 58, ...
└── Host 3: Tasks 11, 27, 43, 59, ...

Machine 3 (4 hosts):
├── Host 0: Tasks 12, 28, 44, 60, ...
├── Host 1: Tasks 13, 29, 45, 61, ...
├── Host 2: Tasks 14, 30, 46, 62, ...
└── Host 3: Tasks 15, 31, 47, 63, ...
```

**How it works**:
```python
# Two-level sharding: machine-level + host-level
for line_idx, line in enumerate(dataset_file):
    # First, check if this machine should process this task
    if line_idx % total_machines != machine_id:
        continue

    # Then, check if this host should process this task
    if (line_idx // total_machines) % num_hosts != host_id:
        continue

    # This specific host on this specific machine processes this task
    process_task(line)
```

## Memory Hierarchy

### On Each Host

```
┌─────────────────────────────────────────────────────────┐
│ Host Memory Layout                                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 1. CPU RAM (~16-32 GB)                                  │
│    ├── Dataset (JSONL file): ~45 MB                     │
│    ├── Python process: ~2-4 GB                          │
│    ├── HuggingFace cache: ~14 GB                        │
│    └── Temporary buffers: ~1-2 GB                       │
│                                                          │
│ 2. Local Disk (SSD, ~100-200 GB)                        │
│    ├── OS and system: ~20 GB                            │
│    ├── Dataset: ~45 MB                                  │
│    ├── Code: ~100 MB                                    │
│    ├── Logs: ~1 GB                                      │
│    └── Activation shards (temporary): ~10-50 GB         │
│                                                          │
│ 3. TPU HBM (8 chips × 16 GB = 128 GB per host)         │
│    ├── Model parameters (sharded): ~7 GB per chip       │
│    ├── Activations (computed): ~4-6 GB per chip         │
│    ├── Intermediate tensors: ~2-3 GB per chip           │
│    └── JAX/XLA runtime: ~1-2 GB per chip                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Across All 4 Hosts

```
Total TPU Memory: 128 GB/host × 4 hosts = 512 GB
├── Model parameters: ~28 GB (distributed across 32 chips)
├── Activations: ~64-96 GB (during computation)
├── Intermediate: ~32-64 GB
└── Runtime: ~16-32 GB

Total Disk Space (temporary): ~50 GB/host × 4 hosts = ~200 GB
└── Deleted after GCS upload if delete_local_after_upload=true
```

## Data Flow During Extraction

### Step-by-Step Process

```
1. INITIALIZATION (each host independently):
   ┌─────────────────────────────────────────────┐
   │ • Download model from HuggingFace           │
   │ • Load dataset from local disk              │
   │ • Initialize JAX distributed                │
   │ • Shard model across TPU chips              │
   └─────────────────────────────────────────────┘

2. TASK ASSIGNMENT (automatic):
   ┌─────────────────────────────────────────────┐
   │ Host 0: Tasks [0, 4, 8, 12, ...]            │
   │ Host 1: Tasks [1, 5, 9, 13, ...]            │
   │ Host 2: Tasks [2, 6, 10, 14, ...]           │
   │ Host 3: Tasks [3, 7, 11, 15, ...]           │
   └─────────────────────────────────────────────┘

3. PROMPT CREATION (per host):
   ┌─────────────────────────────────────────────┐
   │ • Read task from dataset                    │
   │ • Generate prompts with augmentation        │
   │ • Tokenize prompts → input_ids              │
   │ • Batch input_ids (batch_size=16)           │
   └─────────────────────────────────────────────┘

4. ACTIVATION EXTRACTION (distributed across 32 chips):
   ┌─────────────────────────────────────────────┐
   │ • Load batch to TPU memory                  │
   │ • Forward pass (sharded computation)        │
   │ • Extract activations from target layers    │
   │ • Copy activations to CPU memory            │
   └─────────────────────────────────────────────┘

5. BUFFERING (per host):
   ┌─────────────────────────────────────────────┐
   │ Buffer: {layer_10: [...], layer_11: [...]}  │
   │ Size: Accumulates until ~1 GB               │
   └─────────────────────────────────────────────┘

6. SHARD SAVING (when buffer full):
   ┌─────────────────────────────────────────────┐
   │ • Pickle + compress buffer                  │
   │ • Save to local disk: shard_XXXX.pkl.gz     │
   │ • Upload to GCS: gs://.../machine_X_host_Y/ │
   │ • (Optional) Delete local copy              │
   │ • Clear buffer                              │
   └─────────────────────────────────────────────┘

7. REPEAT steps 3-6 until all tasks processed
```

## Storage Costs

### GCS Storage Cost Breakdown

For **10,000 tasks** with **8 predictions per task** = **80,000 samples**:

```
Activation Size per Sample:
├── Qwen 2.5 7B (hidden_size=3584)
├── 14 layers extracted (layers 14-27)
├── Sequence length: 2048 tokens
└── Size: 14 layers × 2048 × 3584 × 4 bytes (float32) = 406 MB per sample

Total Size (uncompressed): 80,000 samples × 406 MB = 32.5 TB

After Compression (gzip, ~10:1 ratio):
└── Total: 3.25 TB

Storage Cost:
├── GCS Standard: $0.020/GB/month
├── Total: 3,250 GB × $0.020 = $65/month
└── (Consider GCS Nearline: $0.010/GB/month = $32.50/month)
```

**Optimization**: Extract fewer layers (e.g., last 4 layers only):
```
4 layers × 2048 × 3584 × 4 bytes = 116 MB per sample
Total: 80,000 × 116 MB = 9.3 TB uncompressed → 930 GB compressed
Cost: 930 GB × $0.020 = $18.60/month
```

### Disk Space on TPU Hosts

```
Per Host (temporary, during extraction):
├── Dataset: ~45 MB
├── Model cache: ~14 GB
├── Code: ~100 MB
├── Logs: ~1 GB
├── Activation shards (before upload): ~20-50 GB
└── Total: ~35-65 GB

Recommended TPU Disk Size: 100 GB per host
```

## Data Redundancy and Fault Tolerance

### Current Implementation

**NO REDUNDANCY** - Data is stored once per host:
```
Host 0 → gs://.../machine_000_host_00/  (no backup)
Host 1 → gs://.../machine_000_host_01/  (no backup)
Host 2 → gs://.../machine_000_host_02/  (no backup)
Host 3 → gs://.../machine_000_host_03/  (no backup)
```

**If a host fails**:
- Data from that host is lost
- Need to re-run extraction for that host's tasks

**GCS Redundancy**:
- GCS automatically provides redundancy (regional or multi-regional)
- But if upload fails, data is lost

### Recommended Improvements

1. **Checkpoint Intermediate State**:
   ```python
   # Save progress every N batches
   checkpoint = {
       'last_processed_task': task_idx,
       'buffer_state': buffer,
       'completed_shards': shard_count
   }
   # Resume from checkpoint if interrupted
   ```

2. **Verify GCS Upload**:
   ```python
   # After upload, verify file exists
   uploaded_size = gsutil.stat(gcs_path)
   if uploaded_size != local_size:
       retry_upload()
   ```

3. **Use Preemptible with Restart Logic**:
   ```bash
   # Automatically restart if preempted
   while ! extraction_complete; do
       launch_v5e64.sh --resume-from-checkpoint
   done
   ```

## Summary

### Key Takeaways

1. **Dataset**:
   - Stored in GCS (single source of truth)
   - Replicated to all hosts (small, ~45 MB)
   - Sharded IN MEMORY by host (not on disk)

2. **Model**:
   - Downloaded from HuggingFace to each host
   - Sharded across ALL 32 TPU chips (FSDP)
   - Each chip holds ~1/32 of model parameters

3. **Activations**:
   - Computed in TPU HBM
   - Buffered in CPU RAM
   - Saved to local disk (temporary)
   - Uploaded to GCS (permanent, host-specific paths)
   - Optionally deleted from local disk

4. **Distribution**:
   - Each host processes different tasks (round-robin)
   - Each host saves to separate GCS path
   - NO data sharing between hosts during extraction

5. **Storage Costs**:
   - ~$18-65/month for 10K tasks (depends on layers extracted)
   - Use GCS Nearline for 50% savings on storage
   - Delete local copies to save TPU disk space

6. **Fault Tolerance**:
   - Currently: No checkpoint/resume
   - Recommended: Add checkpointing for preemptible TPUs
