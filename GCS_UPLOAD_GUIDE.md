# GCS Upload Guide for Activation Extraction

This guide explains how to use the GCS (Google Cloud Storage) upload feature with automatic sharding in the activation extraction pipeline.

## Features

✅ **Automatic Sharding**: Activations are automatically split into configurable-sized shards
✅ **Compression**: Optional gzip compression to reduce storage costs
✅ **Streaming Upload**: Uses fsspec for efficient streaming to GCS
✅ **Local Cleanup**: Optional deletion of local files after successful upload
✅ **Detailed Metadata**: JSON metadata tracking all shards and their contents

---

## Quick Start

### 1. Basic Usage (Local Only)

Extract activations without GCS upload:

```bash
python extract_activations_arc.py \
  --dataset_path test_data_small.json \
  --output_dir ./activations_arc \
  --shard_size_gb 1.0 \
  --verbose
```

### 2. Upload to GCS

Extract and upload to Google Cloud Storage:

```bash
python extract_activations_arc.py \
  --dataset_path test_data_small.json \
  --output_dir ./activations_arc \
  --upload_to_gcs \
  --gcs_bucket my-activation-bucket \
  --gcs_prefix arc_activations \
  --shard_size_gb 2.0 \
  --compress_shards \
  --verbose
```

### 3. Upload and Delete Local Files

Save storage space by deleting local files after upload:

```bash
python extract_activations_arc.py \
  --dataset_path test_data_small.json \
  --upload_to_gcs \
  --gcs_bucket my-activation-bucket \
  --gcs_prefix arc_activations \
  --shard_size_gb 1.0 \
  --compress_shards \
  --delete_local_after_upload \
  --verbose
```

---

## Configuration Options

### GCS Upload Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--upload_to_gcs` | flag | `False` | Enable GCS upload |
| `--gcs_bucket` | string | `None` | GCS bucket name (required if upload enabled) |
| `--gcs_prefix` | string | `activations` | Prefix/folder path in bucket |
| `--shard_size_gb` | float | `1.0` | Size of each shard in GB |
| `--compress_shards` | flag | `True` | Compress shards with gzip |
| `--no_compress_shards` | flag | - | Disable compression |
| `--delete_local_after_upload` | flag | `False` | Delete local files after GCS upload |

### Other Useful Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | string | `KathirKs/qwen-2.5-0.5b` | HuggingFace model path |
| `--dataset_path` | string | `test_data_small.json` | Path to ARC dataset |
| `--output_dir` | string | `./activations_arc` | Local output directory |
| `--n_tasks` | int | `None` | Limit number of tasks to process |
| `--batch_size` | int | `8` | Batch size per device |
| `--layers_to_extract` | int[] | `[10-23]` | Layer indices to extract |
| `--verbose` | flag | `False` | Print detailed progress |
| `--no_data_parallel` | flag | - | Disable TPU data parallelism |

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install fsspec gcsfs google-cloud-storage
```

### 2. Authenticate with GCS

**Option A: Application Default Credentials (Recommended for GCP VMs)**
```bash
gcloud auth application-default login
```

**Option B: Service Account Key**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option C: Already Authenticated on GCP**
If running on a GCP VM with appropriate IAM roles, no additional authentication needed.

### 3. Create GCS Bucket

```bash
# Create bucket
gcloud storage buckets create gs://my-activation-bucket --location=us-central1

# Grant permissions (if needed)
gcloud storage buckets add-iam-policy-binding gs://my-activation-bucket \
  --member=serviceAccount:your-service-account@project.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
```

---

## Output Structure

### Local Directory Structure

```
activations_arc/
├── shard_0001.pkl.gz    # Compressed activation shard 1
├── shard_0002.pkl.gz    # Compressed activation shard 2
├── ...
├── shard_NNNN.pkl.gz    # Last shard
└── metadata.json        # Metadata about all shards
```

### GCS Structure

```
gs://my-activation-bucket/
└── arc_activations/
    ├── shard_0001.pkl.gz
    ├── shard_0002.pkl.gz
    ├── ...
    └── shard_NNNN.pkl.gz
```

**Note**: `metadata.json` is only stored locally, not uploaded to GCS.

---

## Metadata Format

The `metadata.json` file contains:

```json
{
  "total_shards": 50,
  "total_samples": 1000,
  "shard_size_gb": 2.0,
  "upload_to_gcs": true,
  "gcs_bucket": "my-activation-bucket",
  "gcs_prefix": "arc_activations",
  "shards": [
    {
      "shard_id": 1,
      "filename": "shard_0001.pkl.gz",
      "local_path": "./activations_arc/shard_0001.pkl.gz",
      "gcs_path": "gs://my-activation-bucket/arc_activations/shard_0001.pkl.gz",
      "file_size_mb": 1876.5,
      "buffer_size_mb": 2048.0,
      "layers": [10, 11, 12, 13],
      "samples_per_layer": {
        "10": 5,
        "11": 5,
        "12": 5,
        "13": 5
      },
      "total_samples_in_shard": 20
    },
    ...
  ]
}
```

---

## Reading Shards

### From Local Storage

```python
import pickle
import gzip

# Read compressed shard
with gzip.open('activations_arc/shard_0001.pkl.gz', 'rb') as f:
    shard_data = pickle.load(f)

# shard_data is a dict: {layer_idx: [list of activation samples]}
for layer_idx, activations in shard_data.items():
    print(f"Layer {layer_idx}: {len(activations)} samples")
    for act in activations:
        print(f"  Task: {act['task_id']}, Shape: {act['shape']}")
        # act['activation'] is the numpy array
```

### From GCS Using fsspec

```python
import pickle
import gzip
import fsspec

# Open GCS file using fsspec
fs = fsspec.filesystem('gs')
with fs.open('my-activation-bucket/arc_activations/shard_0001.pkl.gz', 'rb') as f:
    with gzip.open(f, 'rb') as gz:
        shard_data = pickle.load(gz)

# Process data
for layer_idx, activations in shard_data.items():
    print(f"Layer {layer_idx}: {len(activations)} samples")
```

### From GCS Using gsutil

```bash
# Download specific shard
gsutil cp gs://my-activation-bucket/arc_activations/shard_0001.pkl.gz .

# Download all shards
gsutil -m cp gs://my-activation-bucket/arc_activations/*.pkl.gz ./local_shards/
```

---

## Best Practices

### 1. Shard Size Selection

- **Small datasets (<10GB)**: Use 1-2 GB shards
- **Medium datasets (10-100GB)**: Use 2-5 GB shards
- **Large datasets (>100GB)**: Use 5-10 GB shards

Larger shards = fewer files but longer upload times per file.

### 2. Compression

Always enable compression (`--compress_shards`) to:
- Reduce storage costs (typically 70-80% smaller)
- Reduce network transfer time
- Reduce GCS egress costs when downloading

### 3. Local Cleanup

Use `--delete_local_after_upload` when:
- Limited local disk space
- Running on preemptible/spot instances
- GCS is your primary storage

Don't use it when:
- You want local backups
- Debugging upload issues
- Planning to re-process locally

### 4. Parallel Extraction

For maximum speed on TPUs:
- Keep `--use_data_parallel` enabled (default)
- Increase `--batch_size` to utilize all TPU cores
- Process multiple tasks in one run

---

## Monitoring & Troubleshooting

### Check Upload Progress

Watch the verbose output:
```
======================================================================
Saving shard 42: shard_0042.pkl.gz (~2048.5 MB)
======================================================================
  ✓ Saved locally: 1876.3 MB
    Layer 10: 25 samples
    Layer 11: 25 samples
  ⬆ Uploading to gs://my-bucket/activations/shard_0042.pkl.gz...
  ✓ Uploaded to GCS: gs://my-bucket/activations/shard_0042.pkl.gz
```

### Verify Upload

```bash
# List uploaded shards
gsutil ls gs://my-activation-bucket/arc_activations/

# Check total size
gsutil du -sh gs://my-activation-bucket/arc_activations/

# Count files
gsutil ls gs://my-activation-bucket/arc_activations/ | wc -l
```

### Common Issues

**Problem**: `Failed to initialize fsspec GCS filesystem`
**Solution**: Authenticate with `gcloud auth application-default login`

**Problem**: `Permission denied` when uploading
**Solution**: Grant `roles/storage.objectAdmin` to your service account

**Problem**: Shards are too small/large
**Solution**: Adjust `--shard_size_gb` parameter

**Problem**: Upload is slow
**Solution**:
- Check network bandwidth
- Use compression (`--compress_shards`)
- Ensure running in same region as bucket

---

## Cost Estimation

### Storage Costs (us-central1)

Assuming 1TB of activations with 75% compression:

- Uncompressed: 1024 GB × $0.020/GB = **$20.48/month**
- Compressed: 256 GB × $0.020/GB = **$5.12/month** ✅

### Network Costs

- **Upload (ingress)**: FREE
- **Download (egress)**: $0.12/GB (first 1TB from North America)

---

## Examples

### Example 1: Full Dataset Extraction with GCS Upload

```bash
python extract_activations_arc.py \
  --model_path KathirKs/qwen-2.5-0.5b \
  --dataset_path arc_full_dataset.json \
  --output_dir ./activations_full \
  --upload_to_gcs \
  --gcs_bucket arc-activations-prod \
  --gcs_prefix full_dataset_v1 \
  --shard_size_gb 5.0 \
  --compress_shards \
  --delete_local_after_upload \
  --batch_size 16 \
  --verbose
```

### Example 2: Extract Specific Layers Only

```bash
python extract_activations_arc.py \
  --dataset_path test_data.json \
  --layers_to_extract 20 21 22 23 \
  --upload_to_gcs \
  --gcs_bucket my-bucket \
  --shard_size_gb 1.0 \
  --verbose
```

### Example 3: Small Test Run

```bash
python extract_activations_arc.py \
  --dataset_path test_data_small.json \
  --n_tasks 5 \
  --upload_to_gcs \
  --gcs_bucket test-bucket \
  --gcs_prefix test_run_001 \
  --shard_size_gb 0.5 \
  --verbose
```

---

## Advanced: Parallel Processing Multiple Datasets

Process multiple datasets in parallel by running multiple instances:

```bash
# Terminal 1
python extract_activations_arc.py \
  --dataset_path train_set.json \
  --gcs_prefix train_activations \
  ...

# Terminal 2
python extract_activations_arc.py \
  --dataset_path val_set.json \
  --gcs_prefix val_activations \
  ...
```

---

## Summary

The GCS upload pipeline provides:
- ✅ Automatic sharding based on size
- ✅ Transparent fsspec-based upload
- ✅ Compression to save storage costs
- ✅ Detailed metadata tracking
- ✅ Optional local cleanup

Perfect for large-scale activation extraction on TPUs with automatic cloud backup!
