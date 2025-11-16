# Dataset Creation and GCS Upload Guide

## Dataset Format

Your extraction pipeline uses **ARC-AGI format** (JSONL):

```json
{
  "task_id": "task_00000000",
  "train": [
    {"input": [[1,5,1],[5,1,5]], "output": [[1,1,1],[5,1,5]]},
    {"input": [[5,6,5],[6,5,6]], "output": [[5,5,5],[6,5,6]]}
  ],
  "test": [
    {"input": [[6,1,6],[1,6,1]]}
  ]
}
```

Each line = 1 task with training examples and a test case.

---

## Option 1: Convert from HuggingFace Dataset

### Using the Provided Script

The repo includes `convert_hf_to_arc_format.py` for converting HuggingFace datasets:

```bash
# Basic conversion (convert all tasks)
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --column_name examples \
  --output_file dataset.jsonl \
  --verbose

# Limited conversion (1000 tasks for testing)
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --column_name examples \
  --output_file dataset_1k.jsonl \
  --max_tasks 1000 \
  --verbose

# Sharded conversion (for distributed processing)
# Worker 0: tasks 0-50000
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY \
  --output_file shard_0.jsonl \
  --start_index 0 \
  --end_index 50000 \
  --verbose

# Worker 1: tasks 50000-100000
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY \
  --output_file shard_1.jsonl \
  --start_index 50000 \
  --end_index 100000 \
  --verbose
```

### Popular HuggingFace Datasets

```bash
# barc0's synthetic ARC datasets
barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
barc0/arc_agi_training_challenges_solutions_combined

# Other ARC datasets
arc-agi/arc-agi-training
```

---

## Option 2: Create Custom Dataset

If you have your own data, create a JSONL file with this structure:

```python
import json

tasks = []
for i in range(100):
    task = {
        "task_id": f"task_{i:08x}",
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[1, 1], [1, 1]]
            },
            # More training examples...
        ],
        "test": [
            {"input": [[0, 1], [1, 0]]}
        ]
    }
    tasks.append(task)

# Write as JSONL (one JSON object per line)
with open('custom_dataset.jsonl', 'w') as f:
    for task in tasks:
        f.write(json.dumps(task) + '\n')
```

---

## Upload Dataset to Google Cloud Storage

### 1. Create GCS Bucket (One-Time Setup)

```bash
# Set your project
export PROJECT_ID="absolute-axis-470415-g6"
export BUCKET_NAME="arc-datasets-us-central1"
export REGION="us-central1"

# Create bucket in same region as TPUs
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${BUCKET_NAME}
```

### 2. Upload Dataset

```bash
# Upload single file
gsutil cp dataset.jsonl gs://${BUCKET_NAME}/

# Upload with progress bar (large files)
gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp dataset.jsonl gs://${BUCKET_NAME}/

# Upload entire directory
gsutil -m cp -r datasets/ gs://${BUCKET_NAME}/

# Verify upload
gsutil ls -lh gs://${BUCKET_NAME}/
```

### 3. Set Permissions (Optional)

```bash
# Make bucket accessible to TPU service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:service-PROJECT_NUMBER@cloud-tpu.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

---

## Using Dataset in Extraction Script

Once uploaded, use the GCS path in your extraction command:

```bash
# Full GCS path
export DATASET="gs://arc-datasets-us-central1/dataset.jsonl"

# In Docker command
sudo docker run -d --name extraction --net=host --privileged \
  -v ~/data:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "python /workspace/extract_activations_arc_v5e64.py \
    --dataset_path ${DATASET} \
    --model_path KathirKs/qwen-2.5-0.5b \
    --batch_size 4 \
    --gcs_bucket ${BUCKET_NAME} \
    --upload_to_gcs \
    --verbose"
```

---

## Quick Start Examples

### Example 1: Small Test Dataset

```bash
# 1. Convert 10 tasks from HuggingFace
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --column_name examples \
  --output_file test_10tasks.jsonl \
  --max_tasks 10 \
  --verbose

# 2. Upload to GCS
gsutil cp test_10tasks.jsonl gs://your-bucket/

# 3. Run extraction
python extract_activations_arc_v5e64.py \
  --dataset_path gs://your-bucket/test_10tasks.jsonl \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --max_tasks 10 \
  --output_dir /tmp/test_output \
  --verbose
```

### Example 2: Production Run (1000 tasks)

```bash
# 1. Convert on local machine
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_file dataset_1k.jsonl \
  --max_tasks 1000 \
  --verbose

# 2. Upload to GCS
gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
  cp dataset_1k.jsonl gs://your-bucket/

# 3. Deploy to TPU v5e-64 (see QUICK_START.md)
export DATASET="gs://your-bucket/dataset_1k.jsonl"
export BUCKET="your-bucket"
export TPU_NAME="arc-extraction"

# ... (follow deployment commands in QUICK_START.md)
```

---

## Dataset Statistics

Check dataset size before extraction:

```bash
# Count tasks
wc -l dataset.jsonl

# File size
ls -lh dataset.jsonl

# Preview first task
head -1 dataset.jsonl | python3 -m json.tool

# Check training examples per task (using jq)
head -10 dataset.jsonl | jq '.train | length'
```

---

## Troubleshooting

### Issue: "Permission denied" when accessing GCS

**Solution:**
```bash
# Authenticate
gcloud auth login

# Or use application default credentials
gcloud auth application-default login
```

### Issue: "Dataset file not found"

**Solution:**
```bash
# Verify file exists
gsutil ls gs://your-bucket/dataset.jsonl

# Check permissions
gsutil acl get gs://your-bucket/dataset.jsonl
```

### Issue: Dataset conversion fails

**Solution:**
```bash
# Check HuggingFace dataset structure
python -c "from datasets import load_dataset; ds = load_dataset('barc0/200k_HEAVY', split='train', streaming=True); print(next(iter(ds)))"

# Verify column name
python -c "from datasets import load_dataset; ds = load_dataset('barc0/200k_HEAVY', split='train', streaming=True); print(next(iter(ds)).keys())"
```

---

## File Locations

- **Conversion script:** `convert_hf_to_arc_format.py`
- **Test dataset:** `test_gcs_dataset.jsonl` (2 tasks, for testing)
- **Output location:** `gs://your-bucket/activations/` (automatic)

---

## Creating Sharded Datasets for Distributed Processing

For large-scale extraction with multiple machines, shard your dataset to enable automatic workload distribution:

### Why Shard?

**Benefits:**
- ✅ Each machine automatically claims an available shard (no manual coordination)
- ✅ Prevents duplicate work (metadata tracks which shards are in use/completed)
- ✅ 100MB chunks enable efficient loading and processing
- ✅ Resume failed jobs (mark shard as available again)

### Create Sharded Dataset

```bash
# Convert HuggingFace dataset
python convert_hf_to_arc_format.py \
  --dataset_name barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_file dataset_full.jsonl \
  --verbose

# Shard into 8 parts with 100MB chunks
python create_sharded_dataset.py \
  --input_file dataset_full.jsonl \
  --output_dir sharded_dataset \
  --num_shards 8 \
  --chunk_size_mb 100 \
  --verbose

# Upload to GCS
gsutil -m cp -r sharded_dataset gs://your-bucket/
```

### Sharded Dataset Structure

```
gs://your-bucket/sharded_dataset/
├── master_metadata.json          # Overall dataset info
├── shard_000/
│   ├── metadata.json             # Shard status: available/in_use/completed
│   ├── chunk_0000.jsonl          # ~100MB
│   ├── chunk_0001.jsonl          # ~100MB
│   └── chunk_XXXX.jsonl
├── shard_001/
│   ├── metadata.json
│   └── chunk_*.jsonl
└── ... (6 more shards)
```

### Use Sharded Dataset in Extraction

Each machine automatically claims a shard:

```bash
# On Machine 0 (or worker 0)
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://your-bucket/sharded_dataset \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --gcs_bucket your-bucket \
  --upload_to_gcs \
  --verbose

# On Machine 1 (runs in parallel, claims different shard)
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://your-bucket/sharded_dataset \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --gcs_bucket your-bucket \
  --upload_to_gcs \
  --verbose

# ... Machines 2-7 (each auto-claims a shard)
```

### Monitoring Shards

```bash
# Check shard status
python shard_manager.py \
  --dataset_dir gs://your-bucket/sharded_dataset \
  --action status

# Output:
#   Shard 0: in_use (assigned to: machine0_host0)
#   Shard 1: in_use (assigned to: machine1_host0)
#   Shard 2: completed
#   Shard 3: available
#   ...
```

### Reset Failed Shard

If a machine fails, reset the shard to make it available again:

```bash
python shard_manager.py \
  --dataset_dir gs://your-bucket/sharded_dataset \
  --action reset \
  --shard_id 5
```

### Multi-Host TPU v5e-64 with Sharding

For TPU v5e-64 (4 hosts), each host can claim a shard:

```bash
# Deploy to all 4 workers with sharding
for WORKER in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=${WORKER} --command="
    sudo docker run -d --name extraction --net=host --privileged \
      -v ~/data:/workspace/data \
      -v ~/.config/gcloud:/root/.config/gcloud:ro \
      gcr.io/${PROJECT_ID}/activation-extraction:latest \
      -c 'python /workspace/extract_activations_arc_v5e64.py \
        --use_sharded_dataset \
        --sharded_dataset_dir gs://${BUCKET}/sharded_dataset \
        --host_id ${WORKER} --num_hosts 4 --multihost \
        --coordinator_address ${COORDINATOR_IP}:8476 \
        --model_path KathirKs/qwen-2.5-0.5b \
        --batch_size 4 \
        --gcs_bucket ${BUCKET} --upload_to_gcs \
        --verbose'
  " &
done
wait
```

---

## Next Steps

**Option A: Single Dataset (Simple)**
1. ✅ Create/convert your dataset → `dataset.jsonl`
2. ✅ Upload to GCS → `gs://your-bucket/dataset.jsonl`
3. ✅ Build Docker image (see `QUICK_START.md`)
4. ✅ Deploy to TPU and run extraction (see `QUICK_START.md`)
5. ✅ Monitor progress → `gsutil ls -lh gs://your-bucket/activations/`

**Option B: Sharded Dataset (Distributed)**
1. ✅ Create/convert your dataset → `dataset.jsonl`
2. ✅ Shard dataset → `python create_sharded_dataset.py`
3. ✅ Upload to GCS → `gsutil cp -r sharded_dataset gs://bucket/`
4. ✅ Build Docker image (see `QUICK_START.md`)
5. ✅ Deploy to multiple machines with `--use_sharded_dataset`
6. ✅ Monitor shard status → `python shard_manager.py --action status`

---

## Performance Notes

**Dataset size impact:**
- 1000 tasks × 8 prompts = 8,000 samples
- Batch size 4 → 2,000 batches
- ~5.7s per batch → ~3.2 hours on 4-host TPU v5e-64
- Output: ~728 MB per shard (compressed)

**Recommended batch sizes:**
- TPU v4: batch_size=2-4
- TPU v5e-64: batch_size=4-8
- Larger batches = better throughput (but more memory)
