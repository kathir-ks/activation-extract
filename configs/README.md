# Multi-Region Deployment Configurations

## Overview

Total capacity: 32 TPU hosts across 4 regions
- **us-central1-a**: 8 x v5e-8 (64 chips)
- **europe-west4-b**: 8 x v5e-8 (64 chips)
- **us-east1-d**: 8 x v6e-8 (64 chips)
- **europe-west4-a**: 8 x v6e-8 (64 chips)

**Total: 32 workers processing 256 chips**

## Stream Assignment (CRITICAL for avoiding duplicates)

Each region is assigned unique stream ranges:

| Region           | TPU Type | Workers | Stream Range | GCS Bucket                    |
|------------------|----------|---------|--------------|-------------------------------|
| us-central1-a    | v5e-8    | 8       | 0-7          | fineweb-data-us-central1      |
| europe-west4-b   | v5e-8    | 8       | 8-15         | fineweb-data-europe-west4     |
| us-east1-d       | v6e-8    | 8       | 16-23        | fineweb-data-us-central1      |
| europe-west4-a   | v6e-8    | 8       | 24-31        | fineweb-data-europe-west4     |

**This ensures NO data duplication across regions.**

## Deployment Steps

### Step 1: Create ALL Dataset Streams (Run ONCE)

```bash
# Create 32 streams for all regions
python create_dataset_streams.py \
  --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
  --num_streams 32 \
  --output_dir ./dataset_streams \
  --verbose
```

### Step 2: Upload Streams to BOTH Buckets

```bash
# Upload to US bucket (streams 0-7 and 16-23 will be used)
./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-us-central1 \
  --prefix datasets \
  --local_dir ./dataset_streams

# Upload to Europe bucket (streams 8-15 and 24-31 will be used)
./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-europe-west4 \
  --prefix datasets \
  --local_dir ./dataset_streams
```

### Step 3: Deploy to Each Region

Deploy to each region in any order (they're independent):

```bash
# US Region 1 - v5e (streams 0-7)
bash configs/deploy_us_central1_v5e.sh

# Europe Region 1 - v5e (streams 8-15)
bash configs/deploy_europe_west4_v5e.sh

# US Region 2 - v6e (streams 16-23)
bash configs/deploy_us_east1_v6e.sh

# Europe Region 2 - v6e (streams 24-31)
bash configs/deploy_europe_west4_v6e.sh
```

**Or use the master script to deploy all at once:**

```bash
# Deploy to all 4 regions sequentially
bash configs/deploy_all_regions.sh
```

## Monitoring

Each regional deployment runs with `--monitor` flag, showing:
- Live progress dashboard per region
- Automatic preemption recovery
- Real-time sample/shard counts

You can run all 4 in separate terminals to monitor all regions simultaneously.

## Output Structure

Activations are stored per-worker in region-specific buckets:

**US Bucket: gs://fineweb-data-us-central1/activations/**
```
tpu_0/    # us-central1-a worker 0 (stream 0)
tpu_1/    # us-central1-a worker 1 (stream 1)
...
tpu_7/    # us-central1-a worker 7 (stream 7)
tpu_16/   # us-east1-d worker 0 (stream 16)
tpu_17/   # us-east1-d worker 1 (stream 17)
...
tpu_23/   # us-east1-d worker 7 (stream 23)
```

**Europe Bucket: gs://fineweb-data-europe-west4/activations/**
```
tpu_8/    # europe-west4-b worker 0 (stream 8)
tpu_9/    # europe-west4-b worker 1 (stream 9)
...
tpu_15/   # europe-west4-b worker 7 (stream 15)
tpu_24/   # europe-west4-a worker 0 (stream 24)
tpu_25/   # europe-west4-a worker 1 (stream 25)
...
tpu_31/   # europe-west4-a worker 7 (stream 31)
```

## Verification

To verify no duplicate processing:

```bash
# Check that each stream is processed exactly once
for i in {0..31}; do
  echo "Stream $i:"
  # Find which worker processed it
  gsutil ls gs://fineweb-data-us-central1/activations/tpu_$i/ 2>/dev/null || \
  gsutil ls gs://fineweb-data-europe-west4/activations/tpu_$i/ 2>/dev/null || \
  echo "  Not found"
done
```

Each stream should appear in exactly one worker's output directory.
