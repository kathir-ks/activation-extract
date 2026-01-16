#!/bin/bash
#
# Setup Dataset Streams for Multi-Region Deployment
#
# Creates 32 streams (one per worker across all 4 regions) and uploads to both buckets
#

set -e

echo "=========================================="
echo "Multi-Region Dataset Stream Setup"
echo "=========================================="
echo "Total workers: 32"
echo "Streams to create: 32 (stream_000.jsonl to stream_031.jsonl)"
echo "Buckets: fineweb-data-us-central1, fineweb-data-europe-west4"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Step 1: Create 32 streams
echo "Step 1: Creating 32 dataset streams..."
echo ""

python create_dataset_streams.py \
  --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
  --num_streams 32 \
  --output_dir ./dataset_streams \
  --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create dataset streams"
    exit 1
fi

echo ""
echo "✓ Created 32 streams in ./dataset_streams/"
echo ""

# Step 2: Upload to US bucket
echo "Step 2: Uploading streams to US bucket (fineweb-data-us-central1)..."
echo ""

./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-us-central1 \
  --prefix datasets \
  --local_dir ./dataset_streams

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upload to US bucket"
    exit 1
fi

echo ""
echo "✓ Uploaded to gs://fineweb-data-us-central1/datasets/"
echo ""

# Step 3: Upload to Europe bucket
echo "Step 3: Uploading streams to Europe bucket (fineweb-data-europe-west4)..."
echo ""

./scripts/upload_dataset_streams_to_gcs.sh \
  --bucket fineweb-data-europe-west4 \
  --prefix datasets \
  --local_dir ./dataset_streams

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upload to Europe bucket"
    exit 1
fi

echo ""
echo "✓ Uploaded to gs://fineweb-data-europe-west4/datasets/"
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Stream assignment per region:"
echo "  us-central1-a (v5e):    streams 0-7   → gs://fineweb-data-us-central1/datasets/"
echo "  europe-west4-b (v5e):   streams 8-15  → gs://fineweb-data-europe-west4/datasets/"
echo "  us-east1-d (v6e):       streams 16-23 → gs://fineweb-data-us-central1/datasets/"
echo "  europe-west4-a (v6e):   streams 24-31 → gs://fineweb-data-europe-west4/datasets/"
echo ""
echo "Next steps:"
echo "  1. Deploy to individual regions:"
echo "     bash configs/deploy_us_central1_v5e.sh"
echo "     bash configs/deploy_europe_west4_v5e.sh"
echo "     bash configs/deploy_us_east1_v6e.sh"
echo "     bash configs/deploy_europe_west4_v6e.sh"
echo ""
echo "  2. Or deploy all at once:"
echo "     bash configs/deploy_all_regions.sh"
echo ""
