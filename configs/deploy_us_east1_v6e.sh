#!/bin/bash
#
# US Region 2: us-east1-d Deployment Configuration
#
# Region: us-east1-d
# TPU Type: v6e-8
# Workers: 8 (64 chips / 8 chips per host)
# Streams: 16-23
# GCS Bucket: fineweb-data-us-central1
#

set -e

echo "=========================================="
echo "US Region 2 Deployment (us-east1-d)"
echo "=========================================="
echo "TPU Type: v6e-8"
echo "Workers: 8"
echo "Stream Range: 16-23"
echo "Bucket: gs://fineweb-data-us-central1"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Deploy with monitoring
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-east1-d \
  --workers_per_zone 8 \
  --tpu_type v6e-8 \
  --model KathirKs/qwen-2.5-0.5b \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --stream_offset 16 \
  --skip_dataset \
  --create_tpus \
  --monitor
