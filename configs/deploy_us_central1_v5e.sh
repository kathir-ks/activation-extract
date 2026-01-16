#!/bin/bash
#
# US Region 1: us-central1-a Deployment Configuration
#
# Region: us-central1-a
# TPU Type: v5e-8
# Workers: 8 (64 chips / 8 chips per host)
# Streams: 0-7
# GCS Bucket: fineweb-data-us-central1
#

set -e

echo "=========================================="
echo "US Region 1 Deployment (us-central1-a)"
echo "=========================================="
echo "TPU Type: v5e-8"
echo "Workers: 8"
echo "Stream Range: 0-7"
echo "Bucket: gs://fineweb-data-us-central1"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Deploy with monitoring
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-us-central1 \
  --zones us-central1-a \
  --workers_per_zone 8 \
  --tpu_type v5e-8 \
  --model KathirKs/qwen-2.5-0.5b \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --stream_offset 0 \
  --skip_dataset \
  --create_tpus \
  --monitor
