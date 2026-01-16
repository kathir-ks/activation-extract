#!/bin/bash
#
# Europe Region 2: europe-west4-a Deployment Configuration
#
# Region: europe-west4-a
# TPU Type: v6e-8
# Workers: 8 (64 chips / 8 chips per host)
# Streams: 24-31
# GCS Bucket: fineweb-data-europe-west4
#

set -e

echo "=========================================="
echo "Europe Region 2 Deployment (europe-west4-a)"
echo "=========================================="
echo "TPU Type: v6e-8"
echo "Workers: 8"
echo "Stream Range: 24-31"
echo "Bucket: gs://fineweb-data-europe-west4"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Deploy with monitoring
./scripts/deploy_to_tpus.sh \
  --gcs_bucket fineweb-data-europe-west4 \
  --zones europe-west4-a \
  --workers_per_zone 8 \
  --tpu_type v6e-8 \
  --model KathirKs/qwen-2.5-0.5b \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --stream_offset 24 \
  --skip_dataset \
  --create_tpus \
  --monitor
