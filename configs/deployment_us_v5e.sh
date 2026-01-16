#!/bin/bash
#
# US Region Deployment Configuration - TPU v5e-8
#
# Available quota: 64 spot v5e chips in us-central1-a (8 hosts)
# GCS Bucket: fineweb-data-us-central1
#

# Configuration
export GCS_BUCKET="fineweb-data-us-central1"
export ZONES="us-central1-a"
export WORKERS_PER_ZONE=8
export TPU_TYPE="v5e-8"
export MODEL="Qwen/Qwen2.5-0.5B"
export DATASET_NAME="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"

echo "=========================================="
echo "US Region Deployment Configuration"
echo "=========================================="
echo "Bucket: gs://$GCS_BUCKET"
echo "Zones: $ZONES"
echo "Workers: $WORKERS_PER_ZONE per zone"
echo "TPU Type: $TPU_TYPE"
echo "Total TPUs: $WORKERS_PER_ZONE"
echo ""

# Deploy with monitoring
./scripts/deploy_to_tpus.sh \
  --gcs_bucket "$GCS_BUCKET" \
  --zones "$ZONES" \
  --workers_per_zone $WORKERS_PER_ZONE \
  --tpu_type "$TPU_TYPE" \
  --model "$MODEL" \
  --dataset "$DATASET_NAME" \
  --create_tpus \
  --monitor
