#!/bin/bash
# Upload code to all TPU machines

set -e

ZONE="${ZONE:-us-central1-a}"
TOTAL_MACHINES="${TOTAL_MACHINES:-32}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================================================="
echo "UPLOADING CODE TO ${TOTAL_MACHINES} MACHINES"
echo "=================================================="
echo "Zone: ${ZONE}"
echo "Code dir: ${CODE_DIR}"
echo ""

for i in $(seq 0 $((TOTAL_MACHINES - 1))); do
  echo "Uploading to machine $i..."
  gcloud compute scp --recurse \
    "${CODE_DIR}"/*.py \
    "${CODE_DIR}"/*.md \
    tpu-vm-$i:~/qwen/ \
    --zone="${ZONE}" &
done

wait
echo ""
echo "✓ Upload complete!"
