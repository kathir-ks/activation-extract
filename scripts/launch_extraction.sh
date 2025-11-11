#!/bin/bash
# Launch activation extraction on all TPU machines

set -e

# Configuration
ZONE="${ZONE:-us-central1-a}"
TOTAL_MACHINES="${TOTAL_MACHINES:-32}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
LAYERS="${LAYERS:-15 16 17 18}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
GCS_BUCKET="${GCS_BUCKET:-your-activations-bucket}"
GCS_PREFIX="${GCS_PREFIX:-qwen_7b_fineweb}"
MESH_TYPE="${MESH_TYPE:-1d}"

echo "=================================================="
echo "LAUNCHING EXTRACTION ON ${TOTAL_MACHINES} MACHINES"
echo "=================================================="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Layers: ${LAYERS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max sequence length: ${MAX_SEQ_LEN}"
echo "Max samples: ${MAX_SAMPLES}"
echo "GCS: gs://${GCS_BUCKET}/${GCS_PREFIX}/"
echo ""

for i in $(seq 0 $((TOTAL_MACHINES - 1))); do
  echo "[$(date +%H:%M:%S)] Starting extraction on machine $i..."

  gcloud compute ssh tpu-vm-$i --zone="${ZONE}" --command="
    cd ~/qwen
    mkdir -p logs
    nohup python extract_activations_fineweb_multihost.py \
      --machine_id $i \
      --total_machines ${TOTAL_MACHINES} \
      --model_path '${MODEL}' \
      --dataset_name '${DATASET}' \
      --dataset_config 'sample-10BT' \
      --dataset_split 'train' \
      --layers_to_extract ${LAYERS} \
      --batch_size ${BATCH_SIZE} \
      --max_seq_length ${MAX_SEQ_LEN} \
      --max_samples ${MAX_SAMPLES} \
      --mesh_type ${MESH_TYPE} \
      --upload_to_gcs \
      --gcs_bucket '${GCS_BUCKET}' \
      --gcs_prefix '${GCS_PREFIX}/machine_$i' \
      --shard_size_gb 0.5 \
      --compress_shards \
      --delete_local_after_upload \
      --verbose \
      > logs/extraction_$i.log 2>&1 &

    echo 'Process started on machine $i (PID: \$!)'
  " &

  # Stagger starts to avoid overwhelming HuggingFace servers
  sleep 2
done

wait
echo ""
echo "✓ All machines started!"
echo ""
echo "Monitor with: ./scripts/monitor_progress.sh"
echo "Check status: ./scripts/check_completion.sh"
