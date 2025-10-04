#!/bin/bash
# Test script for distributed inference on single-host TPU
# All outputs are logged to logs/ directory

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/kathirks_gc/torch_xla/qwen/logs"
LOG_FILE="${LOG_DIR}/test_inference_${TIMESTAMP}.log"

echo "Starting test inference at $(date)" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Run inference with all output redirected to log
cd /home/kathirks_gc/torch_xla/qwen

python distributed_inference_with_activations.py \
  --model_path Qwen/Qwen2.5-0.5B \
  --dataset_path test_data_small.json \
  --output_filepath test_outputs/predictions_${TIMESTAMP}.json \
  --activations_dir test_activations/run_${TIMESTAMP} \
  --batch_size 2 \
  --mesh_shape 2 2 \
  --n_tasks 2 \
  --max_output_tokens 50 \
  --predictions_per_task 2 \
  --layers_to_extract 10 11 12 \
  --extract_activations \
  2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "========================================" | tee -a "${LOG_FILE}"
echo "Test inference completed at $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
