#!/bin/bash

LOG_FILE="logs/perf_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Performance Test Started at $(date) ===" | tee $LOG_FILE
START_TIME=$(date +%s)

python distributed_inference_with_activations.py \
  --model_path KathirKs/qwen-2.5-0.5b \
  --dataset_path test_data_small.json \
  --output_filepath test_outputs/perf_test.json \
  --activations_dir test_activations/perf_test \
  --batch_size 2 \
  --n_tasks 1 \
  --max_output_tokens 100 \
  --predictions_per_task 1 \
  --grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
  --prompt_version output-from-examples-v0 \
  --extract_activations \
  --layers_to_extract 10 11 12 \
  --mesh_shape 2 2 2>&1 | tee -a $LOG_FILE

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a $LOG_FILE
echo "=== Performance Test Completed at $(date) ===" | tee -a $LOG_FILE
echo "Duration: $DURATION seconds" | tee -a $LOG_FILE

# Calculate tokens/second
NUM_PREDICTIONS=$(grep -oP "Creating solutions from \K\d+" $LOG_FILE | head -1)
if [ ! -z "$NUM_PREDICTIONS" ]; then
    TOTAL_TOKENS=$((100 * NUM_PREDICTIONS))
    TOKENS_PER_SEC=$(echo "scale=2; $TOTAL_TOKENS / $DURATION" | bc)
    echo "Predictions: $NUM_PREDICTIONS" | tee -a $LOG_FILE
    echo "Total Tokens: $TOTAL_TOKENS" | tee -a $LOG_FILE
    echo "Tokens/Second: $TOKENS_PER_SEC" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    
    # Compare with old performance
    OLD_RATE=0.64
    SPEEDUP=$(echo "scale=1; $TOKENS_PER_SEC / $OLD_RATE" | bc)
    echo "OLD Performance: $OLD_RATE tok/s" | tee -a $LOG_FILE
    echo "NEW Performance: $TOKENS_PER_SEC tok/s" | tee -a $LOG_FILE
    echo "Speedup: ${SPEEDUP}x" | tee -a $LOG_FILE
fi

echo "Log saved to: $LOG_FILE"
