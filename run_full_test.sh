#!/bin/bash

# Configuration
OUTPUT_FILE="test_outputs/production_predictions.json"
ACTIVATIONS_DIR="test_activations/production_run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/production_run_$(date +%Y%m%d_%H%M%S).log"
TIMING_FILE="logs/timing_$(date +%Y%m%d_%H%M%S).txt"

# Create directories
mkdir -p test_outputs test_activations logs

# Record start time
START_TIME=$(date +%s)
echo "=== Production Run Started at $(date) ===" | tee $TIMING_FILE

# Run the inference with time measurement
/usr/bin/time -v python distributed_inference_with_activations.py \
  --model_path KathirKs/qwen-2.5-0.5b \
  --dataset_path test_data_small.json \
  --output_filepath $OUTPUT_FILE \
  --activations_dir $ACTIVATIONS_DIR \
  --batch_size 2 \
  --n_tasks 2 \
  --max_output_tokens 500 \
  --predictions_per_task 2 \
  --grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
  --prompt_version output-from-examples-v0 \
  --extract_activations \
  --layers_to_extract 10 11 12 \
  --mesh_shape 2 2 2>&1 | tee $LOG_FILE

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate statistics
echo "" | tee -a $TIMING_FILE
echo "=== Production Run Completed at $(date) ===" | tee -a $TIMING_FILE
echo "Total Duration: $DURATION seconds ($((DURATION / 60)) minutes)" | tee -a $TIMING_FILE

# Calculate tokens/second (assuming 500 max tokens * number of predictions)
# Get number of predictions from log
NUM_PREDICTIONS=$(grep -oP "Creating solutions from \K\d+" $LOG_FILE | head -1)
if [ ! -z "$NUM_PREDICTIONS" ]; then
    TOTAL_TOKENS=$((500 * NUM_PREDICTIONS))
    TOKENS_PER_SEC=$(echo "scale=2; $TOTAL_TOKENS / $DURATION" | bc)
    echo "Total Predictions: $NUM_PREDICTIONS" | tee -a $TIMING_FILE
    echo "Approximate Total Tokens: $TOTAL_TOKENS" | tee -a $TIMING_FILE
    echo "Tokens per Second: $TOKENS_PER_SEC" | tee -a $TIMING_FILE
fi

# Check parsing success
PARSE_FAILURES=$(grep -oP "Failed to parse \K\d+" $LOG_FILE | head -1)
if [ ! -z "$PARSE_FAILURES" ]; then
    echo "Parse Failures: $PARSE_FAILURES/$NUM_PREDICTIONS" | tee -a $TIMING_FILE
    SUCCESS_RATE=$(echo "scale=2; 100 * ($NUM_PREDICTIONS - $PARSE_FAILURES) / $NUM_PREDICTIONS" | bc)
    echo "Parse Success Rate: $SUCCESS_RATE%" | tee -a $TIMING_FILE
fi

# Show mesh verification
echo "" | tee -a $TIMING_FILE
echo "=== Mesh Verification ===" | tee -a $TIMING_FILE
grep -A3 "Verifying parameter sharding:" $LOG_FILE | head -5 | tee -a $TIMING_FILE
grep "Input sharding:" $LOG_FILE | head -1 | tee -a $TIMING_FILE

echo "" | tee -a $TIMING_FILE
echo "Log saved to: $LOG_FILE" | tee -a $TIMING_FILE
echo "Timing stats saved to: $TIMING_FILE" | tee -a $TIMING_FILE
echo "Predictions saved to: $OUTPUT_FILE" | tee -a $TIMING_FILE
echo "Activations saved to: $ACTIVATIONS_DIR" | tee -a $TIMING_FILE
