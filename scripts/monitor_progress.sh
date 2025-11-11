#!/bin/bash
# Monitor extraction progress across all machines

ZONE="${ZONE:-us-central1-a}"
TOTAL_MACHINES="${TOTAL_MACHINES:-32}"

while true; do
  clear
  echo "=================================================="
  echo "EXTRACTION PROGRESS ($(date))"
  echo "=================================================="
  echo ""

  running_count=0
  completed_count=0

  for i in $(seq 0 $((TOTAL_MACHINES - 1))); do
    status=$(gcloud compute ssh tpu-vm-$i --zone="${ZONE}" --command="
      if pgrep -f extract_activations_fineweb_multihost.py > /dev/null; then
        # Extract progress from log
        tail -10 ~/qwen/logs/extraction_$i.log 2>/dev/null | \
          grep -oP '\\d+it \\[[0-9:]+' | tail -1 || echo 'Starting...'
      elif [ -f ~/qwen/logs/extraction_$i.log ]; then
        if grep -q 'EXTRACTION COMPLETE' ~/qwen/logs/extraction_$i.log 2>/dev/null; then
          echo 'COMPLETED'
        else
          echo 'FAILED (check logs)'
        fi
      else
        echo 'NOT STARTED'
      fi
    " 2>/dev/null)

    if [[ "$status" == "COMPLETED" ]]; then
      ((completed_count++))
      printf "Machine %2d: ✓ %s\n" $i "$status"
    elif [[ "$status" == "NOT STARTED" ]] || [[ "$status" == "FAILED"* ]]; then
      printf "Machine %2d: ✗ %s\n" $i "$status"
    else
      ((running_count++))
      printf "Machine %2d: ⟳ %s\n" $i "$status"
    fi
  done

  echo ""
  echo "Summary: Running=$running_count, Completed=$completed_count, Total=$TOTAL_MACHINES"
  echo ""
  echo "Press Ctrl+C to exit"
  echo "Refreshing in 30 seconds..."

  sleep 30
done
