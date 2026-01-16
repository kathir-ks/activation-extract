#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tpu_name) tpu_name="$2"; shift ;;
        --tpu_type) tpu_type="$2"; shift ;;
        --region) region="$2"; shift ;;
        --preemptible) preemptible="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$tpu_name" ] || [ -z "$tpu_type" ] || [ -z "$region" ]; then
    echo "Usage: $0 --tpu_name TPU_NAME --tpu_type TPU_TYPE --region REGION [--preemptible true|false]"
    echo "Example: $0 --tpu_name main --tpu_type v3-8 --region us-central1-a --preemptible true"
    exit 1
fi

preemptible_flag=""
if [ "$preemptible" = "true" ]; then
    preemptible_flag="--preemptible"
fi

while true; do
    output=$(gcloud compute tpus tpu-vm describe $tpu_name --zone=$region 2>&1)
    if [[ $output != *"READY"* ]]; then echo "Creating TPU VM '$tpu_name'..."; gcloud compute tpus tpu-vm create $tpu_name --zone=$region --accelerator-type=$tpu_type --version=tpu-ubuntu2204-base $preemptible_flag; sleep 10; fi
    if [ $? -eq 0 ]; then
        echo "TPU VM creation completed successfully"
        sleep 100
    else
        echo "TPU creation failed or TPU preempted, retrying..."
        sleep 20
    fi
done