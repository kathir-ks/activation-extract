#!/bin/bash
#
# Upload dataset streams to GCS
# Usage: ./upload_dataset_streams_to_gcs.sh --bucket BUCKET_NAME --prefix PREFIX [--local_dir DIR]
#

set -e

# Default values
LOCAL_DIR="./dataset_streams"
BUCKET=""
PREFIX="dataset_streams"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bucket) BUCKET="$2"; shift ;;
        --prefix) PREFIX="$2"; shift ;;
        --local_dir) LOCAL_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$BUCKET" ]; then
    echo "Error: --bucket is required"
    echo "Usage: $0 --bucket BUCKET_NAME [--prefix PREFIX] [--local_dir DIR]"
    exit 1
fi

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Local directory $LOCAL_DIR does not exist"
    exit 1
fi

echo "========================================"
echo "UPLOADING DATASET STREAMS TO GCS"
echo "========================================"
echo "Local directory: $LOCAL_DIR"
echo "GCS bucket: gs://$BUCKET/$PREFIX/"
echo ""

# Count files
NUM_FILES=$(find "$LOCAL_DIR" -name "*.jsonl" | wc -l)
echo "Found $NUM_FILES JSONL files to upload"
echo ""

# Upload with gsutil
echo "Starting upload..."
gsutil -m cp -r "$LOCAL_DIR"/*.jsonl "gs://$BUCKET/$PREFIX/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Upload complete!"
    echo ""
    echo "Files available at: gs://$BUCKET/$PREFIX/"
    echo ""
    echo "To list files:"
    echo "  gsutil ls gs://$BUCKET/$PREFIX/"
    echo ""
    echo "To download to workers:"
    echo "  gsutil cp gs://$BUCKET/$PREFIX/stream_000.jsonl ./"
else
    echo ""
    echo "✗ Upload failed!"
    exit 1
fi
