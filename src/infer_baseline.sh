#!/bin/bash

set -e

# ---- Configuration ----
MODEL_DIR="models"
MODEL_WEIGHTS="$MODEL_DIR/yolo11n-640.pt"

DATASET_ROOT="/home/jonathan.platkiewicz/night-vision-camera-drone-detection/data/20260220T171702-drone"
TEST_LIST="$DATASET_ROOT/test.txt"

OUTPUT_DIR="inference_baseline"

# ---- Sanity checks ----
if [ ! -f "$MODEL_WEIGHTS" ]; then
  echo "ERROR: Model weights not found: $MODEL_WEIGHTS"
  exit 1
fi

if [ ! -f "$TEST_LIST" ]; then
  echo "ERROR: test.txt not found: $TEST_LIST"
  exit 1
fi

# ---- Run inference ----
echo ""
echo "Running YOLO inference..."

uvx --from ultralytics yolo \
task=detect \
mode=predict \
model="$MODEL_WEIGHTS" \
source="$TEST_LIST" \
imgsz=640 \
device=0 \
project="$OUTPUT_DIR" \
name="baseline"

echo ""
echo "Inference finished."
echo "Results saved in:"
echo "$OUTPUT_DIR/baseline"