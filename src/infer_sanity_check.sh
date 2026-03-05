#!/bin/bash

set -e

# ---- Configuration ----
MODEL_DIR="runs/detect/experiments/yolo11n-drone"
MODEL_WEIGHTS="$MODEL_DIR/weights/best.pt"

DATASET_ROOT="/home/jonathan.platkiewicz/night-vision-camera-drone-detection"
TRAIN_LIST="$DATASET_ROOT/train.txt"

OUTPUT_DIR="inference_sanity"
SAMPLE_LIST="sample.txt"
NUM_SAMPLES=20

# ---- Sanity checks ----
if [ ! -f "$MODEL_WEIGHTS" ]; then
  echo "ERROR: Model weights not found: $MODEL_WEIGHTS"
  exit 1
fi

if [ ! -f "$TRAIN_LIST" ]; then
  echo "ERROR: train.txt not found: $TRAIN_LIST"
  exit 1
fi

# ---- Create small sample ----
echo "Creating sample of $NUM_SAMPLES images from training set..."
head -n "$NUM_SAMPLES" "$TRAIN_LIST" > "$SAMPLE_LIST"

echo "Sample list created:"
echo "$SAMPLE_LIST"

# ---- Run inference ----
echo ""
echo "Running YOLO inference sanity check..."

uvx --from ultralytics yolo \
task=detect \
mode=predict \
model="$MODEL_WEIGHTS" \
source="$SAMPLE_LIST" \
imgsz=640 \
device=0 \
project="$OUTPUT_DIR" \
name="sanity_check"

echo ""
echo "Inference finished."
echo "Results saved in:"
echo "$OUTPUT_DIR/sanity_check"