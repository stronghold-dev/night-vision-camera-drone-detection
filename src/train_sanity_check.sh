#!/bin/bash

OMP_NUM_THREADS=$(nproc) \
MKL_NUM_THREADS=$(nproc) \
yolo task=detect mode=train \
model=yolo11n.pt \
data=/home/jonathan.platkiewicz/night-vision-camera-drone-detection/data.yaml \
epochs=5 \
imgsz=640 \tre
device=0 \
batch=16 \
workers=$(nproc) \
project=experiments \
name=yolo11n-drone