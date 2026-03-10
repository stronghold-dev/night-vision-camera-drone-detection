#!/bin/bash

OMP_NUM_THREADS=$(nproc) \
MKL_NUM_THREADS=$(nproc) \
yolo task=detect mode=train \
model=yolo11n.pt \
data=/home/jonathan.platkiewicz/night-vision-camera-drone-detection/configs/data.yaml \
epochs=200 \
imgsz=640 \
device=0 \
batch=32 \
workers=$(nproc) \
project=experiments \
name=yolo11n-640-night \
cache=disk \
save=True \
save_period=10 \
patience=50