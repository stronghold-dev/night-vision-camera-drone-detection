"""
YOLO-World inference on downsampled frames → COCO 1.0 JSON per video
Compatible with CVAT's "COCO 1.0" annotation import format.

Usage:
    python yolo_world_infer.py --frames_root frames/ --output_root annotations/
"""

import argparse
import json
import os
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE   = 0.05   # Low threshold: catch more drones, delete FP in CVAT
PROMPT       = ["drone", "UAV", "quadcopter", "flying object"]  # Try also ["drone", "UAV", "quadcopter"] if recall is low
MODEL_NAME   = "yolov8x-world.pt"  # options: yolov8s/m/l/x-world.pt
# ─────────────────────────────────────────────────────────────────────────────


def build_coco_json(img_dir: Path, results, label: str = "drone") -> dict:
    categories = [{"id": 1, "name": label, "supercategory": "object"}]
    images, annotations = [], []
    ann_id = 1

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    for img_id, (img_path, result) in enumerate(zip(img_paths, results), start=1):
        w = result.image_width
        h = result.image_height
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        for pred in result.object_prediction_list:
            bbox = pred.bbox  # sahi BoundingBox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            bw, bh = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
                "score": round(pred.score.value, 4),
            })
            ann_id += 1

    return {"categories": categories, "images": images, "annotations": annotations}


def run(frames_root: str, output_root: str):
    frames_root = Path(frames_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    _yolo = YOLO(MODEL_NAME)
    _yolo.set_classes(PROMPT)
    _yolo.save("yolo_world_custom.pt")
    _model_path = "yolo_world_custom.pt"  # use saved model with classes baked in

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=_model_path,
        confidence_threshold=CONFIDENCE,
        device="cuda:0",
    )
    print(f"Model: {MODEL_NAME} | Prompt: {PROMPT} | Conf: {CONFIDENCE}")

    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    if video_dirs:
        sources = video_dirs
    else:
        sources = [frames_root]

    for vid_dir in sources:
        img_paths = sorted(vid_dir.glob("*.jpg")) + sorted(vid_dir.glob("*.png"))
        if not img_paths:
            print(f"  Skipping {vid_dir.name} — no images found")
            continue

        print(f"\nProcessing {vid_dir.name} ({len(img_paths)} frames)...")
        results = []
        for img_path in img_paths:
            r = get_sliced_prediction(
                str(img_path),
                model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.4,
                overlap_width_ratio=0.4,
                postprocess_match_threshold=0.3,
            )
            results.append(r)

        coco = build_coco_json(vid_dir, results)
        out_path = output_root / f"coco_{vid_dir.name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        n_ann = len(coco["annotations"])
        n_img = len(coco["images"])
        print(f"  → {n_ann} boxes across {n_img} frames saved to {out_path}")
        print(f"  → avg {n_ann/n_img:.1f} boxes/frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", default="frames/")
    parser.add_argument("--output_root", default="annotations/")
    args = parser.parse_args()
    run(args.frames_root, args.output_root)
