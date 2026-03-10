#!/usr/bin/env python3
"""
Convert drone-4k from COCO JSON format to YOLO format and flatten.

Input structure:
    drone-4k/
        train/_annotations.coco.json + *.jpg
        valid/_annotations.coco.json + *.jpg
        test/ _annotations.coco.json + *.jpg

Output structure:
    drone-4k/
        images/   ← all images from all splits
        labels/   ← one .txt per image in YOLO format

Usage:
    python src/convert_coco_to_yolo.py
    python src/convert_coco_to_yolo.py --dataset data/drone-4k
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS    = ["train", "valid", "test"]
ANNO_FILE = "_annotations.coco.json"


def coco_to_yolo(x_min, y_min, w, h, img_w, img_h) -> tuple:
    x_center = (x_min + w / 2) / img_w
    y_center  = (y_min + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def convert_split(split_dir: Path, out_images: Path, out_labels: Path) -> dict:
    anno_path = split_dir / ANNO_FILE
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")

    with anno_path.open() as f:
        coco = json.load(f)

    # Build lookup tables
    images    = {img["id"]: img for img in coco["images"]}
    anno_map  = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        anno_map[ann["image_id"]].append(ann)

    counts = {"images": 0, "labels": 0, "boxes": 0, "skipped": 0}

    for img_id, img_info in images.items():
        fname    = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]
        src_img  = split_dir / fname

        if not src_img.exists():
            print(f"  WARNING: image not found, skipping: {src_img}")
            counts["skipped"] += 1
            continue

        # Copy image
        shutil.copy2(src_img, out_images / fname)
        counts["images"] += 1

        # Write label file (empty if no annotations)
        label_path = out_labels / (Path(fname).stem + ".txt")
        annotations = anno_map.get(img_id, [])
        lines = []
        for ann in annotations:
            x, y, w, h = coco_to_yolo(*ann["bbox"], img_w, img_h)
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts["boxes"] += 1
        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        counts["labels"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Convert drone-4k COCO → YOLO.")
    parser.add_argument(
        "--dataset", default=str(REPO_ROOT / "data" / "drone-4k"),
        help="Path to drone-4k root directory."
    )
    args  = parser.parse_args()
    root  = Path(args.dataset)

    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    out_images = root / "images"
    out_labels = root / "labels"
    out_images.mkdir(exist_ok=True)
    out_labels.mkdir(exist_ok=True)

    totals = {"images": 0, "labels": 0, "boxes": 0, "skipped": 0}

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  WARNING: split not found, skipping: {split_dir}")
            continue
        counts = convert_split(split_dir, out_images, out_labels)
        print(f"  [{split}] images: {counts['images']}  "
              f"labels: {counts['labels']}  boxes: {counts['boxes']}  "
              f"skipped: {counts['skipped']}")
        for k in totals:
            totals[k] += counts[k]

    print(f"\nSummary")
    print(f"  images  : {totals['images']:,}")
    print(f"  labels  : {totals['labels']:,}")
    print(f"  boxes   : {totals['boxes']:,}")
    print(f"  skipped : {totals['skipped']:,}")
    print(f"  output  : {root}")


if __name__ == "__main__":
    main()
