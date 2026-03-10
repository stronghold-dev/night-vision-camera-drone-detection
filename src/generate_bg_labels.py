#!/usr/bin/env python3
"""
Generate empty YOLO label files for background-only images.
Empty labels tell YOLO there are no objects in the image.

Usage:
    python src/generate_background_labels.py
    python src/generate_background_labels.py --dataset data/20260216T174203-bg
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=str(REPO_ROOT / "data" / "20260216T174203-bg")
    )
    args = parser.parse_args()

    img_dir = Path(args.dataset) / "images"
    lbl_dir = Path(args.dataset) / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"images/ not found: {img_dir}")

    lbl_dir.mkdir(exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files found in {img_dir}")

    for img in images:
        (lbl_dir / img.with_suffix(".txt").name).write_text("")

    print(f"  Generated {len(images):,} empty label files → {lbl_dir}")


if __name__ == "__main__":
    main()
