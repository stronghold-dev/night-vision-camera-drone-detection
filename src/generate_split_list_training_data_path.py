#!/usr/bin/env python3
"""
Generate train.txt, val.txt, test.txt and data.yaml
for a combined drone-9k + ir-meridian dataset.

train  → all drone-9k images (train + valid + test splits flattened)
val    → all ir-meridian images
test   → all ir-meridian images (same as val)
"""

import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(".")

DRONE9K_ROOT = BASE_DIR / "data" / "drone-9k"
DRONE9K_SPLITS = ["train", "valid", "test"]

IR_ROOT = BASE_DIR / "data" / "ir-meridian-202602-20260217T121654-drone" / "images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

OUT_TRAIN = BASE_DIR / "train.txt"
OUT_VAL   = BASE_DIR / "val.txt"
OUT_TEST  = BASE_DIR / "test.txt"
OUT_YAML  = BASE_DIR / "data.yaml"
# ─────────────────────────────────────────────────────────────────────────────


def collect_images(directory: Path) -> list[str]:
    """Return sorted list of relative image paths under directory."""
    paths = []
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append("./" + f.as_posix())
    return paths


def write_txt(path: Path, entries: list[str]) -> None:
    path.write_text("\n".join(entries) + "\n")
    print(f"  Written: {path}  ({len(entries)} images)")


def main():
    # ── Train: all drone-9k splits ────────────────────────────────────────────
    train_paths = []
    for split in DRONE9K_SPLITS:
        img_dir = DRONE9K_ROOT / split / "images"
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue
        imgs = collect_images(img_dir)
        print(f"  drone-9k/{split}/images: {len(imgs)} images")
        train_paths.extend(imgs)

    # ── Val / Test: all ir-meridian images ────────────────────────────────────
    if not IR_ROOT.exists():
        raise FileNotFoundError(f"IR dataset not found: {IR_ROOT}")
    ir_paths = collect_images(IR_ROOT)
    print(f"  ir-meridian/images: {len(ir_paths)} images")

    # ── Write files ───────────────────────────────────────────────────────────
    write_txt(OUT_TRAIN, train_paths)
    write_txt(OUT_VAL,   ir_paths)
    write_txt(OUT_TEST,  ir_paths)

    # ── data.yaml ─────────────────────────────────────────────────────────────
    yaml_content = (
        "names:\n"
        "  0: drone\n"
        f"path: {BASE_DIR.resolve().as_posix()}\n"
        f"train: {OUT_TRAIN.name}\n"
        f"val:   {OUT_VAL.name}\n"
        f"test:  {OUT_TEST.name}\n"
    )
    OUT_YAML.write_text(yaml_content)
    print(f"  Written: {OUT_YAML}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSummary")
    print(f"  train : {len(train_paths):,} images  → {OUT_TRAIN}")
    print(f"  val   : {len(ir_paths):,} images  → {OUT_VAL}")
    print(f"  test  : {len(ir_paths):,} images  → {OUT_TEST}")


if __name__ == "__main__":
    main()
