#!/usr/bin/env python3
"""Flatten drone-9k/train|valid|test/images/ into drone-9k/images/"""

from pathlib import Path
import shutil

DRONE9K = Path("data/drone-9k")
SPLITS  = ["train", "valid", "test"]
OUT_IMG = DRONE9K / "images"
OUT_LBL = DRONE9K / "labels"

OUT_IMG.mkdir(exist_ok=True)
OUT_LBL.mkdir(exist_ok=True)

for split in SPLITS:
    for src in (DRONE9K / split / "images").rglob("*"):
        if src.is_file():
            shutil.copy2(src, OUT_IMG / src.name)
    for src in (DRONE9K / split / "labels").rglob("*"):
        if src.is_file():
            shutil.copy2(src, OUT_LBL / src.name)

print("Done.")