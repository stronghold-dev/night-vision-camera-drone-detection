"""
Checks for missing/empty label files for each split defined in data.yaml.
Targets the root cause of:
  WARNING ⚠️ no labels found in detect set, cannot compute metrics without labels

Usage: python check_labels.py --data path/to/data.yaml
"""

import argparse
from pathlib import Path
import yaml


def label_path_from_image(img_path: Path) -> Path:
    parts = img_path.parts
    try:
        idx = parts.index("images")
        new_parts = parts[:idx] + ("labels",) + parts[idx + 1:]
        return Path(*new_parts).with_suffix(".txt")
    except ValueError:
        return Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")


def check_split(split_name: str, txt_path: Path):
    print(f"\n[{split_name.upper()}] {txt_path}")

    if not txt_path.exists():
        print(f"  [ERROR] Split file not found.")
        return

    img_paths = [Path(l.strip()) for l in txt_path.read_text().splitlines() if l.strip()]
    print(f"  Images listed: {len(img_paths)}")

    missing_labels = []
    empty_labels   = []

    for img_path in img_paths:
        lbl_path = label_path_from_image(img_path)
        if not lbl_path.exists():
            missing_labels.append((img_path, lbl_path))
        elif lbl_path.stat().st_size == 0:
            empty_labels.append(img_path)

    status = "[OK]" if not missing_labels else "[ERROR]"
    print(f"  {status} Missing label files : {len(missing_labels)} / {len(img_paths)}")
    for img, lbl in missing_labels[:10]:
        print(f"       image : {img}")
        print(f"       label : {lbl}  ← not found")
    if len(missing_labels) > 10:
        print(f"       ... and {len(missing_labels) - 10} more")

    status = "[OK]" if not empty_labels else "[WARN]"
    print(f"  {status} Empty label files   : {len(empty_labels)} / {len(img_paths)}")
    for img in empty_labels[:10]:
        print(f"       {img}")
    if len(empty_labels) > 10:
        print(f"       ... and {len(empty_labels) - 10} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    args = parser.parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    for split in ("train", "val", "test"):
        if split in cfg:
            check_split(split, Path(cfg[split]))


if __name__ == "__main__":
    main()
