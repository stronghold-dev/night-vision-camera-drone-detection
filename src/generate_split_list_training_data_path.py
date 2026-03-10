#!/usr/bin/env python3
"""
Generate train.txt, val.txt, test.txt and data.yaml
for a combined drone-9k + ir-meridian dataset.

train  → all drone-9k images (train + valid + test splits flattened)
val    → all ir-meridian images
test   → all ir-meridian images (same as val)
"""

import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg"}

REPO_ROOT   = Path(__file__).resolve().parent.parent   # src/ → repo root
DEFAULT_OUT = REPO_ROOT
# ─────────────────────────────────────────────────────────────────────────────


def collect_images(dataset_dir: Path) -> list[str]:
    img_dir = dataset_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"images/ subdirectory not found in dataset: {dataset_dir}"
        )
    paths = []
    for f in sorted(img_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(f.resolve()))  # absolute path, YOLO uses as-is
    return paths


def resolve_datasets(folders) -> list[Path]:
    if not folders:
        return []
    resolved = []
    for f in folders:
        p = "data" / Path(f)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset folder not found: {p}")
        resolved.append(p)
    return resolved


def collect_split(datasets: list[Path], split_name: str) -> list[str]:
    all_paths = []
    for ds in datasets:
        imgs = collect_images(ds)
        print(f"  [{split_name}] {ds.name}/images : {len(imgs)} images")
        all_paths.extend(imgs)
    return all_paths


def write_txt(path: Path, entries: list[str]) -> None:
    path.write_text("\n".join(entries) + "\n")
    written = sum(1 for line in path.read_text().splitlines() if line.strip())
    if written != len(entries):
        raise RuntimeError(
            f"Count mismatch in {path}: expected {len(entries)}, got {written}"
        )
    print(f"  Written : {path}  ({written} images ✓)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO split .txt files and data.yaml."
    )
    parser.add_argument("-train", nargs="+", metavar="DIR")
    parser.add_argument("-val",   nargs="+", metavar="DIR")
    parser.add_argument("-test",  nargs="+", metavar="DIR")
    parser.add_argument("-out", default=str(DEFAULT_OUT), metavar="DIR")
    args = parser.parse_args()

    if not any([args.train, args.val, args.test]):
        parser.error("Provide at least one of -train, -val, -test.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = resolve_datasets(args.train)
    val_ds   = resolve_datasets(args.val)
    test_ds  = resolve_datasets(args.test)

    train_paths = collect_split(train_ds, "train")
    val_paths   = collect_split(val_ds,   "val")
    test_paths  = collect_split(test_ds,  "test")

    out_train = out_dir / "experiments" / "train.txt"
    out_val   = out_dir / "experiments" / "val.txt"
    out_test  = out_dir / "experiments" / "test.txt"

    if train_paths: write_txt(out_train, train_paths)
    if val_paths:   write_txt(out_val,   val_paths)
    if test_paths:  write_txt(out_test,  test_paths)

    yaml_lines = ["names:", "  0: drone", f"path: /"]
    if train_paths: yaml_lines.append(f"train: {out_train.resolve().as_posix()}")
    if val_paths:   yaml_lines.append(f"val:   {out_val.resolve().as_posix()}")
    if test_paths:  yaml_lines.append(f"test:  {out_test.resolve().as_posix()}")

    out_yaml = out_dir / "configs" / "data.yaml"
    out_yaml.write_text("\n".join(yaml_lines) + "\n")
    print(f"  Written : {out_yaml}")

    total = len(train_paths) + len(val_paths) + len(test_paths)
    def pct(n): return f"{n/total*100:.1f}%" if total else "N/A"

    print("\nSummary")
    print(f"  train : {len(train_paths):>6,} images  ({pct(len(train_paths))})")
    print(f"  val   : {len(val_paths):>6,} images  ({pct(len(val_paths))})")
    print(f"  test  : {len(test_paths):>6,} images  ({pct(len(test_paths))})")
    print(f"  total : {total:>6,} images")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
