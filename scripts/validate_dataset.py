"""
validate_dataset.py — Pre-training dataset sanity check.

Run BEFORE python train.py to catch common label/path problems early.

Usage:
    python scripts/validate_dataset.py
"""

import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_ROOT  = PROJECT_ROOT / "dataset"
IMAGE_SPLITS  = ["train", "val"]
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
MIN_POLY_PTS  = 3          # minimum xy pairs per polygon (6 values after class id)
ERRORS        = []
WARNINGS      = []


def err(msg: str):
    ERRORS.append(msg)
    print(f"  [ERROR]   {msg}")


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"  [WARN]    {msg}")


def info(msg: str):
    print(f"  [OK]      {msg}")


# ── 1. Check directory structure ──────────────────────────────────────────────
print("\n── 1. Directory structure ───────────────────────────────────────────────")
for split in IMAGE_SPLITS:
    img_dir   = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not img_dir.exists():
        err(f"Missing: dataset/images/{split}/")
    else:
        info(f"dataset/images/{split}/ exists")

    if not label_dir.exists():
        err(f"Missing: dataset/labels/{split}/")
    else:
        info(f"dataset/labels/{split}/ exists")


# ── 2. Check image/label counts and pairing ───────────────────────────────────
print("\n── 2. Image ↔ label pairing ─────────────────────────────────────────────")
for split in IMAGE_SPLITS:
    img_dir   = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not img_dir.exists() or not label_dir.exists():
        continue  # already flagged above

    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    labels = [f for f in label_dir.iterdir() if f.suffix == ".txt"]

    print(f"\n  [{split}]  {len(images)} images | {len(labels)} label files")

    if len(images) == 0:
        err(f"{split}: No images found in dataset/images/{split}/")
        continue

    if len(labels) == 0:
        err(f"{split}: No label files found in dataset/labels/{split}/")
        continue

    if split == "train" and len(images) < 15:
        warn(f"{split}: Only {len(images)} training images — model will overfit badly. Target 25+.")

    if split == "val" and len(images) < 5:
        warn(f"{split}: Only {len(images)} val images — metrics will be unreliable. Target 6–10.")

    # Check every image has a matching label
    img_stems = {f.stem for f in images}
    lbl_stems = {f.stem for f in labels}

    missing_labels = img_stems - lbl_stems
    orphan_labels  = lbl_stems - img_stems

    if missing_labels:
        for stem in sorted(missing_labels)[:5]:  # show first 5 only
            err(f"{split}: Image '{stem}' has no matching .txt label file")
        if len(missing_labels) > 5:
            err(f"  ... and {len(missing_labels) - 5} more unlabeled images")
    else:
        info(f"{split}: All images have matching label files")

    if orphan_labels:
        for stem in sorted(orphan_labels)[:3]:
            warn(f"{split}: Label '{stem}.txt' has no matching image — will be ignored by YOLO")


# ── 3. Validate label file contents ──────────────────────────────────────────
print("\n── 3. Label content validation ──────────────────────────────────────────")
bad_label_count = 0
total_annotations = 0

for split in IMAGE_SPLITS:
    label_dir = DATASET_ROOT / "labels" / split
    if not label_dir.exists():
        continue

    label_files = list(label_dir.glob("*.txt"))

    for lf in label_files:
        lines = lf.read_text().strip().splitlines()
        empty_file = True

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            empty_file = False
            parts = line.split()

            # Check class id
            try:
                class_id = int(parts[0])
            except (ValueError, IndexError):
                err(f"{split}/{lf.name} line {line_num}: cannot parse class id → '{line[:60]}'")
                bad_label_count += 1
                continue

            if class_id != 0:
                warn(f"{split}/{lf.name} line {line_num}: class_id={class_id} — expected 0 (acne). Wrong export format?")

            # Must be segmentation format: class x1 y1 x2 y2 ... (at least 7 values)
            coords = parts[1:]
            if len(coords) < MIN_POLY_PTS * 2:
                err(
                    f"{split}/{lf.name} line {line_num}: only {len(coords)} coord values "
                    f"({len(coords)//2} points) — needs ≥{MIN_POLY_PTS} points. "
                    "Did you export as YOLOv8 Detection instead of Segmentation?"
                )
                bad_label_count += 1
                continue

            # Odd number of coords means a broken polygon
            if len(coords) % 2 != 0:
                err(f"{split}/{lf.name} line {line_num}: odd number of coordinate values ({len(coords)}) — broken polygon")
                bad_label_count += 1
                continue

            # All coords must be in [0, 1]
            try:
                float_coords = [float(c) for c in coords]
            except ValueError:
                err(f"{split}/{lf.name} line {line_num}: non-numeric coordinate value")
                bad_label_count += 1
                continue

            out_of_bounds = [c for c in float_coords if c < 0.0 or c > 1.0]
            if out_of_bounds:
                err(
                    f"{split}/{lf.name} line {line_num}: {len(out_of_bounds)} coord(s) outside [0,1] "
                    f"— e.g. {out_of_bounds[0]:.4f}"
                )
                bad_label_count += 1
                continue

            total_annotations += 1

        if empty_file:
            # Empty label file = background image (valid, just informational)
            pass  # YOLO handles these correctly

    if bad_label_count == 0:
        info(f"{split}: All label files passed content checks")


# ── 4. data.yaml check ───────────────────────────────────────────────────────
print("\n── 4. data.yaml ─────────────────────────────────────────────────────────")
yaml_path = PROJECT_ROOT / "data.yaml"
if not yaml_path.exists():
    err("data.yaml not found in project root")
else:
    content = yaml_path.read_text()
    if "acne" not in content:
        err("data.yaml: class name 'acne' not found")
    else:
        info("data.yaml: class 'acne' present")

    if "train" not in content and "val" not in content:
        err("data.yaml: missing train/val path entries")
    else:
        info("data.yaml: train and val paths present")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  Total valid annotations found : {total_annotations}")
print(f"  Errors   : {len(ERRORS)}")
print(f"  Warnings : {len(WARNINGS)}")

if ERRORS:
    print("\n  Fix ALL errors before running train.py.")
    sys.exit(1)
elif WARNINGS:
    print("\n  Warnings present — review before training.")
    sys.exit(0)
else:
    print("\n  Dataset looks clean. Ready to train.")
    print("  Run: python train.py")
    sys.exit(0)
