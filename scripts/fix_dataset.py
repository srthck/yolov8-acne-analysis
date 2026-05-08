"""
fix_dataset.py — Fix coordinate overflow and create train/val split.

Issues:
1. Roboflow sometimes exports coords > 1.0 (e.g., 1.0083) — clamp to [0, 1]
2. All data in train/ — need 80/20 split to val/

Usage:
    python scripts/fix_dataset.py
"""

import os
import shutil
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_IMGS = DATASET_ROOT / "images" / "train"
TRAIN_LABELS = DATASET_ROOT / "labels" / "train"
VAL_IMGS = DATASET_ROOT / "images" / "val"
VAL_LABELS = DATASET_ROOT / "labels" / "val"

def clamp_coords(label_path):
    """Clamp all polygon coordinates to [0, 1]."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    changed = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:  # class_id only
            fixed_lines.append(line)
            continue
        
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        # Clamp to [0, 1]
        clamped = [max(0.0, min(1.0, c)) for c in coords]
        
        if coords != clamped:
            changed = True
        
        fixed_line = f"{class_id} " + " ".join(f"{c:.6f}" for c in clamped) + "\n"
        fixed_lines.append(fixed_line)
    
    if changed:
        with open(label_path, 'w') as f:
            f.writelines(fixed_lines)
        return True
    return False

def main():
    print("\n── 1. Fix coordinate overflow ──────────────────────────────────────")
    
    label_files = list(TRAIN_LABELS.glob("*.txt"))
    fixed_count = 0
    
    for lbl_path in label_files:
        if clamp_coords(lbl_path):
            fixed_count += 1
            print(f"  Fixed: {lbl_path.name}")
    
    print(f"  Total fixed: {fixed_count}/{len(label_files)}")
    
    # ────────────────────────────────────────────────────────────────────────
    print("\n── 2. Create 80/20 train/val split ──────────────────────────────────")
    
    img_files = sorted([f for f in TRAIN_IMGS.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    total = len(img_files)
    
    if total == 0:
        print("  ERROR: No images found in train/")
        return
    
    # 80/20 split
    val_count = max(1, int(total * 0.2))
    random.seed(42)  # Reproducible split
    val_indices = set(random.sample(range(total), val_count))
    
    print(f"  Total images: {total}")
    print(f"  Train: {total - val_count} | Val: {val_count}")
    
    moved_count = 0
    for idx, img_file in enumerate(img_files):
        if idx in val_indices:
            # Move image
            dst_img = VAL_IMGS / img_file.name
            shutil.move(str(img_file), str(dst_img))
            
            # Move label
            lbl_src = TRAIN_LABELS / f"{img_file.stem}.txt"
            lbl_dst = VAL_LABELS / f"{img_file.stem}.txt"
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(lbl_dst))
            
            moved_count += 1
    
    print(f"  Moved: {moved_count} image/label pairs to val/")
    print("\n✅ Dataset fixed! Run: python scripts/validate_dataset.py")

if __name__ == "__main__":
    main()
