"""
evaluate_model.py — Evaluate segmentation model on validation set.

Run inference on all validation images and visualize results.

Usage:
    python scripts/evaluate_model.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_IMGS = PROJECT_ROOT / "dataset" / "images" / "val"
MODEL_PATH = PROJECT_ROOT / "runs" / "segment" / "outputs" / "acne_segmentation" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

def evaluate():
    """Run inference on all validation images and save annotated results."""
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return
    
    if not VAL_IMGS.exists():
        print(f"[ERROR] Validation images not found: {VAL_IMGS}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # Get all validation images
    val_files = sorted([f for f in VAL_IMGS.glob("*.JPG") if f.suffix.upper() in {".JPG", ".JPEG", ".PNG"}])
    
    if not val_files:
        print(f"[ERROR] No validation images found in {VAL_IMGS}")
        return
    
    print(f"\nRunning inference on {len(val_files)} validation images...\n")
    
    total_lesions = 0
    for img_file in val_files:
        print(f"Processing: {img_file.name}")
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  ❌ Could not read image")
            continue
        
        # Run inference directly on cropped image (no face detection needed)
        results = model.predict(
            source=image,
            conf=0.25,
            imgsz=416,
            verbose=False,
        )
        
        result = results[0]
        lesion_count = 0
        annotated = image.copy()
        
        # Draw segmentation masks
        if result.masks is not None:
            masks_xy = result.masks.xy
            lesion_count = len(masks_xy)
            total_lesions += lesion_count
            
            overlay = annotated.copy()
            for poly in masks_xy:
                pts = poly.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color=(0, 0, 220))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 180), thickness=2)
            
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
        
        # Add lesion count text
        cv2.putText(
            annotated,
            f"Lesions: {lesion_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Save result
        output_file = OUTPUT_DIR / f"eval_{img_file.stem}.jpg"
        cv2.imwrite(str(output_file), annotated)
        print(f"  ✅ Detected {lesion_count} lesions → {output_file.name}")
    
    print(f"\n{'='*70}")
    print(f"Total lesions detected across {len(val_files)} validation images: {total_lesions}")
    print(f"Average lesions per image: {total_lesions / len(val_files):.1f}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
