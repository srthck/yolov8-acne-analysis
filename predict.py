import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── OpenCV Haar Cascade face detector ──
_face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def crop_face(image: np.ndarray, padding_ratio: float = 0.3) -> np.ndarray | None:
    """
    Detects the first face in an image and returns a padded crop.
    Returns None if no face is found or the crop is invalid.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance image contrast for better detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Try detection with relaxed parameters
    detections = _face_detector.detectMultiScale(
        enhanced_gray,
        scaleFactor=1.03,
        minNeighbors=2,
        minSize=(20, 20)
    )
    
    # Fallback to original image if no detections
    if len(detections) == 0:
        detections = _face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(25, 25)
        )
    
    if len(detections) == 0:
        return None
    
    # Get the largest detected face
    detection = max(detections, key=lambda x: x[2] * x[3])
    xmin, ymin, bw, bh = detection

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, xmin - pad_x)
    y1 = max(0, ymin - pad_y)
    x2 = min(w, xmin + bw + pad_x)
    y2 = min(h, ymin + bh + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def get_severity(count: int) -> tuple[str, str]:
    """
    Maps lesion count to a severity label and a plain-language recommendation.
    This is NOT a medical diagnosis.
    """
    if count <= 5:
        return (
            "Mild",
            "Maintain a consistent cleansing routine. Non-comedogenic moisturizer recommended.",
        )
    elif count <= 15:
        return (
            "Moderate",
            "Consider OTC salicylic acid or benzoyl peroxide products. Consult a dermatologist if persistent.",
        )
    else:
        return (
            "Severe",
            "Strongly recommended to consult a licensed dermatologist for a personalised treatment plan.",
        )


def run_inference(
    image_path: str | Path,
    model_path: str | Path,
    conf_threshold: float = 0.25,
) -> dict:
    """
    End-to-end inference on a single image.

    Returns a dict with:
        result_image  – annotated BGR numpy array (face crop with masks drawn)
        acne_count    – int, number of detected lesions
        severity      – str, one of: Mild / Moderate / Severe
        recommendation– str, plain-language advice
        error         – str or None
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": "Could not read image. Check file format."}

    # ── 1. Crop face ──────────────────────────────────────────────────────────
    face_crop = crop_face(image)
    if face_crop is None:
        return {"error": "No face detected. Please use a clear front-facing photo."}

    # ── 2. Run YOLOv8 segmentation ────────────────────────────────────────────
    model = YOLO(str(model_path))
    predictions = model.predict(
        source=face_crop,
        conf=conf_threshold,
        imgsz=640,
        verbose=False,
    )

    result = predictions[0]
    acne_count = 0
    annotated = face_crop.copy()

    # ── 3. Draw segmentation masks ────────────────────────────────────────────
    if result.masks is not None:
        masks_xy = result.masks.xy          # list of (N, 2) polygon arrays
        acne_count = len(masks_xy)

        overlay = annotated.copy()
        for poly in masks_xy:
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            # Semi-transparent filled polygon
            cv2.fillPoly(overlay, [pts], color=(0, 0, 220))
            # Solid polygon outline
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 180), thickness=1)

        # Blend overlay for transparency effect
        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    # ── 4. Burn lesion count into the image ───────────────────────────────────
    cv2.putText(
        annotated,
        f"Lesions: {acne_count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # ── 5. Severity assessment ────────────────────────────────────────────────
    severity, recommendation = get_severity(acne_count)

    return {
        "result_image": annotated,
        "acne_count": acne_count,
        "severity": severity,
        "recommendation": recommendation,
        "error": None,
    }


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_path> <model_path>")
        print("Example: python predict.py test.jpg models/best.pt")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_path = sys.argv[2]

    out = run_inference(img_path, model_path)

    if out.get("error"):
        print(f"[ERROR] {out['error']}")
        sys.exit(1)

    print(f"Acne Count  : {out['acne_count']}")
    print(f"Severity    : {out['severity']}")
    print(f"Advice      : {out['recommendation']}")

    output_path = Path("outputs") / f"result_{Path(img_path).name}"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), out["result_image"])
    print(f"Result saved: {output_path}")
