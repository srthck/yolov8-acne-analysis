# Acne Segmentation and Severity Analysis System

A Python pipeline for detecting and segmenting acne lesions in face images using YOLOv8 instance segmentation. Built with MediaPipe for face detection, Ultralytics YOLOv8s-seg for lesion segmentation, and Streamlit for a simple inference UI.

---

## Features

- Face detection and cropping with 30% padding (MediaPipe)
- Instance segmentation of individual acne lesions (YOLOv8s-seg)
- Rule-based severity classification (Mild / Moderate / Severe)
- Streamlit UI with annotated result image and downloadable output
- Fully local — no cloud, no database, no authentication

---

## Project Structure

```
skin_ai_project/
├── dataset/
│   ├── raw/              ← Original uploaded face images
│   ├── cropped_raw/      ← Face-cropped outputs (input to annotation)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/               ← Place trained best.pt here
├── outputs/              ← Inference results saved here
├── scripts/
│   └── crop_faces.py     ← Step 1: preprocessing
├── app.py                ← Streamlit UI
├── train.py              ← YOLOv8 training script
├── predict.py            ← Inference pipeline
├── data.yaml             ← Dataset config for YOLO
└── requirements.txt
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Install PyTorch with CUDA support first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the pip install above.

---

## Workflow

### Step 1 — Preprocess raw images

Place your raw face photos in `dataset/raw/`, then run:

```bash
python scripts/crop_faces.py
```

Cropped face images are saved to `dataset/cropped_raw/`.

---

### Step 2 — Annotate with Roboflow

1. Go to [roboflow.com](https://roboflow.com) → New Project → **Instance Segmentation**
2. Upload all images from `dataset/cropped_raw/`
3. Draw **polygon masks** around each individual acne lesion (class name: `acne`)
4. Export as **YOLOv8 format** with an 80/20 train/val split
5. Place exported files into:
   - `dataset/images/train/` and `dataset/images/val/`
   - `dataset/labels/train/` and `dataset/labels/val/`

---

### Step 3 — Train

```bash
python train.py
```

Training configuration (editable inside `train.py`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs`  | 50      | Increase for larger datasets |
| `imgsz`   | 640     | Fixed — matches annotation resolution |
| `batch`   | 16      | Reduce to 8 if VRAM is limited |
| `patience`| 10      | Early stopping rounds |

Trained weights are saved to `outputs/acne_segmentation/weights/best.pt`.

Copy `best.pt` to the `models/` directory:

```bash
copy outputs\acne_segmentation\weights\best.pt models\best.pt
```

---

### Step 4 — Run inference (CLI)

```bash
python predict.py path/to/image.jpg models/best.pt
```

Annotated result is saved to `outputs/result_<filename>`.

---

### Step 5 — Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload a front-facing photo and view the segmentation result, lesion count, severity rating, and recommendation.

---

## Severity Thresholds

| Lesion Count | Severity |
|---|---|
| 0 – 5   | Mild     |
| 6 – 15  | Moderate |
| 16+     | Severe   |

Thresholds are defined in `predict.py → get_severity()` and can be adjusted freely.

---

## Limitations

- Accuracy depends entirely on the quality and size of your annotated dataset
- Low lighting, heavy makeup, or extreme angles will reduce detection quality
- The model segments visually visible lesions only — subclinical lesions are not detectable
- Severity classification is rule-based and not clinically validated
- Performance on skin tones not represented in training data may be reduced

---

## Disclaimer

This system is built for **educational and research purposes only**. It does not provide medical advice, a clinical diagnosis, or a treatment recommendation. Always consult a licensed dermatologist for any skin health concerns.

---

## Tech Stack

| Component | Library |
|---|---|
| Face detection | MediaPipe 0.10+ |
| Segmentation model | Ultralytics YOLOv8s-seg |
| Deep learning backend | PyTorch 2.0+ |
| Image processing | OpenCV |
| UI | Streamlit |

## Demo

![Demo](docs/screenshots/demo.png)