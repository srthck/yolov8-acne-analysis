import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from predict import run_inference

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Acne Severity Analyzer",
    page_icon="🔬",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = Path("models") / "best.pt"

SEVERITY_COLORS = {
    "Mild":     "#2ecc71",   # green
    "Moderate": "#f39c12",   # amber
    "Severe":   "#e74c3c",   # red
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_path_input = st.text_input(
        "Model path",
        value=str(DEFAULT_MODEL),
        help="Path to your trained YOLOv8 segmentation weights (.pt file).",
    )
    conf_threshold = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower = more detections (may include false positives).",
    )
    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: This tool is for educational and informational "
        "purposes only. It does not constitute medical advice or a clinical "
        "diagnosis. Consult a licensed dermatologist for professional evaluation."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Acne Severity Analyzer")
st.markdown(
    "Upload a clear, front-facing photo. The system will detect and segment "
    "acne lesions, count them, and estimate severity using rule-based thresholds."
)
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# ── Save upload to a temp file ────────────────────────────────────────────────
tmp_path = Path("outputs") / f"_upload_{uploaded_file.name}"
tmp_path.parent.mkdir(exist_ok=True)
tmp_path.write_bytes(uploaded_file.read())

# ── Validate model ────────────────────────────────────────────────────────────
model_path = Path(model_path_input)
if not model_path.exists():
    st.error(
        f"Model not found at `{model_path}`. "
        "Train the model first with `python train.py`, or update the path in the sidebar."
    )
    st.stop()

# ── Run inference ─────────────────────────────────────────────────────────────
with st.spinner("Detecting face and analysing lesions…"):
    output = run_inference(
        image_path=tmp_path,
        model_path=model_path,
        conf_threshold=conf_threshold,
    )

# ── Handle errors ─────────────────────────────────────────────────────────────
if output.get("error"):
    st.error(f"❌ {output['error']}")
    st.stop()

# ── Results layout ────────────────────────────────────────────────────────────
result_bgr  = output["result_image"]
acne_count  = output["acne_count"]
severity    = output["severity"]
advice      = output["recommendation"]

# Convert BGR → RGB for Streamlit display
result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

col_img, col_metrics = st.columns([3, 2], gap="large")

with col_img:
    st.subheader("Segmentation Result")
    st.image(result_rgb, use_container_width=True, caption="Red overlays = detected lesions")

with col_metrics:
    st.subheader("Analysis")

    # Lesion count
    st.metric(label="Lesions detected", value=acne_count)

    # Severity badge via coloured markdown
    badge_color = SEVERITY_COLORS.get(severity, "#95a5a6")
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{badge_color};
            color:#fff;
            padding:6px 18px;
            border-radius:20px;
            font-weight:700;
            font-size:1.1rem;
            margin:8px 0 16px 0;
        ">{severity}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Recommendation**")
    st.info(advice)

st.divider()

# ── Save annotated result ─────────────────────────────────────────────────────
result_save_path = Path("outputs") / f"result_{uploaded_file.name}"
cv2.imwrite(str(result_save_path), result_bgr)

st.download_button(
    label="⬇️ Download annotated image",
    data=cv2.imencode(".jpg", result_bgr)[1].tobytes(),
    file_name=f"acne_result_{uploaded_file.name}",
    mime="image/jpeg",
)

# Clean up temp upload
tmp_path.unlink(missing_ok=True)
