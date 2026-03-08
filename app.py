import os
import pytorch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import streamlit as st

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Off-Road Semantic Scene Segmentation",
    page_icon="🚙",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
.sub-text {
    font-size: 1.1rem;
    color: #d1d5db;
    margin-bottom: 1rem;
}
.section-card {
    background: rgba(255,255,255,0.03);
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.small-note {
    color: #bfc7d5;
    font-size: 0.95rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    padding: 0.8rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PRED_DIR = BASE_DIR / "predictions"
COMPARE_DIR = PRED_DIR / "comparisons"
COLOR_MASK_DIR = PRED_DIR / "masks_color"
TRAIN_STATS_DIR = BASE_DIR / "train_stats"
METRICS_FILE = PRED_DIR / "evaluation_metrics.txt"
PER_CLASS_METRICS = PRED_DIR / "per_class_metrics.png"

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def get_image_files(folder: Path):
    if not folder.exists():
        return []
    exts = [".png", ".jpg", ".jpeg"]
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])

def read_text_file(file_path: Path):
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception:
            return "Could not read file."
    return "File not found."

comparison_files = get_image_files(COMPARE_DIR)
mask_files = get_image_files(COLOR_MASK_DIR)
metrics_text = read_text_file(METRICS_FILE)

# ---------------------------------------------------
# Model Loader
# ---------------------------------------------------
def run_segmentation(image, model):
    img = np.array(image)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img_tensor = torch.tensor(img).float()

    with torch.no_grad():
        output = model(img_tensor)

    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return mask

MODEL_PATH = BASE_DIR / "segmentation_head.pth"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None
    

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🚀 Project Overview")
st.sidebar.markdown("""
### Problem
Autonomous vehicles in off-road environments must understand terrain like rocks, bushes, sky, and landscape for safe navigation.

### Solution
We developed an AI-based semantic segmentation system trained on synthetic desert data generated from Duality AI’s Falcon digital twin platform.

### Approach
- Synthetic RGB + segmentation mask dataset  
- DINOv2 backbone for feature extraction  
- Segmentation head for pixel-level classification  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Dataset Summary")
st.sidebar.markdown("""
- **Training:** 2857 images  
- **Validation:** 317 images  
- **Test:** 1002 images  
- **Classes:** 10  
- **Source:** Falcon Digital Twin Platform  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Current Progress")
st.sidebar.markdown("""
- Dataset integration ✅  
- Model training ✅  
- Evaluation completed ✅  
- Prediction outputs generated ✅  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Terrain Classes")
st.sidebar.markdown("""
- Trees  
- Lush Bushes  
- Dry Grass  
- Dry Bushes  
- Ground Clutter  
- Flowers  
- Logs  
- Rocks  
- Landscape  
- Sky  
""")

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown('<div class="main-title">🚙 Live AI Terrain Segmentation Prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">A transformer-based semantic segmentation system for off-road terrain understanding using synthetic digital twin data.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Hero cards
# ---------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><h4>Mean IoU</h4><h2>0.2149</h2></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><h4>Validation Accuracy</h4><h2>69.3%</h2></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><h4>Best Validation IoU</h4><h2>0.276</h2></div>', unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🖼 Demo Results",
    "📊 Metrics",
    "⚙️ Architecture",
    "🌍 Impact & Future"
])

# ---------------------------------------------------
# TAB 1 - Overview
# ---------------------------------------------------
with tab1:
    left, right = st.columns([1, 1])

    with left:
     st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Demo Image")

    uploaded_file = st.file_uploader(
        "Upload an off-road image",
        type=["png", "jpg", "jpeg"],
        key="uploader"
    )

    if uploaded_file is not None:

        uploaded_img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)

        st.success("Image uploaded successfully.")

        if st.button("🧠 Run Segmentation", use_container_width=True):

            model = load_model()

            if model is None:
                st.error("Model file not found.")
            else:
                mask = run_segmentation(uploaded_img, model)

                st.subheader("Segmentation Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(uploaded_img, caption="Original Image")

                with col2:
                    st.image(mask, caption="Predicted Mask")

    else:
        st.markdown(
            '<p class="small-note">Upload section is provided for prototype interaction. Current version demonstrates trained outputs and evaluation results.</p>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🧠 Prototype Description")
        st.markdown("""
**What this prototype does**
- Accepts off-road terrain imagery  
- Uses a trained segmentation model pipeline  
- Generates terrain masks  
- Helps support obstacle awareness and safe navigation  

**Why it matters**
- Reduces dependency on expensive real-world data  
- Uses synthetic digital twin environments  
- Supports autonomous systems in complex terrain  
- Provides pixel-level understanding of desert scenes  
""")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("✨ Key Features")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("""
- Pixel-level terrain segmentation  
- Synthetic dataset training using digital twin simulation  
- Transformer-based DINOv2 backbone  
- Segmentation head for terrain classification  
""")
    with fc2:
        st.markdown("""
- Evaluation using IoU and accuracy metrics  
- Visualization of prediction masks  
- Comparison outputs for demo and analysis  
- Prototype-ready UI for judges and presentation  
""")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# TAB 2 - Demo Results
# ---------------------------------------------------
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🖼 Sample Prediction Results")

    if comparison_files:
        cols = st.columns(min(3, len(comparison_files)))
        for i, img_path in enumerate(comparison_files[:3]):
            with cols[i]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.warning("No comparison images found in predictions/comparisons.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎯 Colored Prediction Masks")

    if mask_files:
        selected_mask_name = st.selectbox(
            "Select a prediction mask to preview",
            [f.name for f in mask_files]
        )
        selected_mask_path = COLOR_MASK_DIR / selected_mask_name
        st.image(str(selected_mask_path), caption=selected_mask_name, use_container_width=True)
    else:
        st.warning("No colored prediction masks found in predictions/masks_color.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# TAB 3 - Metrics
# ---------------------------------------------------
with tab3:
    st.subheader("📊 Metrics")

    metrics_text = read_text_file(METRICS_FILE)

    mc1, mc2 = st.columns([1, 1])

    with mc1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📄 Evaluation Metrics")

        if METRICS_FILE.exists():
            st.code(metrics_text)
        else:
            st.warning("Evaluation metrics file not found.")
            st.caption(f"Expected path: {METRICS_FILE}")

        st.markdown('</div>', unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📈 Per-Class Metrics")

        if PER_CLASS_METRICS.exists():
            st.image(
                str(PER_CLASS_METRICS),
                caption="Per-Class Performance",
                use_container_width=True
            )
        else:
            st.warning("Per-class metrics chart not found.")
            st.caption(f"Expected path: {PER_CLASS_METRICS}")

        st.markdown('</div>', unsafe_allow_html=True)
# ---------------------------------------------------
# TAB 4 - Architecture
# ---------------------------------------------------
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ System Architecture")
    st.markdown("""
```text
Input Off-Road Image
        ↓
Image Preprocessing
        ↓
DINOv2 Backbone (Feature Extraction)
        ↓
Segmentation Head
        ↓
Pixel-wise Terrain Classification
        ↓
Predicted Segmentation Mask
                
""")

with tab5:
    st.subheader("🌍 Impact & Future")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Real-World Applications")
        st.markdown("""
- Autonomous off-road vehicles  
- Agricultural robots  
- Military terrain navigation  
- Disaster rescue robotics  
- Environmental monitoring systems  
- Terrain understanding for autonomous navigation  
""")

        st.markdown("### ✅ Prototype Status")
        st.markdown("""
This project currently functions as a **working machine learning prototype**.

It includes:
- trained segmentation model  
- evaluation metrics  
- generated terrain masks  
- comparison outputs  
- UI for demonstration and result visualization  
""")

    with col2:
        st.markdown("### 🔮 Future Improvements")
        st.markdown("""
- Live inference on uploaded images  
- FastAPI / Flask backend deployment  
- Real-time segmentation for vehicle camera feeds  
- Better model performance with GPU training  
- Data augmentation for stronger generalization  
- Hybrid training with synthetic + real-world data  
""")

        st.markdown("### 📌 Project Impact")
        st.markdown("""
This project demonstrates how **synthetic digital twin data** can be used to build computer vision systems for autonomous terrain understanding.

It reduces the need for expensive real-world labeling while providing a scalable way to train segmentation models for off-road scenarios.

""")
