import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="🌝 Bulan Image Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Load Model (dengan cache)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header dan Menu Navigasi
# ==========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: white;
    }
    .subtext {
        color: black;
        font-size: 16px;
    }
    .emoji-card {
        background-color: rgba(255,255,255,0.05);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 22px;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>BULAN IMAGE DETECTION DASHBOARD 🌙</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtext'>This app performs <b>Image Detection</b> (YOLO) and <b>Image Classification</b> (CNN) for animal, fashion, food, and nature images.</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================
# Statistik Ringkas
# ==========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Available", value=0, delta="Total detections")

with col2:
    st.metric("Income", value="0.00%", delta="Session accuracy")

with col3:
    st.metric("Expense", value="0.0 ms", delta="Inference time")

st.caption("*Accuracy here = confidence proxy (max prob for classification, mean conf for detection).*")

# ==========================
# Quick Actions
# ==========================
st.subheader("Quick Actions")
st.markdown("""
- Go to **Image Detection** to draw bounding boxes with YOLO.  
- Go to **Image Classification** to predict one of the four classes.  
- See **Statistics** for session metrics, chart, and logs.  
- Check **About Model** and **How It Works** for documentation.  
""")

st.divider()

# ==========================
# Mode: Deteksi atau Klasifikasi
# ==========================
menu = st.radio("Select Mode:", ["Home", "Image Detection", "Image Classification"], horizontal=True)

if menu == "Image Detection":
    st.header("🔍 Object Detection (YOLO)")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        start_time = time.time()
        results = yolo_model(img)
        inference_time = (time.time() - start_time) * 1000  # in ms

        st.success("Detection complete!")
        st.image(results[0].plot(), caption="Detected Objects", use_container_width=True)
        st.metric("Inference Time", f"{inference_time:.2f} ms")

elif menu == "Image Classification":
    st.header("🧠 Image Classification (CNN)")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        start_time = time.time()
        prediction = classifier.predict(img_array)
        inference_time = (time.time() - start_time) * 1000

        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success("Classification complete!")
        st.metric("Predicted Class", str(class_index))
        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Inference Time", f"{inference_time:.2f} ms")

else:
    st.info("👋 Welcome! Use the top menu to explore Image Detection or Classification features.")
