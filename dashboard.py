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
    page_title="Bulan Image Detection Dashboard 🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Load Model (cached)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header
# ==========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: white;
        text-align: left;
    }
    .subtext {
        color: black;
        font-size: 16px;
    }
    .stApp {
        background-color: #fdfdf5;
    }

    /* Opsional: ubah warna teks agar kontras */
    h1, h2, h3, h4, h5, h6, p {
        color: #2c2c2c;
    }
    .category-card {
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        color: black;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .category-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .cat-title {
        font-size: 16px;
        background-color: #eee;
        border-radius: 20px;
        padding: 3px 15px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .cat-img {
        border-radius: 15px;
        width: 100%;
        height: 120px;
        object-fit: cover;
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
# Kartu Kategori
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='category-card' style='background-color:#EDE2FF;'>
        <div class='cat-title'>Animal</div>
        <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/5824/5824024.png' />
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='category-card' style='background-color:#FFD6E0;'>
        <div class='cat-title'>Fashion</div>
        <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/2258/2258432.png' />
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='category-card' style='background-color:#FFF4CC;'>
        <div class='cat-title'>Food</div>
        <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/6774/6774898.png' />
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='category-card' style='background-color:#D9FCE3;'>
        <div class='cat-title'>Nature</div>
        <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/4447/4447748.png' />
    </div>
    """, unsafe_allow_html=True)

st.caption("Each category represents a classification target available in this app.")

st.divider()

# ==========================
# Mode: Deteksi / Klasifikasi
# ==========================
menu = st.radio("Select Mode:", ["Home", "Image Detection", "Image Classification"], horizontal=True)

if menu == "Image Detection":
    st.header("🔍 Object Detection (YOLO)")
    st.write("Upload an image below and run YOLO detection.")

    # layout dua kolom (sedikit dipersempit biar hasilnya kecil)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Select an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)  # 

        run_button = st.button("🚀 Run Detection")

    with col2:
        st.markdown("### 🧠 Detection Result")
        result_placeholder = st.empty()

        if uploaded_file and run_button:
            start_time = time.time()
            results = yolo_model(img)
            inference_time = (time.time() - start_time) * 1000

            # 🔹 ambil hasil deteksi dan ubah format warna BGR → RGB agar tidak berubah
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # 🔹 tampilkan gambar hasil deteksi lebih kecil tapi warna tetap asli
            result_placeholder.image(result_img, caption="Detected Objects", width=300)

            st.write(f"⏱️ Inference Time: **{inference_time:.2f} ms**")

            # tampilkan jumlah objek terdeteksi
            detected_objects = [box.cls for box in results[0].boxes]
            if detected_objects:
                st.write(f"🎯 **Objects detected:** {len(detected_objects)}")
            else:
                st.warning("No objects detected.")

    # styling halus untuk background
    st.markdown("""
    <style>
    .block-container {
        background-color: #FFFBEA;
        padding: 2rem;
        border-radius: 1rem;
    }
    .stButton>button {
        background-color: #E0BBE4;
        color: black;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #D291BC;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.info("👋 Welcome! Use the top menu to explore Image Detection or Classification features.")
