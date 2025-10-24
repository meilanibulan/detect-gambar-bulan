import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import cv2

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Bulan Image Detection Dashboard 🌙", layout="wide")

# ==========================
# LOAD MODELS (cached)
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Meilani Bulandari Hsb_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Meilani Bulandari Hsb_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# SIDEBAR MENU
# ==========================
with st.sidebar:
    st.markdown("### Features That\nCan Be Used")
    menu = st.radio(
        "Navigation",
        ["Home", "Image Detection", "Image Classification", "Statistics", "Dataset", "About"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Thank you for using this website")

# ==========================
# STYLE
# ==========================
st.markdown("""
<style>
[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#fff7ef 0%,#f6f0ff 100%) !important;
  border-right: 1px solid rgba(0,0,0,.06);
}
.stApp { background: #fffbe8; }
.block-container{
  background: linear-gradient(145deg,#ffe9e9 0%,#efe7ff 50%,#e8fff3 100%);
  border-radius: 16px;
  padding: 1.5rem 1.8rem;
}
.panel-white{
  background:#fff; border:1px solid rgba(0,0,0,.06);
  border-radius:16px; padding:18px; box-shadow:0 8px 26px rgba(0,0,0,.08);
}
.category-card{
  background:#fff; border:1px solid rgba(0,0,0,.06);
  border-radius:18px; padding:16px; text-align:center;
  box-shadow:0 6px 16px rgba(0,0,0,.08);
  transition:transform .15s ease, box-shadow .15s ease;
}
.category-card:hover{ transform:translateY(-3px); box-shadow:0 12px 28px rgba(0,0,0,.12); }
.cat-pill{
  display:inline-block; padding:6px 14px; border-radius:18px;
  background:#f3f3f7; border:1px solid rgba(0,0,0,.06); margin-bottom:10px; font-weight:700;
}
.cat-img{ width:100%; height:130px; object-fit:cover; border-radius:12px; }
.howto{
  background:#fff; border:1px solid rgba(0,0,0,.06); border-radius:12px;
  padding:16px; box-shadow:0 6px 18px rgba(0,0,0,.06);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HOME PAGE
# ==========================
if menu == "Home":
    st.markdown("## WELCOME TO MY IMAGE DETECTION 🌙")
    st.caption("Welcome to Bulandari's image detection website! Choose the features that best suit your needs.")
    st.markdown("<div class='panel-white'>", unsafe_allow_html=True)
    st.markdown("**You can use this website to detect images by theme:**")

    # layout 2 atas, 2 bawah
    row1_col1, row1_col2 = st.columns(2, gap="large")
    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row1_col1:
        st.markdown("<div class='cat-pill'>Animal</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-card'><img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/5824/5824024.png'/></div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("<div class='cat-pill'>Fashion</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-card'><img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/2258/2258432.png'/></div>", unsafe_allow_html=True)

    with row2_col1:
        st.markdown("<div class='cat-pill'>Food</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-card'><img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/6774/6774898.png'/></div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown("<div class='cat-pill'>Nature</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-card'><img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/4447/4447748.png'/></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # How to use
    st.markdown("""
    <div class='howto'>
      <h4>How to use this dashboard</h4>
      <ol>
        <li>Pilih menu di sidebar: <b>Image Detection</b> atau <b>Image Classification</b>.</li>
        <li>Upload gambar JPG/PNG di panel kiri, lalu klik tombol Run.</li>
        <li>Hasil akan muncul di panel kanan (gambar anotasi / label & confidence).</li>
        <li>Lihat <b>Statistics</b> untuk ringkas jumlah run sesi ini, dan <b>Dataset</b> / <b>About</b> untuk info tambahan.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# IMAGE DETECTION
# ==========================
elif menu == "Image Detection":
    st.header("🔍 Object Detection (YOLO)")
    st.write("Upload an image below and run YOLO detection.")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### 📤 Upload Image")
        f = st.file_uploader("Select an image (JPG/PNG)", type=["jpg","jpeg","png"])
        if f:
            img = Image.open(f)
            st.image(img, caption="Uploaded Image", width=300)
        run = st.button("🚀 Run Detection")
    with c2:
        st.markdown("### 🧠 Detection Result")
        if f and run:
            t0 = time.time()
            results = yolo_model(img)
            t = (time.time()-t0)*1000
            rimg = results[0].plot()
            rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            st.image(rimg, caption="Detected Objects", width=300)
            st.write(f"⏱️ Inference Time: {t:.2f} ms")

# ==========================
# IMAGE CLASSIFICATION
# ==========================
elif menu == "Image Classification":
    st.header("🖼️ Image Classification (CNN)")
    st.write("Upload an image below to classify it using the CNN model.")
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown("### 📤 Upload Image")
        f = st.file_uploader("Select an image (JPG/PNG)", type=["jpg","jpeg","png"], key="cls")
        if f:
            img = Image.open(f)
            st.image(img, caption="Uploaded Image", use_container_width=True)
        run = st.button("🚀 Run Classification")
    with c2:
        st.markdown("### 🧠 Classification Result")
        if f and run:
            arr = image.img_to_array(img.resize((224,224)))
            arr = np.expand_dims(arr, axis=0)/255.0
            t0 = time.time()
            pred = classifier.predict(arr)
            t = (time.time()-t0)*1000
            i = np.argmax(pred); conf = np.max(pred)*100
            label = ["Animal","Fashion","Food","Nature"][i]
            st.success("✅ Classification complete!")
            st.write(f"🎯 **Predicted:** {label}")
            st.write(f"🔥 **Confidence:** {conf:.2f}%")
            st.write(f"⏱️ **Inference Time:** {t:.2f} ms")

# ==========================
# STATISTICS / DATASET / ABOUT
# ==========================
elif menu == "Statistics":
    st.header("📊 Statistics")
    st.info("Here you can display detection/classification run stats per session.")
elif menu == "Dataset":
    st.header("🗂 Dataset")
    st.write("Describe or upload your dataset here.")
elif menu == "About":
    st.header("🌙 About This Dashboard")
    st.write("""
    The **Bulan Image Detection Dashboard** was created to make image recognition tasks
    simple and accessible for everyone.  
    Using advanced computer vision models, this app can automatically detect and classify
    images into four main categories: **Animal**, **Fashion**, **Food**, and **Nature**.
    """)

    st.markdown("### 💡 Purpose")
    st.write("""
    This dashboard was designed for educational and demonstration purposes — to help users
    understand how artificial intelligence can recognize visual patterns from everyday images.
    """)

    st.markdown("### ⚙️ Key Features")
    st.write("""
    - **Image Detection** — Identifies objects within an uploaded image.  
    - **Image Classification** — Predicts which of the four categories the image belongs to.  
    - **Statistics** — Displays detection and classification performance for the current session.  
    - **Dataset** — Provides space for adding or testing your own images.
    """)

    st.markdown("### 👩‍💻 Developer Note")
    st.write("""
    This project was developed by **Bulandari** as part of an academic initiative in computer vision
    and data science. It integrates interactive visualization and AI models into one streamlined app
    built with **Streamlit**.
    """)

    st.markdown("---")
    st.caption("© 2025 Bulandari – All rights reserved. Educational use only.")
