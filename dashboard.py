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
st.set_page_config(page_title="Bulan Image Detection Dashboard", page_icon="🌙", layout="wide")

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
        ["Home", "Image Detection", "Image Classification", "Statistics", "About"],
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
/* --- SIDEBAR --- */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#fff7ef 0%,#f6f0ff 100%) !important;
  border-right: 1px solid rgba(0,0,0,.06);
  width: 220px !important;
  min-width: 220px !important;
  max-width: 220px !important;
  padding: 24px 16px !important;
}

/* Jarak antar menu */
[data-testid="stSidebar"] [data-testid="stRadio"] > div{
  display: flex;
  flex-direction: column;
  gap: 14px !important;
  margin-top: 12px !important;
}

/* Item aktif */
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){
  background: linear-gradient(90deg,#fdd1f7 0%, #f5e0ff 100%);
  border-radius: 10px;
  color: #000 !important;
  font-weight: 600;
  padding: 6px 10px;
}

/* Font item sidebar */
[data-testid="stSidebar"] label{
  font-size: 15px;
  font-weight: 500;
  color: #5a5a5a !important;
}

/* --- AREA UTAMA --- */
.stApp {
  background: #fffbe8;
}
.block-container{
  background: linear-gradient(145deg,#ffe9e9 0%,#efe7ff 50%,#e8fff3 100%);
  border-radius: 16px;
  margin-top: 30px !important;
  padding: 3rem 4rem !important;         /* isi halaman lebih lebar */
  width: 100% !important;                /* full layar */
  max-width: 100% !important;
  min-height: 100vh !important;          /* biar tinggi memenuhi layar */
  box-sizing: border-box;
}

/* --- KARTU KATEGORI --- */
.category-card{
  background:#fff; border:1px solid rgba(0,0,0,.06);
  border-radius:22px; padding:24px; text-align:center;
  box-shadow:0 6px 16px rgba(0,0,0,.08);
  transition:transform .15s ease, box-shadow .15s ease;
  height: 220px;                         /* biar kotaknya seragam dan isi seimbang */
}
.category-card:hover{
  transform:translateY(-3px);
  box-shadow:0 12px 28px rgba(0,0,0,.12);
}
.cat-img{
  width:100%; height:130px; object-fit:contain;
  border-radius:12px; margin-top:12px;
}

/* --- BLOK HOW TO --- */
.howto{
  background:#fff; border:1px solid rgba(0,0,0,.06);
  border-radius:14px; padding:24px;
  box-shadow:0 6px 18px rgba(0,0,0,.06);
}

/* --- GRID & SPACING --- */
[data-testid="stHorizontalBlock"]{ gap: 28px !important; }
.vspace-16{ height: 16px; }
.vspace-24{ height: 24px; }
.vspace-32{ height: 32px; }

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
    row1_col1, row1_col2 = st.columns(2, gap="medium")
    row2_col1, row2_col2 = st.columns(2, gap="medium")

    with row1_col1:
         st.markdown("""
        <div class='category-card' style='background-color:#EDE2FF;'>
            <div class='cat-title'>Animal</div>
            <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/5824/5824024.png' />
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
         st.markdown("""
        <div class='category-card' style='background-color:#FFD6E0;'>
            <div class='cat-title'>Fashion</div>
            <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/2258/2258432.png' />
        </div> <div class='vspace-32'></div>
        """, unsafe_allow_html=True)
        
    with row2_col1:
        st.markdown("""
        <div class='category-card' style='background-color:#FFF4CC;'>
            <div class='cat-title'>Food</div>
            <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/6774/6774898.png' />
        </div>
        """, unsafe_allow_html=True)
        
    with row2_col2:
        st.markdown("""
        <div class='category-card' style='background-color:#D9FCE3;'>
            <div class='cat-title'>Nature</div>
            <img class='cat-img' src='https://cdn-icons-png.flaticon.com/512/4447/4447748.png' />
        </div>
        """, unsafe_allow_html=True)
        
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

elif menu == "About":
    st.header("🌙 Tentang Dashboard Ini")
    st.write("""
    **Bulan Image Detection Dashboard** dikembangkan sebagai media pembelajaran dan demonstrasi
    penerapan teknologi **Kecerdasan Buatan (AI)** di bidang **Computer Vision**.
    Melalui dashboard ini, pengguna dapat melakukan deteksi objek dan klasifikasi gambar
    secara interaktif dan mudah digunakan.
    """)

    st.markdown("### 🎯 Tujuan Pengembangan")
    st.write("""
    Dashboard ini bertujuan untuk memberikan gambaran bagaimana model deep learning
    mampu mengenali pola visual dari gambar, serta membantu mahasiswa dan pengguna umum
    memahami konsep dasar **deteksi objek (Object Detection)** dan **klasifikasi gambar (Image Classification)**.
    """)

    st.markdown("### ⚙️ Fitur Utama")
    st.write("""
    - **Image Detection** – Mendeteksi objek yang terdapat di dalam gambar menggunakan model YOLO.  
    - **Image Classification** – Mengelompokkan gambar ke dalam empat kategori: *Animal, Fashion, Food,* dan *Nature.*  
    - **Statistics** – Menampilkan hasil dan performa deteksi/klasifikasi selama sesi berjalan.  
    - **Dataset** – Menyediakan ruang untuk menambah atau menguji gambar secara mandiri.
    """)

    st.markdown("### 👩‍💻 Pengembang")
    st.write("""
    Dashboard ini dibuat oleh **Bulandari** sebagai bagian dari proyek akademik di bidang
    **Data Science** dan **Computer Vision**, menggunakan framework **Streamlit** untuk antarmuka interaktif.
    """)

    st.markdown("---")
    st.caption("© 2025 Bulandari – Untuk tujuan edukasi dan penelitian.")
