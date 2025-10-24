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

if "hist" not in st.session_state:
    st.session_state["hist"] = {
        "det": [],  # detection history
        "cls": []   # classification history
    }

if "log" not in st.session_state:
    st.session_state["log"] = []
    
# ==========================
# SIDEBAR MENU
# ==========================
with st.sidebar:
    st.markdown("### Features That Can Be Used")
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
  background: linear-gradient(120deg, #c5429c 73%, #b2ba15 17%) !important;
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

.block-container{
  background: linear-gradient(180deg,#f7f5ff 0%,#fefaf3 100%) !important;
  border-radius: 0 !important;         
  margin-top: 0 !important;
  padding: 3rem 4rem !important;
  width: 100% !important;
  max-width: 100% !important;
  min-height: 100vh !important;         
  box-sizing: border-box;
}

/* --- KARTU KATEGORI --- */
.category-card{
  background:#fff; border:1px solid rgba(0,0,0,.06);
  border-radius:22px; padding:24px; text-align:center;
  box-shadow:0 6px 16px rgba(0,0,0,.08);
  transition:transform .15s ease, box-shadow .15s ease;
  height: 220px;                     
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
    st.markdown("# **WELCOME TO IMAGE DETECTION WEBSITE**🎀")
    st.caption("#### Welcome to Bulandari's image detection website! Choose the features that best suit your needs.")
    st.markdown("<div class='black'>", unsafe_allow_html=True)
    st.markdown("#### **You can use this website to detect images by theme:**")

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
      <h4>How to use this site:</h4>
      <ol>
        <li>Select the menu in the sidebar: <b>Image Detection</b> or <b>Image Classification</b>.</li>
        <li>Upload a JPG/PNG image in the left panel, then click the Run button.</li>
        <li>The results will appear in the right panel (image annotation/label and confidence).</li>
        <li>See <b>Statistics</b> for a summary of the number of runs for this session, dan <b>About</b> for additional information.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# IMAGE DETECTION
# ==========================
elif menu == "Image Detection":
    st.header("🔍 Object Detection (YOLO)")
    st.write("Upload an image below and run YOLO detection.")

    # Layout dua kolom
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 📤 Upload Image")
        f = st.file_uploader("Select an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="det")

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

            # ambil label objek yang terdeteksi
            detected_objects = []
            for box in results[0].boxes:
                cls_id = int(box.cls)
                label = results[0].names[cls_id]
                detected_objects.append(label)

            num_objs = len(detected_objects)

            # tampilkan di layar
            if num_objs > 0:
                st.write(f"🎯 **Objects Detected ({num_objs}):** {', '.join(detected_objects)}")
            else:
                st.warning("No objects detected.")

            # simpan ke history dan log
            st.session_state["hist"]["det"].append({
                "ts": time.time(),
                "ms": float(t),
                "objects": detected_objects
            })
            st.session_state["log"].append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "detection",
                "file": getattr(f, "name", "-"),
                "objects": detected_objects,
                "ms": float(t)
            })

            
# ==========================
# IMAGE CLASSIFICATION
# ==========================
elif menu == "Image Classification":
    st.header("🖼️ Image Classification (CNN)")
    st.write("Upload an image below to classify it using the CNN model.")

    # Layout dua kolom
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 📤 Upload Image")
        f = st.file_uploader("Select an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="cls")

        if f:
            img = Image.open(f)
            st.image(img, caption="Uploaded Image", use_container_width=True)

        run = st.button("🚀 Run Classification")

    with c2:
        st.markdown("### 🧠 Classification Result")

        if f and run:
            # ==========================
            # Preprocessing
            # ==========================
            arr = image.img_to_array(img.resize((224, 224)))
            arr = np.expand_dims(arr, axis=0) / 255.0

            # ==========================
            # Prediction
            # ==========================
            t0 = time.time()
            pred = classifier.predict(arr)
            t = (time.time() - t0) * 1000

            i = np.argmax(pred)
            conf = np.max(pred) * 100
            labels = ["Animal", "Fashion", "Food", "Nature"]
            label = labels[i] if i < len(labels) else f"Class {i}"

            # ==========================
            # Output ke Streamlit
            # ==========================
            st.success("✅ Classification complete!")
            st.write(f"🎯 **Predicted Class:** {label}")
            st.write(f"🔥 **Confidence:** {conf:.2f}%")
            st.write(f"⏱️ **Inference Time:** {t:.2f} ms")

            # ==========================
            # Logging ke session_state
            # ==========================
            if "hist" not in st.session_state:
                st.session_state["hist"] = {"cls": []}
            if "log" not in st.session_state:
                st.session_state["log"] = []

            st.session_state["hist"]["cls"].append({
                "ts": time.time(),
                "ms": float(t),
                "label": label,
                "conf": float(conf)
            })

            st.session_state["log"].append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "classification",
                "file": getattr(f, "name", "-"),
                "label": label,
                "confidence": float(conf),
                "ms": float(t)
            })

# ==========================
# STATISTICS / DATASET / ABOUT
# ==========================
elif menu == "Statistics":
    st.header("📊 Statistics")

    # --- Pastikan state ada (menghindari KeyError) ---
    hist = st.session_state.setdefault("hist", {"det": [], "cls": []})
    log  = st.session_state.setdefault("log",  [])

    det_runs = hist["det"]
    cls_runs = hist["cls"]

    if not det_runs and not cls_runs:
        st.info("No data yet. Run *Image Detection* or *Image Classification* first.")
    else:
        import pandas as pd 
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Detection Runs", len(det_runs))
        with c2:
            st.metric("Total Classification Runs", len(cls_runs))
        with c3:
            all_ms = [r["ms"] for r in det_runs] + [r["ms"] for r in cls_runs]
            avg_ms = (sum(all_ms) / len(all_ms)) if all_ms else 0.0
            st.metric("Avg Inference Time", f"{avg_ms:.1f} ms")

        st.markdown("### ⏱️ Inference Time per Run")
        if det_runs:
            df_det = pd.DataFrame({
                "run": list(range(1, len(det_runs)+1)),
                "ms": [r["ms"] for r in det_runs],
                "objects": [r.get("objects", 0) for r in det_runs],
            }).set_index("run")
            st.subheader("Detection")
            st.line_chart(df_det[["ms"]], height=220, use_container_width=True)
            st.caption("Jumlah objek per run: " + ", ".join(map(str, df_det["objects"].tolist())))
        else:
            st.info("Belum ada data Detection.")

        if cls_runs:
            df_cls = pd.DataFrame({
                "run": list(range(1, len(cls_runs)+1)),
                "ms": [r["ms"] for r in cls_runs],
                "label": [r["label"] for r in cls_runs],
                "confidence": [r["conf"]*100 for r in cls_runs],
            }).set_index("run")
            st.subheader("Classification")
            st.line_chart(df_cls[["ms"]], height=220, use_container_width=True)
            st.caption("Label: " + ", ".join(df_cls["label"].tolist()))
        else:
            st.info("Belum ada data Classification.")

        st.markdown("### 🧾 Session Log")
        if log:
            df_log = pd.DataFrame(log)
            st.dataframe(df_log, use_container_width=True, height=260)
        else:
            st.write("Log kosong.")

elif menu == "About":
    st.markdown("### 🌙 About This Dashboard")
    st.write("""
    **Bulan Image Detection Dashboard** was developed as a learning tool and demonstration tool for the application 
    of Artificial Intelligence (AI) technology in the field of Computer Vision.
    This dashboard allows users to perform object detection and image classification in an interactive and easy-to-use manner.
    """)

    st.markdown("### 🎯 Development Goals")
    st.write("""
    This dashboard aims to provide an overview of how deep learning models recognize visual patterns in 
    images and to help students and general users understand the basic concepts of 
    **Object Detection** and **Image Classification**.
    """)

    st.markdown("### ⚙️ Key Features")
    st.write("""
    - **Image Detection** – Detecting objects in images using the YOLO model.  
    - **Image Classification** – Grouping images into four categories: *Animal, Fashion, Food,* and *Nature.*  
    - **Statistics** –Displays detection/classification results and performance during the running session.
    """)

    st.markdown("### 👩‍💻 Owner")
    st.write("""
    This dashboard was created by **Meilani Bulandari Hasibuan**, a Statistics student at Syiah Kuala University in 2023.
    """)

    st.markdown("---")
    st.caption("© 2025 Meilani Bulandari Hasibuan")
