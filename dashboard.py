import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Models
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bulan Dashboard", page_icon="🌝", layout="wide")

LABELS   = ['animal', 'fashion', 'food', 'nature']
YOLO_PATH = "model/Meilani Bulandari Hsb_Laporan 4.pt"
CLF_PATH  = "model/Meilani Bulandari Hsb_Laporan 2.h5"

classifier = None
yolo_model = None

# State init
if "page" not in st.session_state:    st.session_state["page"]  = "Home"
if "filter" not in st.session_state:  st.session_state["filter"] = "All"
if "logs" not in st.session_state:    st.session_state["logs"]   = []
if "scores" not in st.session_state:  st.session_state["scores"] = []
if "det_count" not in st.session_state: st.session_state["det_count"] = 0

st.markdown("""
<style>
/* === COLOR PATCH (kontras + keterbacaan) === */

/* 1) Token warna — lebih gelap di bg, lebih terang di teks & teal */
:root{
  --bg-1:#081018;           /* lebih gelap */
  --bg-2:#0A1520;
  --bg-3:#0F2430;
  --text:#F4FBFD;           /* teks utama lebih putih */
  --muted:#C8D6DE;          /* teks sekunder lebih terang */
  --teal-1:#C7FBF3;         /* mint terang */
  --teal-2:#44E6D2;         /* aksen utama (lebih vivid) */
  --teal-3:#14B9AE;         /* aksen sekunder */
  --teal-4:#0E6E69;
  --glass:rgba(255,255,255,.08);
}

/* 2) Nav pills — teks/ikon lebih jelas saat aktif & non-aktif */
.stButton > button{
  color: var(--text) !important;
}
.stButton > button:disabled{
  background: linear-gradient(160deg, var(--teal-2) 0%, var(--teal-1) 70%) !important;
  color:#092126 !important; /* kontras lebih tinggi */
}

/* 3) KPI cards — border & isi lebih “pop” */
.card--peach{
  background: linear-gradient(135deg, rgba(31,158,140,.45) 0%, rgba(12,94,89,.40) 100%),
              linear-gradient(180deg, #0d1a22 0%, #102632 100%);
  border: 1px solid rgba(68,230,210,.55);
}
.card--peach h3{ color:#FFFFFF; }
.card--peach .big{ color:#FFFFFF; }
.card--peach .pill{
  background: rgba(199,251,243,.12);
  color:#ECFFFD;
  border:1px solid rgba(68,230,210,.55);
}

/* 4) Panel Upload/Result/Statistics — teks dan tepi lebih terang */
.panel{
  background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg-3) 100%);
  border: 1px solid rgba(255,255,255,.12);
  color: var(--text);
}
.panel h3{ color:#FFFFFF; }

/* 5) File uploader — garis & teks terlihat jelas */
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,.06) !important;
  border: 1.8px dashed var(--teal-2) !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] *{
  color:#F2FFFF !important;
  opacity:1 !important;
}

/* 6) Divider & teks umum */
hr{ border-color: rgba(255,255,255,.14) !important; }
.muted, .stMarkdown p, .stMarkdown li{ color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# MODELS (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    yolo, clf = None, None
    try:
        yolo = YOLO(YOLO_PATH)
    except Exception as e:
        st.warning(f"YOLO model gagal dimuat: {e}")
    try:
        clf = tf.keras.models.load_model(CLF_PATH, compile=False)
    except Exception as e:
        st.warning(f"Classifier gagal dimuat: {e}")
    return yolo, clf

# =========================
# NAVBAR
# =========================
def navbar(active: str):
    labels = ["Home", "Image Detection", "Image Classification", "Statistics", "About Model", "How It Works"]
    st.markdown("<div class='navbar'>", unsafe_allow_html=True)
    cols = st.columns(len(labels), gap="small")
    for i, lab in enumerate(labels):
        with cols[i]:
            if lab == active:
                st.button(lab, disabled=True, key=f"nav_{lab}")
            else:
                if st.button(lab, key=f"nav_{lab}"):
                    st.session_state["page"] = lab
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

navbar(st.session_state["page"])
st.markdown("<div class='after-nav'></div>", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def to_png_bytes(img_np_bgr) -> bytes:
    """Convert Ultralytics plotted (BGR numpy) -> PNG bytes (no OpenCV)."""
    rgb = img_np_bgr[..., ::-1]               # BGR -> RGB
    pil = Image.fromarray(rgb.astype("uint8"))
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return buf.getvalue()

def add_log(filename, mode, label, conf):
    st.session_state["logs"].append({"file": filename, "mode": mode, "label": label, "confidence": float(conf)})
    st.session_state["scores"].append(float(conf))

def kpi_cards(inf_ms: float, det_total: int):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card--peach'><h3>Available</h3><div class='big'>{det_total}</div><div class='pill'>Total detections</div></div>", unsafe_allow_html=True)
    with c2:
        acc = (np.mean(st.session_state["scores"])*100) if st.session_state["scores"] else 0.0
        st.markdown(f"<div class='card--peach'><h3>Income</h3><div class='big'>{acc:.2f}%</div><div class='pill'>Session accuracy*</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card--peach'><h3>Expense</h3><div class='big'>{inf_ms:.1f} ms</div><div class='pill'>Inference time</div></div>", unsafe_allow_html=True)
    st.caption("*Accuracy here = confidence proxy (max prob for classification, mean conf for detection).")

def transactions_list():
    st.markdown("### Transactions")
    for r in reversed(st.session_state["logs"][-12:]):
        color = "#A1E887" if r["confidence"] >= 0.5 else "#F98F8F"
        st.markdown(
            f"<div class='txn'><div style='display:flex; justify-content:space-between;'>"
            f"<div><b>{r['file']}</b><div class='muted'>{r['mode']} → {r['label']}</div></div>"
            f"<div style='font-weight:800; color:{color};'>{r['confidence']:.3f}</div></div></div>",
            unsafe_allow_html=True
        )

def log_df() -> pd.DataFrame:
    if not st.session_state["logs"]:
        return pd.DataFrame(columns=["file","mode","label","confidence"])
    return pd.DataFrame(st.session_state["logs"])

# =========================
# PAGES
# =========================
page = st.session_state["page"]

if page == "Home":
    st.markdown("<h1>BULAN IMAGE DETECTION DASHBOARD 🌙</h1>", unsafe_allow_html=True)
    st.write("This app performs **Image Detection** (YOLO) and **Image Classification** (CNN) for **animal, fashion, food, and nature** images.")
    st.markdown("---")
    kpi_cards(inf_ms=0.0, det_total=st.session_state["det_count"])
    st.markdown("### Quick Actions")
    st.write("- Go to **Image Detection** to draw bounding boxes with YOLO.")
    st.write("- Go to **Image Classification** to predict one of the four classes.")
    st.write("- See **Statistics** for session metrics, chart, and logs.")
    st.write("- Check **About Model** and **How It Works** for documentation.")

elif page == "Image Detection":
    st.markdown("<h1>Image Detection</h1>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("<div class='panel'><h3>Upload</h3>", unsafe_allow_html=True)
        f = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="det_up")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='gold-frame'>", unsafe_allow_html=True)
            st.image(img, caption="Preview", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'><h3>Result</h3>", unsafe_allow_html=True)
        if f:
            if not yolo_model:
                st.error("YOLO model not available.")
            else:
                t0 = time.time()
                res = yolo_model(img, verbose=False)
                ms = (time.time() - t0) * 1000
                plotted = res[0].plot()  # numpy (BGR)

                st.markdown("<div class='aqua-frame'>", unsafe_allow_html=True)
                st.image(plotted[..., ::-1], caption="Detection", use_container_width=True)  # BGR -> RGB
                st.markdown("</div>", unsafe_allow_html=True)

                boxes = res[0].boxes
                det_total = int(boxes.shape[0]) if boxes is not None else 0
                mean_conf = float(boxes.conf.mean().item()) if (boxes is not None and len(boxes) > 0) else 0.0
                st.session_state["det_count"] += det_total
                add_log(f.name, "detection", "objects", mean_conf)

                kpi_cards(inf_ms=ms, det_total=st.session_state["det_count"])

                png_bytes = to_png_bytes(plotted)
                st.download_button("Download annotated image (PNG)", data=png_bytes,
                                   file_name=f"{f.name.rsplit('.',1)[0]}_detected.png", mime="image/png")
        else:
            st.info("Upload an image to run detection.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Image Classification":
    st.markdown("<h1>Image Classification</h1>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("<div class='panel'><h3>Upload</h3>", unsafe_allow_html=True)
        f = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="cls_up")
        if f:
            img = Image.open(f).convert("RGB")
            st.markdown("<div class='gold-frame'>", unsafe_allow_html=True)
            st.image(img, caption="Preview", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'><h3>Result</h3>", unsafe_allow_html=True)
        if f:
            if not classifier:
                st.error("Classifier model not available.")
            else:
                t0 = time.time()
                imr = img.resize((224, 224))
                arr = keras_image.img_to_array(imr)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = classifier.predict(arr, verbose=0)
                idx = int(np.argmax(pred))
                conf = float(np.max(pred))
                label = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
                ms = (time.time() - t0) * 1000

                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {conf:.4f}")
                kpi_cards(inf_ms=ms, det_total=st.session_state['det_count'])
                add_log(f.name, "classification", label, conf)
        else:
            st.info("Upload an image to classify.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Statistics":
    st.markdown("<h1>Statistics</h1>", unsafe_allow_html=True)
    avg_acc = (np.mean(st.session_state["scores"]) * 100) if st.session_state["scores"] else 0.0
    st.markdown(f"<div class='badge-balance'><div class='label'>Total Accuracy (Session)</div><div class='value'>{avg_acc:.2f}%</div><small class='muted'>proxy from confidences</small></div>", unsafe_allow_html=True)

    if st.session_state["scores"]:
        df = pd.DataFrame({"upload": list(range(1, len(st.session_state["scores"])+1)),
                           "accuracy": [s*100 for s in st.session_state["scores"]]})
        df = df.set_index("upload")
        st.markdown("### Accuracy per Upload")
        st.line_chart(df, height=260, use_container_width=True)
    else:
        st.info("No uploads yet to plot.")

    transactions_list()
    df_log = log_df()
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download session log (CSV)", data=csv, file_name="vision_session_log.csv", mime="text/csv")

elif page == "About Model":
    st.markdown("<h1>About Model</h1>", unsafe_allow_html=True)
    st.write("- **Detector:** YOLOv8 (custom weights) for locating objects.")
    st.write(f"- **Classifier:** TensorFlow/Keras CNN (`.h5`) predicting one of: {', '.join(LABELS)}.")
    st.write("- **Preprocess:** 224×224, rescale 1/255.")
    st.write("- **Accuracy proxy:** using confidences until ground-truth labels are provided.")

elif page == "How It Works":
    st.markdown("<h1>How It Works</h1>", unsafe_allow_html=True)
    st.write("1. Upload an image on the desired page.")
    st.write("2. For Detection, YOLO draws bounding boxes and we compute **mean box confidence**.")
    st.write("3. For Classification, CNN outputs probabilities; we take the **max probability** as confidence.")
    st.write("4. All results are logged and summarized under **Statistics** with a chart and downloads.")
