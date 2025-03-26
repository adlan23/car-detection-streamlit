import streamlit as st
from PIL import Image
import os
import subprocess
import shutil
import time
import sys


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Car Detection System", layout="centered")

# ---------- STYLING ----------
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #3f8efc;
        color: white;
        padding: 0.5em 1.2em;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2b6bd6;
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center; color: #3f8efc;'>🚗 Car Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload image to detect car using YOLOv5</p>", unsafe_allow_html=True)

# ---------- SETUP ----------
upload_dir = "uploads"
result_dir = "yolov5/runs/detect"
os.makedirs(upload_dir, exist_ok=True)

# ---------- MODEL CHOICE ----------
model_options = {
    "Vehicle Model (New)": "yolov5/runs/train/vehicle-detectorNEW/weights/vehicle_best.pt",
    "Stanford (Old)": "yolov5/runs/train/stanford-train/weights/stanford_best.pt",
    "Pretrained COCO (yolov5s.pt)": "yolov5/yolov5s.pt"
}
model_choice = st.selectbox("🧠 Choose Model:", list(model_options.keys()))
model_path = model_options[model_choice]

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    upload_path = os.path.join(upload_dir, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Now"):
        with st.spinner("Processing... please wait..."):
            command = f"{sys.executable} yolov5/detect.py --weights {model_path} --img 640 --conf 0.25 --source {upload_path} --save-txt --save-conf"
            subprocess.run(command, shell=True)

            # Get latest result folder
            folders = sorted(os.listdir(result_dir), key=lambda x: os.path.getctime(os.path.join(result_dir, x)))
            latest_folder = os.path.join(result_dir, folders[-1])
            result_img_path = os.path.join(latest_folder, uploaded_file.name)

            time.sleep(1.0)

            if os.path.exists(result_img_path):
                st.success("✅ Detection Completed!")
                st.image(Image.open(result_img_path), caption="Detection Result", use_container_width=True)

                label_path = os.path.join(latest_folder, "labels", uploaded_file.name.replace(".jpg", ".txt").replace(".png", ".txt"))
                if os.path.exists(label_path):
                    st.markdown("### 🔍 Detected Class ID(s):")
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            class_id = parts[0]
                            conf = float(parts[-1])
                            st.markdown(f"- Class: {class_id} | Confidence: {conf:.2f}")
                else:
                    st.warning("⚠️ No label data found.")
            else:
                st.error("❌ No detection found.")

        # Optional: Delete uploaded image after use
        os.remove(upload_path)