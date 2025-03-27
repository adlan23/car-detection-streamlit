import streamlit as st
from PIL import Image
import os
import subprocess
import shutil
import time
import sys
import matplotlib.pyplot as plt

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
    .title h1, .title p {
        margin-bottom: 0.2rem;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
    <div class='title' style='text-align: center;'>
        <h1 style='font-family: monospace; font-size:40px; color: #3f8efc;'>🚗 CAR DETECTION SYSTEM</h1>
        <p style='font-size:16px; color:#999;'>👨🏻‍💻 Created by Adlan Anuar</p>
        <p><b style='font-family: monospace;'>📤 Upload image to detect car using YOLOv5</b></p>
    </div>
""", unsafe_allow_html=True)

# ---------- SETUP ----------
upload_dir = "uploads"
result_dir = "yolov5/runs/detect"
os.makedirs(upload_dir, exist_ok=True)

# ---------- MODEL CHOICE ----------
model_options = {
    "Vehicle Model (New)": {
        "path": "yolov5/runs/train/vehicle-detectorNEW/weights/vehicle_best.pt",
        "classes": {0: "car", 1: "threewheel", 2: "bus", 3: "truck", 4: "motorbike", 5: "van"}
    },
    "Stanford (Old)": {
        "path": "yolov5/runs/train/stanford-detectorNEW/weights/stanford_best.pt",
        "classes": {i: f"Stanford Car {i}" for i in range(197)}
    },
    "Pretrained COCO (yolov5s.pt)": {
        "path": "yolov5/yolov5s.pt",
        "classes": {2: "car", 5: "bus", 7: "truck", 0: "person", 3: "motorcycle"}
    }
}
model_choice = st.selectbox("🧠 Choose Model:", list(model_options.keys()))
model_path = model_options[model_choice]["path"]
class_names = model_options[model_choice]["classes"]

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

            folders = sorted(os.listdir(result_dir), key=lambda x: os.path.getctime(os.path.join(result_dir, x)))
            latest_folder = os.path.join(result_dir, folders[-1])
            result_img_path = os.path.join(latest_folder, uploaded_file.name)

            time.sleep(1.0)

            if os.path.exists(result_img_path):
                st.success("✅ Detection Completed!")
                st.image(Image.open(result_img_path), caption="Detection Result", use_container_width=True)

                label_path = os.path.join(latest_folder, "labels", uploaded_file.name.replace(".jpg", ".txt").replace(".png", ".txt"))
                class_counter = {}

                if os.path.exists(label_path):
                    st.markdown("### 🔍 Detected Object(s):")
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            conf = float(parts[-1])
                            class_label = class_names.get(class_id, f"Unknown ({class_id})")
                            class_counter[class_label] = class_counter.get(class_label, 0) + 1
                            st.markdown(f"- **{class_label}** | Confidence: `{conf:.2f}`")

                    if class_counter:
                        st.markdown("### 📊 Detection Pie Chart:")
                        fig, ax = plt.subplots()
                        ax.pie(class_counter.values(), labels=class_counter.keys(), autopct='%1.1f%%', startangle=140)
                        ax.axis('equal')
                        st.pyplot(fig)
                else:
                    st.warning("⚠️ No label data found.")
            else:
                st.error("❌ No detection found.")

        os.remove(upload_path)
