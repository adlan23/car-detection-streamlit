# 🚗 Car Detection System

This project is a web-based car detection system that uses the YOLOv5 object detection model, integrated with a Streamlit GUI, and deployed to Streamlit Cloud for live usage.

---

## 🧠 Project Objective

To build a complete pipeline for detecting cars in uploaded images, using deep learning, from dataset preprocessing to model training, GUI development, and final web deployment.

---

## 📁 Dataset

Originally, the project was required to use the **Stanford Cars Dataset**. However, after thorough testing and annotation inspection, it was discovered that the dataset's `.mat` file (`cars_annos.mat`) contained inconsistencies and misaligned label structures, leading to:
- Incorrect or missing bounding boxes
- Class labels not matching car images
- Poor detection performance after training

⚠️ **As a result, a pretrained YOLOv5 model was used as a backup to ensure full functionality of the system.**

---

## 🧩 Features

- Upload image of a car
- Detect object(s) using **YOLOv5**
- Display bounding boxes and **class IDs**
- View confidence score per detection
- Option to switch between:
  - ✅ Pretrained YOLOv5 (`yolov5s.pt`)
  - ⚙️ Custom-trained model (if available)
- Fully deployed online

---

## 🚀 Live App

👉 [Click to open the deployed Streamlit app](https://car-detection-streamlit.streamlit.app)

---

## ⚙️ Tech Stack

- Python 3.10
- PyTorch (torch)
- OpenCV
- YOLOv5 (Ultralytics)
- Streamlit
- GitHub + Streamlit Cloud

---

## 📦 How to Run Locally

### 1. Clone this repo:
```bash
git clone https://github.com/adlan23/car-detection-streamlit.git
cd car-detection-streamlit
```

### 2. Create & activate environment:
```bash
conda create -n car-detect python=3.10 -y
conda activate car-detect
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the app:
```bash
streamlit run streamlit_app.py
```

---

## 🧠 How It Works

### Model:
- YOLOv5 object detection model used
- Supports both pretrained and custom models
- Inference runs on CPU (compatible with Streamlit Cloud)

### GUI:
- Built with Streamlit
- Supports file uploads & preview
- Displays detection output with labels and confidence

### Deployment:
- Deployed on [Streamlit Cloud](https://streamlit.io/cloud)
- System-level packages (e.g., `libgl1`) installed via `packages.txt` to support OpenCV

---

## 📝 Submission Checklist ✅

| Requirement                      | Status     |
|----------------------------------|------------|
| Use deep learning model (YOLOv5) | ✅ Done     |
| GUI for upload & detection       | ✅ Done     |
| Web deployment                   | ✅ Done     |
| Evaluation (mAP, IoU*)           | ✅ Partially done (training attempted, but Stanford Cars dataset caused issues) |
| Stanford Dataset used            | ⚠️ Attempted, but dropped due to annotation errors |

---

## 📌 Notes

- Detection results may vary based on model (custom vs pretrained)
- The project is modular and can be easily extended or integrated with another dataset

---

## 🧑‍💻 Created By

**Adlan Anuar**   
March 2025