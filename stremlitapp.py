# =========================
# app.py — YOLOv8 Streamlit App
# =========================

import streamlit as st
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import tempfile

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("YOLOv8 Object Detection with Streamlit")

# ------------------------
# Sidebar - Upload File
# ------------------------
st.sidebar.header("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Choose an image or video", type=["jpg","jpeg","png","mp4","mov"])

# ------------------------
# Load YOLOv8 Model
# ------------------------
st.sidebar.header("YOLOv8 Model")
model_option = st.sidebar.selectbox("Select YOLOv8 model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
model = YOLO(model_option)

# ------------------------
# Image Detection
# ------------------------
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext in ["jpg", "jpeg", "png"]:
        # Save uploaded image temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        # Run YOLO prediction
        results = model(temp_file_path)

        # Save output
        output_path = "output_image.jpg"
        results[0].save(output_path)
        st.image(output_path, caption="YOLOv8 Output", use_column_width=True)

    elif file_ext in ["mp4", "mov"]:
        # Save uploaded video temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        st.video(temp_file_path, format="video/mp4", start_time=0)

        # Run YOLO prediction on video
        results = model.predict(
            source=temp_file_path,
            save=True,
            project="runs",
            name="video_output",
            save_frames=False
        )

        output_video_path = os.path.join("runs/video_output", os.listdir("runs/video_output")[0])
        st.video(output_video_path, format="video/mp4")

# ------------------------
# Instructions
# ------------------------
st.sidebar.markdown("""
### Instructions:
1. Upload an image (jpg/png) or video (mp4/mov).
2. Select the YOLOv8 model.
3. View the output image or video.
""")

