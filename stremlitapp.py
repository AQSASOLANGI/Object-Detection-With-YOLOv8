import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv8 Object Detection App")

model = YOLO("yolov8n.pt")

upload = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if upload:
    img = Image.open(upload)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    results = model.predict(np.array(img))
    output = results[0].plot()

    st.image(output, caption="Detected Output", use_column_width=True)
