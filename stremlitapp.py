import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Object Detection App")
st.write("Upload an image and detect objects!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model.predict(img_array)

    # Convert result to annotated image
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Objects", use_column_width=True)

    # Optionally download result
    result_img = Image.fromarray(annotated)
    result_img.save("result.jpg")

    with open("result.jpg", "rb") as file:
        st.download_button(
            label="Download Result",
            data=file,
            file_name="detected_image.jpg",
            mime="image/jpeg"
        )
