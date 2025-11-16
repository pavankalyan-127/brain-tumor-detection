import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# --------------------------------------------------------
# 🔍 Find Model Automatically inside the app/ folder
# --------------------------------------------------------
def find_model():
    paths = [
        "app/model.h5",
        "app/model.keras",
        "app/brain_tumor_model.h5"
    ]
    
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# --------------------------------------------------------
# 🧠 Load the model (cached)
# --------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = find_model()
    if model_path is None:
        st.error("❌ No model file found in app/ folder!")
        st.stop()
    st.success(f"Model loaded: {model_path}")
    return tf.keras.models.load_model(model_path)

model = load_model()

# --------------------------------------------------------
# 🧠 Class labels
# --------------------------------------------------------
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# --------------------------------------------------------
# 🖼️ Preprocessing function
# --------------------------------------------------------
def preprocess_image(img):
    img = img.resize((128, 128))          # Change if your model uses different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------------
# 🎨 Streamlit UI
# --------------------------------------------------------
st.title("🧠 Brain Tumor Detection using CNN")
st.write("Upload an MRI image or capture via webcam to classify tumor type.")

mode = st.sidebar.radio("Select Mode:", ["📤 Upload Image", "📸 Capture from Webcam"])

# --------------------------------------------------------
# 📤 Upload Image
# --------------------------------------------------------
if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)

        if st.button("🔍 Predict"):
            input_data = preprocess_image(img)
            preds = model.predict(input_data)[0]

            label = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"### 🧠 Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

# --------------------------------------------------------
# 📸 Webcam Mode
# --------------------------------------------------------
elif mode == "📸 Capture from Webcam":
    st.write("Use your webcam to capture an MRI image.")

    camera_img = st.camera_input("Take a picture")

    if camera_img is not None:
        img = Image.open(camera_img).convert("RGB")
        st.image(img, caption="Captured MRI Image", use_column_width=True)

        if st.button("🔍 Predict"):
            input_data = preprocess_image(img)
            preds = model.predict(input_data)[0]

            label = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"### 🧠 Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")


