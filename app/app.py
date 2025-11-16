import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# -----------------------------
# 🔍 Auto-detect available model
# -----------------------------
def find_model():
    
    if os.path.exists("model.h5"):
        return "model.h5"
    elif os.path.exists("model.keras"):
        return "model.keras"
    else:
        st.error("❌ No model file found! Upload model.h5 or model.keras in the same folder.")
        st.stop()

MODEL_PATH = find_model()

# -----------------------------
# 🧩 Load your trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
st.success(f"✅ Loaded model: **{MODEL_PATH}**")

# 🧠 Define class labels
CLASS_NAMES = ['Glioma', 'Meningioma',' No Tumor','Pituitary']

st.title("🧠 Brain Tumor Detection using CNN")
st.write("Upload an MRI image or capture from webcam to classify tumor type.")

# Sidebar options
option = st.sidebar.radio("Choose Input Mode", ("Upload Image", "Capture from Webcam"))

# -----------------------------
# 🖼️ Preprocess Function
# -----------------------------
def preprocess_image(img):
    img = img.resize((128, 128))  # change this to your training input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# 📤 Upload Image
# -----------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        if st.button("🔍 Predict Tumor"):
            input_data = preprocess_image(image)
            preds = model.predict(input_data)[0]
            label = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"🧠 Prediction: **{label}** ({confidence:.2f}% confidence)")

# -----------------------------
# 📸 Webcam Capture
# -----------------------------
elif option == "Capture from Webcam":
    st.write("Click below to capture an MRI-like image.")

    camera_image = st.camera_input("Capture MRI Image")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured MRI Image", use_container_width=True)

        if st.button("🔍 Predict Tumor"):
            input_data = preprocess_image(image)
            preds = model.predict(input_data)[0]
            label = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"🧠 Prediction: **{label}** ({confidence:.2f}% confidence)")
# import os
# print("FILES IN CURREN
#T FOLDER:", os.listdir("."))
# import streamlit as st
