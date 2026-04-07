import os
import json
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# -------------------------------
# Setup paths
# -------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(working_dir, "plant-disease-predictionl.h5")
json_path = os.path.join(working_dir, "class_indices.json")

# -------------------------------
# Load model
# -------------------------------
model = load_model(model_path)

# -------------------------------
# Load class indices safely
# -------------------------------
with open(json_path, "r") as f:
    class_indices = json.load(f)

# -------------------------------
# Image preprocessing
# -------------------------------
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")  # ensure 3 channels
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

# -------------------------------
# Prediction function
# -------------------------------
def predict_image_class(model, image_file, class_indices):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    predicted_class_name = class_indices[str(predicted_class_index)]

    return predicted_class_name, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image and get disease prediction instantly.")

uploaded_image = st.file_uploader(
    "Upload an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Classify"):
            prediction, confidence = predict_image_class(
                model, uploaded_image, class_indices
            )

            st.success(f"🌱 Prediction: {prediction}")
            st.info(f"📊 Confidence: {confidence*100:.2f}%")