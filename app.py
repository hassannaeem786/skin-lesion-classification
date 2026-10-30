
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬"
)

st.title("Skin Lesion Classification")
st.write("Educational CNN project using HAM10000 dataset")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_lesion_model.keras")

model = load_model()

with open("class_names.json", "r") as f:
    classes = json.load(f)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Classify Image"):
        prediction = model.predict(image_array)
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100

        st.subheader("Model Prediction")
        st.write("Predicted category:", classes[predicted_index])
        st.write(f"Model confidence: {confidence:.2f}%")

        st.warning(
            "Educational demonstration only. "
            "This model is not a medical diagnostic tool. "
            "Predictions may be incorrect."
        )
