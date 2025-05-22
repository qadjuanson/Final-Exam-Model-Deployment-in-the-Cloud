import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Weather Image Classifier")
st.markdown("Upload an image and let the AI predict the weather condition.")

# Define the class names
class_names = ['Cloudy', 'Fog', 'Rain', 'Shine', 'Sunrise']

# Load the model with error handling
try:
    model = tf.keras.models.load_model("weather_mobilenetv2.h5", compile=False)
except Exception as e:
    st.error("Failed to load the model. Please ensure the .h5 file is correct and doesn't contain unsupported custom layers.")
    st.stop()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Display prediction
        st.success(f"Prediction: **{predicted_class}**")

    except Exception as e:
        st.error(f" Error during prediction: {str(e)}")
