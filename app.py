import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model
from my_custom_objects import MyCustomLayer
# Load model
model = load_model("weather_mobilenetv2.h5", custom_objects={"MyCustomLayer": MyCustomLayer})
class_names = ['Cloudy', 'Fog', 'Rain', 'Shine', 'Sunrise']

st.title("Weather Image Classifier ")
st.markdown("Upload an image and let the AI predict the weather condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Prediction: **{predicted_class}**")
