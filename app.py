import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("weather_model.h5")
class_names = ['cloudy', 'rain', 'shine', 'sunrise', 'fog', 'snow']

def preprocess_image(image):
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("Weather Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_tensor = preprocess_image(image)
    prediction = model.predict(img_tensor)
    st.write(f"Prediction: **{class_names[np.argmax(prediction)]}**")
