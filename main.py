import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.api.preprocessing.image import load_img, img_to_array
import numpy as np


# Load your trained model
@st.cache_resource
def load_trained_model():
    model = load_model("my_model.h5")  # Path to your model
    return model


model = load_trained_model()

# Define the labels (replace with your actual labels)
LABELS = ['elefante', 'farfalla', 'gatto', 'pecora', 'ragno']

# Streamlit app interface
st.title("Image Classification App")
st.write("Upload an image, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the class index with highest probability

    # Display the result
    st.write(f"Predicted Class: {LABELS[predicted_class]}")
