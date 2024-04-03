import streamlit as st
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.layers import Layer, DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

# Define custom DepthwiseConv2D layer
class CustomDepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CustomDepthwiseConv2D, self).__init__()
        self.depthwise_conv2d = DepthwiseConv2D(kernel_size=(3, 3), padding='same', **kwargs)

    def call(self, inputs):
        return self.depthwise_conv2d(inputs)

# Function to load model with custom layer
def load_model_with_custom_layer(model_path):
    try:
        # Define custom objects to remove 'groups' parameter
        custom_objects = {
            'DepthwiseConv2D': lambda kwargs: tf.keras.layers.DepthwiseConv2D(kwargs.pop('name', None), **kwargs)
        }
        # Load the model with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error("Error loading model: {}".format(str(e)))
        return None

# Function to load labels
@st.cache_data
def load_labels(label_path):
    try:
        class_names = open(label_path, "r").readlines()
        return class_names
    except Exception as e:
        st.error("Error loading labels: {}".format(str(e)))
        return None

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Function to make prediction
def predict_disease(image, model, class_names):
    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = preprocess_image(image)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        return class_name, confidence_score
    except Exception as e:
        st.error("Error predicting disease: {}".format(str(e)))
        return None, None

# Main function
def main():
    st.title("Plantcare: Disease Detection System")
    image_path = "home_page.jpeg"
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the PlantCare: Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
    
    if choice == "Home":
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect Disease"):
                model_path = "keras_model.h5"
                label_path = "labels.txt"
                model = load_model_with_custom_layer(model_path)
                class_names = load_labels(label_path)
                if model is not None and class_names is not None:
                    class_name, confidence_score = predic
