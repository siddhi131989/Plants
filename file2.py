#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# In[4]:


# Function to load model and labels
@st.cache(allow_output_mutation=True)
def load_data():
    model_path = "/Users/siddh/OneDrive/Desktop/file/keras_model.h5"
    label_path = "/Users/siddh/OneDrive/Desktop/file/labels.txt"
    model = load_model(model_path, compile=False)
    class_names = open(label_path, "r").readlines()
    return model, class_names


# In[5]:


# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array


# In[6]:


# Function to make prediction
def predict_disease(image, model):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score


# In[8]:


# Main function
def main():
    st.title("Plantcare: Disease Detection System")
    image_path = "/Users/siddh/OneDrive/Desktop/model final/home_page.jpeg"
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
                model, class_names = load_data()
                class_name, confidence_score = predict_disease(image, model)
                st.write("Class:", class_name[2:])
                st.write("Confidence Score:", confidence_score)

    elif choice == "About":
        st.title("About")
        st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 8K RGB images of healthy and diseased crop leaves which are categorized into 28 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (6400 images)
                2. test (49 images)
                3. validation (1551 images)
                """)
if __name__ == "__main__":
    main()


# In[ ]:




