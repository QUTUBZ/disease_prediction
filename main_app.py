# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Load the Model
model = load_model('C:\\Users\\Khaja Qutubuddin\\OneDrive\\Desktop\\Plant-Disease-Detection\\plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust')

# Initialize session state for storing predictions
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Title of the App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button("Predict Disease")

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4D
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        disease = result.split('-')[1]
        plant = result.split('-')[0]

        # Store the prediction and image in session state
        st.session_state["predictions"].append({
            "plant": plant,
            "disease": disease,
            "image": file_bytes
        })

        # Display the result
        st.title(f"This is a {plant} leaf with {disease}")

# Display predictions as cards at the bottom
st.markdown("---")
st.subheader("Predicted Diseases")

if st.session_state["predictions"]:
    for i, pred in enumerate(st.session_state["predictions"]):
        col1, col2 = st.columns([1, 3])
        with col1:
            # Display the saved image
            st.image(cv2.imdecode(np.asarray(bytearray(pred["image"]), dtype=np.uint8), 1), channels="BGR", use_column_width=True)
        with col2:
            # Display the disease prediction
            st.write(f"### Prediction {i + 1}")
            st.write(f"**Plant:** {pred['plant']}")
            st.write(f"**Disease:** {pred['disease']}")
else:
    st.write("No predictions yet.")
