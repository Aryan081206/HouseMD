import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('model.h5')
class_labels = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']

# Function to process image
def prepare_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="Radiology Scan Analyzer",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
st.sidebar.info("Upload a medical scan to detect possible diseases.")
st.sidebar.markdown("---")
st.sidebar.success("Made with ❤️ using Streamlit + TensorFlow")

# --- MAIN PAGE ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩻 Radiology Disease Detector")
st.write("**Upload an X-ray / CT scan and get AI-powered disease detection.**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Loading your scan...'):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scan', use_column_width=True)

    if st.button('Predict Disease'):
        with st.spinner('Analyzing scan...'):
            prepared_img = prepare_image(image)
            prediction = model.predict(prepared_img)[0]
            predicted_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_index]
            confidence = prediction[predicted_index] * 100

        # Show results
        st.success(f"**Prediction: {predicted_class} ({confidence:.2f}% confidence)**")

        # Progress bar (for cool visual)
        st.info("Calculating probabilities...")
        progress_text = "Processing probabilities..."
        my_bar = st.progress(0)
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)

        # Show all class probabilities
        st.subheader("Detailed Class Probabilities:")
        for idx, label in enumerate(class_labels):
            st.write(f"**{label}** : {prediction[idx]*100:.2f}%")
else:
    st.warning("Please upload a scan to start diagnosis.")

