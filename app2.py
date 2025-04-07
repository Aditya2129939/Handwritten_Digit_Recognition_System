import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("digit_recognition_system.h5")

# Custom CSS for sleek UI
st.markdown(
    """
    <style>
        body {
            background-color: #0d1117;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stTabs [role="tablist"] {
            justify-content: center;
        }
        .stButton>button {
            background-color: #F97316 !important;
            color: white !important;
            font-weight: 600;
            font-size: 16px;
            border-radius: 12px;
            border: none;
            padding: 10px 20px;
            margin-top: 10px;
        }
        .stButton>button:hover {
            background-color: #EA580C !important;
        }
        .stAlert {
            background-color: #1f2937;
        }
        .prediction-box {
            border: 2px solid #F97316;
            padding: 16px;
            border-radius: 12px;
            background-color: #111827;
            color: #F97316;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #F97316;'>Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Choose a mode below: Draw or Upload</p>", unsafe_allow_html=True)

# Function to preprocess the image
def preprocess_image(img_array):
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)
    img_resized = cv2.resize(img_dilated, (28, 28))
    img_resized = img_resized / 255.0
    img_reshaped = img_resized.reshape(1, 28, 28, 1)
    return img_reshaped, img_resized

# Tab-style layout
tab1, tab2 = st.tabs(["🖊️ Draw Digit", "📁 Upload Digit"])

with tab1:
    st.subheader("Digit Recognition - Draw Mode")
    col1, col2 = st.columns([1, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        if st.button("Predict from Drawing"):
            if canvas_result.image_data is not None:
                img_reshaped, img_display = preprocess_image(np.array(canvas_result.image_data))
                prediction = model.predict(img_reshaped)
                digit = np.argmax(prediction)
                with col2:
                    st.markdown(f'<div class="prediction-box">Predicted Digit: {digit}</div>', unsafe_allow_html=True)
                    st.image(img_display, caption="Processed Image", width=150)
            else:
                st.warning("Please draw a digit first!")

with tab2:
    st.subheader("Digit Recognition - Upload Mode")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L")
            image = image.resize((28, 28))
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_reshaped = img_array.reshape(1, 28, 28, 1)
            prediction = model.predict(img_reshaped)
            digit = np.argmax(prediction)

            st.image(image, caption="Uploaded Image", width=150)
            st.markdown(f'<div class="prediction-box">Predicted Digit: {digit}</div>', unsafe_allow_html=True)
