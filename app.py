import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("digit_recognition_system.h5")

# CSS Styles
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
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
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #EA580C !important;
        }
        .prediction-box {
            border: 2px solid #F97316;
            padding: 20px;
            border-radius: 12px;
            background-color: #111827;
            color: #F97316;
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #F97316;'>Digit Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Choose from Draw, Upload or Camera input modes</p>", unsafe_allow_html=True)

# Preprocess Image
def preprocess_image(img_array):
    if img_array.shape[-1] == 4:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    else:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    img_dilated = cv2.dilate(img_thresh, np.ones((3, 3), np.uint8), iterations=1)
    img_resized = cv2.resize(img_dilated, (28, 28))
    img_normalized = img_resized / 255.0
    return img_normalized.reshape(1, 28, 28, 1), img_resized

# Prediction and Confidence Plot
def show_prediction(img_reshaped, processed_img):
    prediction = model.predict(img_reshaped)[0]
    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100

    st.markdown(f'<div class="prediction-box">Predicted Digit: {digit}<br>Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
    st.image(processed_img, caption="Processed Image", width=150)

    # Confidence Bar Plot
    fig, ax = plt.subplots()
    bars = ax.bar(range(10), prediction, color='#F97316')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence per Class")
    ax.set_ylim([0, 1])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + 0.3, yval + 0.01, f'{yval:.2f}', fontsize=8)
    st.pyplot(fig)

# Tabs
tab1, tab2, tab3 = st.tabs(["🖊️ Draw Digit", "📁 Upload Digit", "📷 Camera Input"])

# === Tab 1: Drawing ===
with tab1:
    st.subheader("Draw a digit below:")
    col1, col2 = st.columns([1, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        if st.button("🎯 Predict from Drawing"):
            if canvas_result.image_data is not None:
                img_reshaped, img_processed = preprocess_image(np.array(canvas_result.image_data))
                with col2:
                    show_prediction(img_reshaped, img_processed)
            else:
                st.warning("Draw something first!")

# === Tab 2: Upload ===
with tab2:
    st.subheader("Upload an image of a digit:")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_array = np.array(image)
        img_reshaped = (img_array / 255.0).reshape(1, 28, 28, 1)
        show_prediction(img_reshaped, image)

# === Tab 3: Camera Input ===
with tab3:
    st.subheader("Take a picture of a digit:")
    camera_image = st.camera_input("Capture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("L").resize((28, 28))
        img_array = np.array(image)
        img_reshaped = (img_array / 255.0).reshape(1, 28, 28, 1)
        show_prediction(img_reshaped, image)
