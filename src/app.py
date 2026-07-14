
import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN

# -----------------------------
# Global Configuration & Labels
# -----------------------------
# Unified emotion list (formerly EMOTIONS / RAF_DB_EMOTIONS)
EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]

# -----------------------------
# Cached Loaders
# -----------------------------
@st.cache_resource
def load_uploaded_model(model_path):
    """Loads the compiled Keras model."""
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_face_detector():
    """Caches the MTCNN detector instance to prevent reloading on every run."""
    return MTCNN()

# -----------------------------
# Image Processing Functions
# -----------------------------
def preprocess_image(image, target_size=(100, 100)):
    """Preprocesses a PIL Image or numpy array to match the model input shape."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Handle grayscale or alpha channels
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 4:  # RGBA
        image = image[:, :, :3]

    # Resize to target size (width, height) for cv2.resize
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image


def detect_face(image, detector):
    """Detects and crops the largest face from an image with padding."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure 3-channel RGB format for MTCNN
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            rgb_image = image.copy()
        elif image.shape[2] == 4:
            rgb_image = image[:, :, :3]
        else:
            return None
    else:
        return None

    try:
        results = detector.detect_faces(rgb_image)
        if results:
            # Pick the largest face area found
            largest_face = max(
                results,
                key=lambda x: x["box"][2] * x["box"][3]
            )

            x, y, w, h = largest_face["box"]

            # Add a 20% padding around the box bounding area
            padding = int(max(w, h) * 0.2)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(rgb_image.shape[1], x + w + padding)
            y2 = min(rgb_image.shape[0], y + h + padding)

            # Extract cropped face zone
            return rgb_image[y1:y2, x1:x2]
            
    except Exception as e:
        st.error(f"Face detection error: {e}")
        
    return None


def predict_emotion(model, processed_image):
    """Runs prediction using the loaded model."""
    try:
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return EMOTIONS[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0, None


# =====================================================================
# Main Streamlit Application UI
# =====================================================================
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Emotion Detection")

# Sidebar - Model Configuration
st.sidebar.header("Model Configuration")
model_file = st.sidebar.file_uploader(
    "Upload .keras model",
    type=["keras"]
)

# Application logic wrapper
if model_file:
    # Save uploaded file to local temp storage temporarily to load via Keras
    temp_model_path = "temp_model.keras"
    with open(temp_model_path, "wb") as f:
        f.write(model_file.getbuffer())

    try:
        model = load_uploaded_model(temp_model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Input shape: {model.input_shape}")

        # Extract target size configuration directly dynamically from model metadata
        if len(model.input_shape) == 4:  # (batch, height, width, channels)
            target_h, target_w = model.input_shape[1], model.input_shape[2]
            target_size = (target_w, target_h)  # cv2 expects (width, height)
        else:
            target_size = (100, 100)  # Default fallback configuration

        st.sidebar.info(f"Target size: {target_size}")

        # Detection Settings UI Controls
        st.sidebar.subheader("Settings")
        use_face_detection = st.sidebar.checkbox(
            "Use face detection (Crop to face)",
            value=True
        )

        # Main Body - Image Inputs Processing
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supports both RGB and grayscale images",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            # Metadata Display Columns setup
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            with col2:
                st.write("**Image Info:**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                if hasattr(image, "format"):
                    st.write(f"Format: {image.format}")

            # Prediction Action trigger
            if st.button("🔍 Predict Emotion", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    processed_image = None
                    face_detected = False

                    if use_face_detection:
                        detector = load_face_detector()
                        face = detect_face(image, detector)

                        if face is not None:
                            st.success("✅ Face detected!")
                            # Show cropped target face visual output inside results block
                            st.image(face, caption="Detected Face Zone", width=150)
                            processed_image = preprocess_image(face, target_size)
                            face_detected = True
                        else:
                            st.warning("⚠️ No clear face detected; using default fallback full dimensions.")

                    # Use global image format if bounding setup skips or falls over
                    if processed_image is None:
                        processed_image = preprocess_image(image, target_size)

                    # Compute classification inference passes
                    emotion, confidence, all_preds = predict_emotion(model, processed_image)

                    if emotion is not None:
                        st.header("🎯 Prediction Breakdown")
                        
                        col_metrics1, col_metrics2 = st.columns(2)
                        with col_metrics1:
                            st.metric("Predicted Emotion", emotion)
                        with col_metrics2:
                            st.metric("Confidence Score", f"{confidence:.1%}")

                        # Sort metrics to isolate highest probability groupings
                        sorted_emotions = sorted(
                            zip(EMOTIONS, all_preds),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3]

                        st.subheader("Top 3 Predictions")
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            st.write(f"**{i+1}. {emo}**: {prob:.2%}")

                        # Complete dynamic array status display
                        st.subheader("All Emotion Probabilities")
                        for emo, prob in zip(EMOTIONS, all_preds):
                            st.progress(float(prob), text=f"{emo}: {prob:.2%}")
                    else:
                        st.error("❌ Prediction failed unexpectedly.")

    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        st.error("Please ensure you uploaded a valid .keras compiled model matrix file.")

else:
    st.sidebar.warning("Please upload a .keras model file to configure operational settings.")
    st.info("👆 Upload your trained emotion detection model (.keras structure format) inside the sidebar view pane to begin processing files.")

# Footer Section Separator
st.markdown("---")
st.markdown("<center><b>Emotion Detection System using CNN + MTCNN | Powered by Streamlit</b></center>", unsafe_allow_html=True)