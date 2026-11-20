import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from mtcnn import MTCNN


# ============================================================
# PROJECT ROOT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# INTERNSHIP TASK IMPORTS
# ============================================================

from internship_tasks.task1_car_colour_detection.traffic_detection import (
    TrafficDetector,
)

from internship_tasks.task1_car_colour_detection.utils import (
    pil_to_cv2,
    cv2_to_rgb,
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Computer Vision Intelligence System",
    page_icon="🧠",
    layout="centered",
)


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
]


# ============================================================
# CACHED MODEL LOADERS
# ============================================================

@st.cache_resource
def load_uploaded_model(model_path, model_signature):
    """
    Load and cache the uploaded Keras emotion detection model.

    model_signature is used as part of the Streamlit cache key.
    """

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_face_detector():
    """Load and cache the MTCNN face detector."""

    return MTCNN()


@st.cache_resource
def load_traffic_detector():
    """Load and cache the Task 1 traffic detector."""

    return TrafficDetector()


# ============================================================
# EMOTION DETECTION FUNCTIONS
# ============================================================

def preprocess_image(image, target_size=(100, 100)):
    """
    Preprocess an image for the emotion detection CNN model.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:
        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    elif len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    image = cv2.resize(
        image,
        target_size,
    )

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(
        image,
        axis=0,
    )

    return image


def detect_face(image, detector):
    """
    Detect and crop the largest face from the input image.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:

        rgb_image = np.stack(
            [image, image, image],
            axis=-1,
        )

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

        if not results:
            return None

        largest_face = max(
            results,
            key=lambda face: (
                face["box"][2] * face["box"][3]
            ),
        )

        x, y, width, height = largest_face["box"]

        padding = int(
            max(width, height) * 0.2
        )

        x1 = max(
            0,
            x - padding,
        )

        y1 = max(
            0,
            y - padding,
        )

        x2 = min(
            rgb_image.shape[1],
            x + width + padding,
        )

        y2 = min(
            rgb_image.shape[0],
            y + height + padding,
        )

        face_image = rgb_image[
            y1:y2,
            x1:x2
        ]

        if face_image.size == 0:
            return None

        return face_image

    except Exception as error:

        st.error(
            f"Face detection error: {error}"
        )

        return None


def predict_emotion(model, processed_image):
    """
    Predict emotion using the loaded CNN model.
    """

    try:

        predictions = model.predict(
            processed_image,
            verbose=0,
        )

        predicted_class = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_class]
        )

        if predicted_class >= len(EMOTIONS):
            raise ValueError(
                "Model output classes do not match "
                "the configured emotion labels."
            )

        return (
            EMOTIONS[predicted_class],
            confidence,
            predictions[0],
        )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )

        return None, 0.0, None


# ============================================================
# SIDEBAR MODULE SELECTION
# ============================================================

st.sidebar.title(
    "🧠 Computer Vision System"
)

app_mode = st.sidebar.selectbox(
    "Select Computer Vision Module",
    [
        "Emotion Detection",
        "Traffic Car Colour Detection",
    ],
)


# ============================================================
# MODULE 1: EMOTION DETECTION
# ============================================================

if app_mode == "Emotion Detection":

    st.title(
        "🎭 Emotion Detection"
    )

    st.write(
        "Upload a trained Keras emotion detection model "
        "and an image to predict the visible facial emotion."
    )

    st.sidebar.header(
        "Model Configuration"
    )

    model_file = st.sidebar.file_uploader(
        "Upload .keras model",
        type=["keras"],
        key="emotion_model",
    )

    if model_file is not None:

        temp_model_path = os.path.join(
            tempfile.gettempdir(),
            "emotion_uploaded_model.keras",
        )

        with open(
            temp_model_path,
            "wb",
        ) as file:

            file.write(
                model_file.getbuffer()
            )

        try:

            model_signature = (
                model_file.name,
                model_file.size,
            )

            model = load_uploaded_model(
                temp_model_path,
                model_signature,
            )

            st.sidebar.success(
                "✅ Model loaded successfully!"
            )

            st.sidebar.info(
                f"Input shape: {model.input_shape}"
            )

            if (
                len(model.input_shape) == 4
                and model.input_shape[1] is not None
                and model.input_shape[2] is not None
            ):

                target_height = int(
                    model.input_shape[1]
                )

                target_width = int(
                    model.input_shape[2]
                )

                target_size = (
                    target_width,
                    target_height,
                )

            else:

                target_size = (
                    100,
                    100,
                )

            st.sidebar.info(
                f"Target size: {target_size}"
            )

            st.sidebar.subheader(
                "Settings"
            )

            use_face_detection = st.sidebar.checkbox(
                "Use face detection (Crop to face)",
                value=True,
            )

            st.header(
                "📤 Upload Image"
            )

            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "bmp",
                    "tiff",
                ],
                help=(
                    "Supports RGB and grayscale images."
                ),
                key="emotion_image",
            )

            if uploaded_file is not None:

                try:

                    image = Image.open(
                        uploaded_file
                    )

                    col1, col2 = st.columns(
                        [2, 1]
                    )

                    with col1:

                        st.image(
                            image,
                            caption="Input Image",
                            use_container_width=True,
                        )

                    with col2:

                        st.write(
                            "**Image Information**"
                        )

                        st.write(
                            f"Size: {image.size}"
                        )

                        st.write(
                            f"Mode: {image.mode}"
                        )

                        if image.format:

                            st.write(
                                f"Format: {image.format}"
                            )

                    if st.button(
                        "🔍 Predict Emotion",
                        type="primary",
                        use_container_width=True,
                    ):

                        with st.spinner(
                            "Processing emotion image..."
                        ):

                            processed_image = None

                            if use_face_detection:

                                face_detector = (
                                    load_face_detector()
                                )

                                face = detect_face(
                                    image,
                                    face_detector,
                                )

                                if face is not None:

                                    st.success(
                                        "✅ Face detected!"
                                    )

                                    st.image(
                                        face,
                                        caption=(
                                            "Detected Face Zone"
                                        ),
                                        width=150,
                                    )

                                    processed_image = (
                                        preprocess_image(
                                            face,
                                            target_size,
                                        )
                                    )

                                else:

                                    st.warning(
                                        "⚠️ No clear face detected. "
                                        "Using the full image."
                                    )

                            if processed_image is None:

                                processed_image = (
                                    preprocess_image(
                                        image,
                                        target_size,
                                    )
                                )

                            (
                                emotion,
                                confidence,
                                all_predictions,
                            ) = predict_emotion(
                                model,
                                processed_image,
                            )

                            if emotion is not None:

                                st.header(
                                    "🎯 Prediction Breakdown"
                                )

                                (
                                    metric_col1,
                                    metric_col2,
                                ) = st.columns(2)

                                metric_col1.metric(
                                    "Predicted Emotion",
                                    emotion,
                                )

                                metric_col2.metric(
                                    "Confidence Score",
                                    f"{confidence:.1%}",
                                )

                                sorted_emotions = sorted(
                                    zip(
                                        EMOTIONS,
                                        all_predictions,
                                    ),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )[:3]

                                st.subheader(
                                    "Top 3 Predictions"
                                )

                                for index, (
                                    emotion_name,
                                    probability,
                                ) in enumerate(
                                    sorted_emotions,
                                    start=1,
                                ):

                                    st.write(
                                        f"**{index}. "
                                        f"{emotion_name}**: "
                                        f"{probability:.2%}"
                                    )

                                st.subheader(
                                    "All Emotion Probabilities"
                                )

                                for (
                                    emotion_name,
                                    probability,
                                ) in zip(
                                    EMOTIONS,
                                    all_predictions,
                                ):

                                    st.progress(
                                        float(probability),
                                        text=(
                                            f"{emotion_name}: "
                                            f"{probability:.2%}"
                                        ),
                                    )

                            else:

                                st.error(
                                    "❌ Emotion prediction failed."
                                )

                except Exception as error:

                    st.error(
                        f"❌ Image processing failed: {error}"
                    )

        except Exception as error:

            st.sidebar.error(
                f"❌ Model loading failed: {error}"
            )

            st.error(
                "Please upload a valid .keras model file."
            )

    else:

        st.sidebar.warning(
            "Please upload a .keras model file."
        )

        st.info(
            "👆 Upload your trained emotion detection "
            "model from the sidebar to begin."
        )


# ============================================================
# MODULE 2: INTERNSHIP TASK 1
# TRAFFIC CAR COLOUR DETECTION
# ============================================================

elif app_mode == "Traffic Car Colour Detection":

    st.title(
        "🚗 Traffic Car Colour Detection"
    )

    st.write(
        "This internship enhancement detects cars in a "
        "traffic image, identifies blue cars, counts cars, "
        "and reports the number of detected people."
    )

    st.info(
        "🔴 Red Rectangle = Blue Car | "
        "🔵 Blue Rectangle = Other Colour Car"
    )

    uploaded_file = st.file_uploader(
        "Upload Traffic Image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="traffic_image",
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            # ------------------------------------------------
            # INPUT IMAGE PREVIEW
            # ------------------------------------------------

            st.subheader(
                "📷 Input Image Preview"
            )

            st.image(
                image,
                caption="Traffic Input Image",
                use_container_width=True,
            )

            st.write(
                f"Image size: {image.size}"
            )

            if st.button(
                "🔍 Analyse Traffic",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Detecting cars and people..."
                ):

                    try:

                        cv_image = pil_to_cv2(
                            image
                        )

                        traffic_detector = (
                            load_traffic_detector()
                        )

                        (
                            result_image,
                            statistics,
                        ) = traffic_detector.detect(
                            cv_image
                        )

                        # ------------------------------------
                        # DETECTION RESULT
                        # ------------------------------------

                        st.subheader(
                            "🎯 Detection Result"
                        )

                        st.image(
                            cv2_to_rgb(
                                result_image
                            ),
                            caption=(
                                "Traffic Detection Result"
                            ),
                            use_container_width=True,
                        )

                        # ------------------------------------
                        # TRAFFIC STATISTICS
                        # ------------------------------------

                        st.subheader(
                            "📊 Traffic Statistics"
                        )

                        (
                            col1,
                            col2,
                            col3,
                            col4,
                        ) = st.columns(4)

                        col1.metric(
                            "Total Cars",
                            statistics["cars"],
                        )

                        col2.metric(
                            "Blue Cars",
                            statistics["blue_cars"],
                        )

                        col3.metric(
                            "Other Cars",
                            statistics["other_cars"],
                        )

                        col4.metric(
                            "People",
                            statistics["people"],
                        )

                        st.success(
                            "✅ Traffic analysis completed "
                            "successfully!"
                        )

                        st.info(
                            "🔴 Red Rectangle = Blue Car | "
                            "🔵 Blue Rectangle = Other Colour Car"
                        )

                    except Exception as error:

                        st.error(
                            "❌ Traffic detection failed: "
                            f"{error}"
                        )

        except Exception as error:

            st.error(
                f"❌ Unable to open image: {error}"
            )

    else:

        st.info(
            "👆 Upload a traffic image to begin analysis."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
        <b>
            Computer Vision Intelligence System |
            Emotion Detection + Traffic Analysis |
            Powered by Streamlit
        </b>
    </center>
    """,
    unsafe_allow_html=True,
)