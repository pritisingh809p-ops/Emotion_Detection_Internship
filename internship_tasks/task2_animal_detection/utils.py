import cv2
import numpy as np
from PIL import Image
import tempfile
import os


def load_image(uploaded_file):
    """
    Convert uploaded Streamlit image to OpenCV format.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def convert_to_rgb(image):
    """
    Convert OpenCV image to RGB for Streamlit display.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_uploaded_video(uploaded_file):
    """
    Save uploaded video temporarily and return its path.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name


def delete_temp_file(path):
    """
    Delete temporary file if it exists.
    """
    if os.path.exists(path):
        os.remove(path)