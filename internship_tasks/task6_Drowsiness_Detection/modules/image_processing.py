"""
Image and Video Validation Module for Drowsiness Detection.
"""

from typing import Tuple, Any
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv"}
MAX_FILE_SIZE_MB = 50


class MediaValidationError(Exception):
    """Raised when uploaded file fails validation."""
    pass


def validate_uploaded_image(uploaded_file: Any) -> Tuple[Image.Image, np.ndarray]:
    """Validate uploaded image file."""
    if uploaded_file is None:
        raise MediaValidationError("No image file provided.")
    
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    except Exception as exc:
        raise MediaValidationError(f"Unable to read file: {exc}") from exc

    if not raw_bytes:
        raise MediaValidationError("Uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaValidationError("Invalid or corrupted image file.") from exc

    bgr_img = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return pil_img, bgr_img
