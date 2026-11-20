"""
Image Validation and Preprocessing Module for Sign Language Images.
"""

from typing import Tuple, Any
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_DIMENSION = 64


class ImageValidationError(Exception):
    """Raised when an uploaded sign image fails format, size, or integrity checks."""
    pass


def validate_uploaded_file(uploaded_file: Any) -> Tuple[Image.Image, np.ndarray]:
    """Validate uploaded sign image and return PIL RGB image and OpenCV BGR array."""
    if uploaded_file is None:
        raise ImageValidationError("No image file provided. Please upload a JPG, JPEG, or PNG image.")

    file_name = getattr(uploaded_file, "name", "")
    if file_name and "." in file_name:
        ext = file_name.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(f"Unsupported file format '.{ext}'. Allowed: JPG, JPEG, PNG.")

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    except Exception as exc:
        raise ImageValidationError(f"Unable to read uploaded file: {exc}") from exc

    if not raw_bytes:
        raise ImageValidationError("The uploaded file is empty (0 bytes).")

    if (len(raw_bytes) / (1024 * 1024)) > MAX_FILE_SIZE_MB:
        raise ImageValidationError(f"File size exceeds limit of {MAX_FILE_SIZE_MB} MB.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("Invalid or corrupted image file.") from exc

    width, height = pil_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(f"Image resolution ({width}x{height}) is too small.")

    bgr_image = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return pil_img, bgr_image
