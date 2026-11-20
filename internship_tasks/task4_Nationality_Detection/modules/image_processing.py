"""
Image Validation and Preprocessing Module.

Handles uploaded image validation (format, size, readability, dimensions)
and safe conversion between PIL Images and OpenCV BGR/RGB NumPy arrays.
"""

from typing import Tuple, Optional, Any
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 4096


class ImageValidationError(Exception):
    """Raised when an uploaded image fails format, size, or integrity checks."""
    pass


def validate_uploaded_file(uploaded_file: Any) -> Tuple[Image.Image, np.ndarray]:
    """
    Validate an uploaded image file object (e.g., Streamlit UploadedFile or BytesIO)
    and return both a PIL RGB Image and an OpenCV BGR NumPy array.

    Args:
        uploaded_file: File-like object with `.read()`, `.seek()`, and optional `.name` / `.size`.

    Returns:
        Tuple[Image.Image, np.ndarray]: (PIL RGB Image, OpenCV BGR uint8 array)

    Raises:
        ImageValidationError: If file is missing, too large, wrong extension, or corrupted.
    """
    if uploaded_file is None:
        raise ImageValidationError("No image file provided. Please upload a JPG, JPEG, or PNG image.")

    # Check extension if filename attribute is available
    file_name = getattr(uploaded_file, "name", "")
    if file_name and "." in file_name:
        ext = file_name.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file format '.{ext}'. Allowed formats: JPG, JPEG, PNG."
            )

    # Read bytes safely
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

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"File size ({size_mb:.2f} MB) exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."
        )

    # Decode with Pillow
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()  # Verify file integrity
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            "Invalid or corrupted image file. Please upload a valid JPG, JPEG, or PNG image."
        ) from exc

    width, height = pil_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image resolution ({width}x{height}) is too small. Minimum required is "
            f"{MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
        )

    bgr_image = pil_to_bgr(pil_img)
    return pil_img, bgr_image


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image to an OpenCV BGR NumPy array."""
    rgb_array = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR NumPy array to an RGB NumPy array."""
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def resize_for_inference(
    bgr_image: np.ndarray, max_dim: int = 1024
) -> np.ndarray:
    """
    Resize an image proportionally if either dimension exceeds `max_dim`,
    improving inference speed and memory stability while preserving aspect ratio.

    Args:
        bgr_image: OpenCV BGR image array.
        max_dim: Maximum allowed width or height in pixels.

    Returns:
        np.ndarray: Resized (or original if already smaller) BGR image.
    """
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Cannot resize an empty image.")

    height, width = bgr_image.shape[:2]
    if max(height, width) <= max_dim:
        return bgr_image.copy()

    scale = max_dim / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def detect_face_region(bgr_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest frontal face bounding box using OpenCV Haar Cascade.
    Used as a fast helper for clothing ROI alignment and face presence checks.

    Args:
        bgr_image: OpenCV BGR image array.

    Returns:
        Optional[Tuple[int, int, int, int]]: (x, y, w, h) of largest detected face, or None.
    """
    if bgr_image is None or bgr_image.size == 0:
        return None

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    # Return the largest face by area (w * h)
    largest_face = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
    return tuple(int(v) for v in largest_face)
