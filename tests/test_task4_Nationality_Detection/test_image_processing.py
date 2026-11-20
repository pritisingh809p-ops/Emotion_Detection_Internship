"""Unit tests for image validation and preprocessing utilities."""

import io
import numpy as np
import pytest
from PIL import Image
from modules.image_processing import (
    validate_uploaded_file,
    resize_for_inference,
    pil_to_bgr,
    bgr_to_rgb,
    ImageValidationError,
)


def _create_sample_image_bytes(width: int = 200, height: int = 200, fmt: str = "JPEG") -> io.BytesIO:
    """Helper to generate an in-memory image file buffer with a `.name` attribute."""
    img = Image.new("RGB", (width, height), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    ext = "jpg" if fmt.upper() == "JPEG" else fmt.lower()
    buf.name = f"sample_person.{ext}"
    return buf


def test_validate_valid_jpeg_image():
    buf = _create_sample_image_bytes(256, 256, "JPEG")
    pil_img, bgr_img = validate_uploaded_file(buf)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (256, 256)
    assert isinstance(bgr_img, np.ndarray)
    assert bgr_img.shape == (256, 256, 3)


def test_validate_valid_png_image():
    buf = _create_sample_image_bytes(180, 220, "PNG")
    pil_img, bgr_img = validate_uploaded_file(buf)
    assert pil_img.size == (180, 220)
    assert bgr_img.shape == (220, 180, 3)


def test_validate_none_file_raises():
    with pytest.raises(ImageValidationError, match="No image file provided"):
        validate_uploaded_file(None)


def test_validate_unsupported_extension_raises():
    buf = io.BytesIO(b"fake data")
    buf.name = "document.gif"
    with pytest.raises(ImageValidationError, match="Unsupported file format"):
        validate_uploaded_file(buf)


def test_validate_corrupted_bytes_raises():
    buf = io.BytesIO(b"not_a_real_image_header_12345")
    buf.name = "broken.jpg"
    with pytest.raises(ImageValidationError, match="Invalid or corrupted"):
        validate_uploaded_file(buf)


def test_validate_too_small_image_raises():
    buf = _create_sample_image_bytes(32, 32, "PNG")
    with pytest.raises(ImageValidationError, match="too small"):
        validate_uploaded_file(buf)


def test_resize_for_inference_scales_down_large_image():
    large_bgr = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized = resize_for_inference(large_bgr, max_dim=1000)
    assert resized.shape == (1000, 500, 3)


def test_resize_for_inference_keeps_small_image():
    small_bgr = np.zeros((400, 300, 3), dtype=np.uint8)
    resized = resize_for_inference(small_bgr, max_dim=1000)
    assert resized.shape == (400, 300, 3)


def test_color_space_roundtrip():
    pil_img = Image.new("RGB", (50, 50), color=(10, 80, 220))
    bgr = pil_to_bgr(pil_img)
    rgb = bgr_to_rgb(bgr)
    assert tuple(rgb[0, 0]) == (10, 80, 220)
