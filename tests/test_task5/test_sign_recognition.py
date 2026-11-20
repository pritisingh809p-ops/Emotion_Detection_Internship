"""Unit tests for sign language recognition module."""
import numpy as np
from modules.sign_recognition import recognize_sign


def test_recognize_sign_empty_image():
    res = recognize_sign(None)
    assert res["success"] is False
    assert "Invalid or empty image" in res["error"]
