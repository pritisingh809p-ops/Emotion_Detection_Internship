"""Unit tests for apparent age estimation module using mocked DeepFace inference."""

from unittest.mock import MagicMock, patch
import numpy as np
from modules.age_detection import estimate_age


def test_estimate_age_success():
    dummy_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_deepface = MagicMock()
    mock_deepface.analyze.return_value = [{"age": 27.6}]

    with patch("modules.age_detection._get_deepface", return_value=mock_deepface):
        result = estimate_age(dummy_bgr)

    assert result["success"] is True
    assert result["estimated_age"] == 28
    assert "Estimated, not exact" in result["label"]
    assert result["error"] is None


def test_estimate_age_no_face_detected():
    dummy_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_deepface = MagicMock()
    mock_deepface.analyze.side_effect = ValueError("Face could not be detected")

    with patch("modules.age_detection._get_deepface", return_value=mock_deepface):
        result = estimate_age(dummy_bgr)

    assert result["success"] is False
    assert result["estimated_age"] is None
    assert "No clear face was detected" in result["error"]


def test_estimate_age_empty_image():
    result = estimate_age(np.array([]))
    assert result["success"] is False
    assert "Invalid or empty image" in result["error"]
