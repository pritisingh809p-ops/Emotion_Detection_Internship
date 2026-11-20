"""Unit tests for facial emotion detection module using mocked DeepFace inference."""

from unittest.mock import MagicMock, patch
import numpy as np
from modules.emotion_detection import detect_emotion


def test_detect_emotion_success():
    dummy_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_deepface = MagicMock()
    mock_deepface.analyze.return_value = [
        {
            "dominant_emotion": "happy",
            "emotion": {
                "happy": 88.45,
                "neutral": 8.10,
                "sad": 1.20,
                "angry": 0.50,
                "surprise": 1.00,
                "fear": 0.50,
                "disgust": 0.25,
            },
        }
    ]

    with patch("modules.emotion_detection._get_deepface", return_value=mock_deepface):
        result = detect_emotion(dummy_bgr)

    assert result["success"] is True
    assert result["dominant_emotion"] == "Happy"
    assert result["confidence"] == 88.45
    assert "Happy" in result["all_emotions"]
    assert result["error"] is None


def test_detect_emotion_no_face_detected():
    dummy_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_deepface = MagicMock()
    mock_deepface.analyze.side_effect = ValueError("Face could not be detected in numpy array.")

    with patch("modules.emotion_detection._get_deepface", return_value=mock_deepface):
        result = detect_emotion(dummy_bgr)

    assert result["success"] is False
    assert result["dominant_emotion"] is None
    assert "No clear face was detected" in result["error"]


def test_detect_emotion_invalid_image():
    result = detect_emotion(None)
    assert result["success"] is False
    assert "Invalid or empty image" in result["error"]


def test_detect_emotion_import_error_handled():
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("modules.emotion_detection._get_deepface", side_effect=ImportError("No module named deepface")):
        result = detect_emotion(dummy_bgr)

    assert result["success"] is False
    assert "DeepFace library is not installed" in result["error"]
