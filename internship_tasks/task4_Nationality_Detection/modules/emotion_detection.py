"""
Facial Emotion Detection Module.

Uses DeepFace pretrained emotion recognition models to detect the dominant
facial emotion and confidence score from an input image.
"""

from typing import Dict, Any
import numpy as np


def _get_deepface():
    """Lazy import of DeepFace to keep module import fast and testable."""
    from deepface import DeepFace
    return DeepFace


def detect_emotion(
    bgr_image: np.ndarray,
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
) -> Dict[str, Any]:
    """
    Detect the dominant facial emotion in an image using DeepFace.

    Args:
        bgr_image: OpenCV BGR image array.
        detector_backend: Face detector backend ('opencv', 'retinaface', 'mtcnn', etc.).
        enforce_detection: Raise error if no face is detected when True.

    Returns:
        Dict[str, Any]: Standardized result dictionary with keys:
            - success (bool)
            - dominant_emotion (str or None)
            - confidence (float or None): Percentage confidence (0-100) if available
            - all_emotions (dict): Scores across emotion classes
            - error (str or None): User-friendly error message if failed
    """
    if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return {
            "success": False,
            "dominant_emotion": None,
            "confidence": None,
            "all_emotions": {},
            "error": "Invalid or empty image provided for emotion analysis.",
        }

    try:
        DeepFace = _get_deepface()
    except ImportError as exc:
        return {
            "success": False,
            "dominant_emotion": None,
            "confidence": None,
            "all_emotions": {},
            "error": (
                "DeepFace library is not installed or failed to import. "
                f"Please verify dependencies ({exc})."
            ),
        }

    try:
        analysis = DeepFace.analyze(
            img_path=bgr_image,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            silent=True,
        )

        # DeepFace >= 0.0.79 returns a list of face dicts; older versions return a dict
        if isinstance(analysis, list):
            if len(analysis) == 0:
                return {
                    "success": False,
                    "dominant_emotion": None,
                    "confidence": None,
                    "all_emotions": {},
                    "error": "No human face could be detected in the uploaded image.",
                }
            face_result = analysis[0]
        elif isinstance(analysis, dict):
            face_result = analysis
        else:
            return {
                "success": False,
                "dominant_emotion": None,
                "confidence": None,
                "all_emotions": {},
                "error": "Unexpected output format from emotion analysis model.",
            }

        dominant_emotion = face_result.get("dominant_emotion")
        emotion_scores = face_result.get("emotion", {})

        if not dominant_emotion:
            return {
                "success": False,
                "dominant_emotion": None,
                "confidence": None,
                "all_emotions": {},
                "error": "Emotion prediction could not be determined from the detected face.",
            }

        confidence = None
        if isinstance(emotion_scores, dict) and dominant_emotion in emotion_scores:
            raw_conf = float(emotion_scores[dominant_emotion])
            # Normalize if returned as 0..1 instead of 0..100
            if 0.0 <= raw_conf <= 1.0 and max(emotion_scores.values()) <= 1.0:
                raw_conf *= 100.0
            confidence = round(raw_conf, 2)

        cleaned_scores = {
            str(k).capitalize(): round(float(v), 2)
            for k, v in (emotion_scores.items() if isinstance(emotion_scores, dict) else [])
        }

        return {
            "success": True,
            "dominant_emotion": str(dominant_emotion).capitalize(),
            "confidence": confidence,
            "all_emotions": cleaned_scores,
            "error": None,
        }

    except ValueError as exc:
        msg = str(exc).lower()
        if "face could not be detected" in msg or "no face" in msg:
            return {
                "success": False,
                "dominant_emotion": None,
                "confidence": None,
                "all_emotions": {},
                "error": (
                    "No clear face was detected in the image. Please upload a well-lit, "
                    "front-facing photo of a person."
                ),
            }
        return {
            "success": False,
            "dominant_emotion": None,
            "confidence": None,
            "all_emotions": {},
            "error": f"Emotion analysis could not process the image: {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "dominant_emotion": None,
            "confidence": None,
            "all_emotions": {},
            "error": (
                "An error occurred while running the emotion detection model "
                f"(check internet connection if downloading weights for the first time): {exc}"
            ),
        }
