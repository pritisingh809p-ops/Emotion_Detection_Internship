"""
Apparent Age Estimation Module.

Uses DeepFace pretrained age regression model to estimate apparent age in years.
Note: Outputs are strictly framed as approximate estimates, never exact biological age.
"""

from typing import Dict, Any
import numpy as np


def _get_deepface():
    """Lazy import of DeepFace to keep module import fast and testable."""
    from deepface import DeepFace
    return DeepFace


def estimate_age(
    bgr_image: np.ndarray,
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
) -> Dict[str, Any]:
    """
    Estimate apparent age in years from a facial image using DeepFace.

    Args:
        bgr_image: OpenCV BGR image array.
        detector_backend: Face detector backend ('opencv', 'retinaface', etc.).
        enforce_detection: Raise error if no face is detected when True.

    Returns:
        Dict[str, Any]: Standardized result dictionary with keys:
            - success (bool)
            - estimated_age (int or None): Estimated age in years
            - label (str or None): Clearly labeled estimate string
            - error (str or None): User-friendly error message if failed
    """
    if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return {
            "success": False,
            "estimated_age": None,
            "label": None,
            "error": "Invalid or empty image provided for age estimation.",
        }

    try:
        DeepFace = _get_deepface()
    except ImportError as exc:
        return {
            "success": False,
            "estimated_age": None,
            "label": None,
            "error": (
                "DeepFace library is not installed or failed to import. "
                f"Please verify dependencies ({exc})."
            ),
        }

    try:
        analysis = DeepFace.analyze(
            img_path=bgr_image,
            actions=["age"],
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            silent=True,
        )

        if isinstance(analysis, list):
            if len(analysis) == 0:
                return {
                    "success": False,
                    "estimated_age": None,
                    "label": None,
                    "error": "No human face could be detected for age estimation.",
                }
            face_result = analysis[0]
        elif isinstance(analysis, dict):
            face_result = analysis
        else:
            return {
                "success": False,
                "estimated_age": None,
                "label": None,
                "error": "Unexpected output format from age estimation model.",
            }

        raw_age = face_result.get("age")
        if raw_age is None:
            return {
                "success": False,
                "estimated_age": None,
                "label": None,
                "error": "Age prediction could not be extracted from the detected face.",
            }

        estimated_age = int(round(float(raw_age)))
        if estimated_age < 0 or estimated_age > 120:
            return {
                "success": False,
                "estimated_age": None,
                "label": None,
                "error": "Model returned an out-of-range age estimate.",
            }

        return {
            "success": True,
            "estimated_age": estimated_age,
            "label": f"~{estimated_age} years (Estimated, not exact)",
            "error": None,
        }

    except ValueError as exc:
        msg = str(exc).lower()
        if "face could not be detected" in msg or "no face" in msg:
            return {
                "success": False,
                "estimated_age": None,
                "label": None,
                "error": (
                    "No clear face was detected in the image. Please upload a well-lit, "
                    "front-facing photo of a person."
                ),
            }
        return {
            "success": False,
            "estimated_age": None,
            "label": None,
            "error": f"Age estimation could not process the image: {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "estimated_age": None,
            "label": None,
            "error": (
                "An error occurred while running the age estimation model "
                f"(check internet connection if downloading weights for the first time): {exc}"
            ),
        }
