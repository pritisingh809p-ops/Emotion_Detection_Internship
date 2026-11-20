"""
Clothing Region Segmentation and Dominant Dress Colour Estimation Module.

Uses MediaPipe Selfie Segmentation (with graceful fallback to face-guided torso
ROI + OpenCV GrabCut foreground segmentation if MediaPipe weights/API are unavailable)
to isolate the upper-body clothing region rather than sampling background pixels.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from internship_tasks.task4_Nationality_Detection.modules.image_processing import detect_face_region


# Standard named reference colours in RGB for human-readable colour mapping
REFERENCE_COLOURS_RGB = {
    "Black": (20, 20, 20),
    "White": (240, 240, 240),
    "Grey": (128, 128, 128),
    "Light Grey": (195, 195, 195),
    "Charcoal": (54, 69, 79),
    "Red": (200, 30, 30),
    "Maroon": (128, 0, 32),
    "Pink": (235, 130, 165),
    "Orange": (230, 115, 25),
    "Yellow": (235, 205, 35),
    "Mustard": (205, 165, 35),
    "Green": (35, 145, 55),
    "Olive": (105, 120, 45),
    "Teal": (20, 130, 130),
    "Blue": (35, 95, 205),
    "Navy Blue": (25, 45, 105),
    "Sky Blue": (115, 185, 235),
    "Purple": (125, 50, 165),
    "Brown": (125, 75, 40),
    "Beige / Cream": (220, 200, 165),
}


def rgb_to_colour_name(rgb_tuple: Tuple[int, int, int]) -> str:
    """
    Map an RGB colour tuple to the closest human-readable colour name
    using Euclidean distance in perceptual LAB colour space.
    """
    rgb_arr = np.uint8([[list(rgb_tuple)]])
    lab_target = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)

    best_name = "Unknown"
    min_dist = float("inf")

    for name, ref_rgb in REFERENCE_COLOURS_RGB.items():
        ref_arr = np.uint8([[list(ref_rgb)]])
        ref_lab = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
        dist = float(np.linalg.norm(lab_target - ref_lab))
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name


def _segment_person_mediapipe(bgr_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt person foreground segmentation using MediaPipe SelfieSegmentation.
    Returns a binary uint8 mask (255 = person, 0 = background) or None if unavailable.
    """
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "selfie_segmentation"):
            return None

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            results = segmenter.process(rgb_image)
            if results.segmentation_mask is None:
                return None
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            return mask
    except Exception:
        return None


def _segment_person_grabcut(
    bgr_image: np.ndarray, face_box: Optional[Tuple[int, int, int, int]]
) -> Optional[np.ndarray]:
    """
    Fallback person segmentation using OpenCV GrabCut initialized with a
    torso-aware foreground rectangle derived from the detected face box.
    """
    h, w = bgr_image.shape[:2]
    if face_box is None:
        return None

    fx, fy, fw, fh = face_box
    # Define expected upper-body bounding rectangle below the face
    rect_x = max(1, int(fx - fw * 0.9))
    rect_y = max(1, int(fy + fh * 0.85))
    rect_w = min(w - rect_x - 1, int(fw * 2.8))
    rect_h = min(h - rect_y - 1, int(fh * 3.2))

    if rect_w < 20 or rect_h < 20:
        return None

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr_image,
            mask,
            (rect_x, rect_y, rect_w, rect_h),
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
            np.uint8
        )
        return fg_mask
    except Exception:
        return None


def _extract_torso_clothing_mask(
    bgr_image: np.ndarray,
    person_mask: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Restrict the segmented person mask specifically to the torso/clothing area
    below the chin/neck and filter out exposed skin pixels in YCrCb colour space.
    """
    h, w = bgr_image.shape[:2]
    torso_roi = np.zeros((h, w), dtype=np.uint8)

    if face_box is not None:
        fx, fy, fw, fh = face_box
        # Start below chin (fy + 1.15 * fh) down to bottom of frame or 3.8 * fh below chin
        top_y = int(fy + fh * 1.15)
        bottom_y = min(h, int(fy + fh * 4.5))
        left_x = max(0, int(fx - fw * 1.1))
        right_x = min(w, int(fx + fw * 2.1))

        if top_y >= h - 15 or bottom_y - top_y < 15 or right_x - left_x < 15:
            # Clothing is cropped out (e.g., tight face-only headshot)
            return np.zeros((h, w), dtype=np.uint8)

        torso_roi[top_y:bottom_y, left_x:right_x] = 255
    else:
        # Without a face reference, inspect central lower-torso region (55% to 92% height)
        top_y = int(h * 0.55)
        bottom_y = int(h * 0.92)
        left_x = int(w * 0.25)
        right_x = int(w * 0.75)
        torso_roi[top_y:bottom_y, left_x:right_x] = 255

    clothing_mask = cv2.bitwise_and(person_mask, torso_roi)

    # Filter out skin tones using standard YCrCb skin-range thresholding so neck/hands don't skew colour
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    non_skin_mask = cv2.bitwise_not(skin_mask)

    final_mask = cv2.bitwise_and(clothing_mask, non_skin_mask)
    return final_mask


def _dominant_colour_kmeans(rgb_pixels: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
    """
    Find the dominant RGB colour among masked clothing pixels using OpenCV K-Means clustering.
    """
    n_pixels = len(rgb_pixels)
    clusters = min(k, n_pixels)
    data = np.float32(rgb_pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant_cluster = int(np.argmax(counts))
    dominant_rgb = centers[dominant_cluster]
    return (
        int(np.clip(round(dominant_rgb[0]), 0, 255)),
        int(np.clip(round(dominant_rgb[1]), 0, 255)),
        int(np.clip(round(dominant_rgb[2]), 0, 255)),
    )


def detect_dress_colour(bgr_image: np.ndarray) -> Dict[str, Any]:
    """
    Detect the dominant clothing (dress/shirt/top) colour from an image of a person.

    Pipeline:
    1. Locate face bounding box to anchor the upper-body clothing zone.
    2. Segment the person foreground via MediaPipe Selfie Segmentation (or GrabCut fallback).
    3. Crop to the torso clothing ROI and exclude skin pixels.
    4. Validate that sufficient clothing pixels exist; warn if clothing is not visible.
    5. Run K-Means clustering to extract dominant RGB and map to a human-readable colour name.

    Args:
        bgr_image: OpenCV BGR image array.

    Returns:
        Dict[str, Any]: Standardized result dictionary with keys:
            - success (bool)
            - colour_name (str or None)
            - rgb (Tuple[int, int, int] or None)
            - hex_code (str or None)
            - segmentation_method (str or None)
            - warning (str or None): Warning message if clothing cannot be detected reliably
    """
    if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return {
            "success": False,
            "colour_name": None,
            "rgb": None,
            "hex_code": None,
            "segmentation_method": None,
            "warning": "Invalid or empty image provided for dress colour detection.",
        }

    h, w = bgr_image.shape[:2]
    total_pixels = h * w
    face_box = detect_face_region(bgr_image)

    # Check if face fills almost the entire frame (close-up headshot with no visible clothing)
    if face_box is not None:
        _, fy, _, fh = face_box
        if (fy + fh) > h * 0.88:
            return {
                "success": False,
                "colour_name": None,
                "rgb": None,
                "hex_code": None,
                "segmentation_method": None,
                "warning": (
                    "Clothing is not clearly visible in this close-up photo. "
                    "Please upload an image showing the person's upper-body outfit."
                ),
            }

    segmentation_method = "MediaPipe Selfie Segmentation"
    person_mask = _segment_person_mediapipe(bgr_image)

    if person_mask is None:
        segmentation_method = "OpenCV GrabCut Torso Segmentation"
        person_mask = _segment_person_grabcut(bgr_image, face_box)

    if person_mask is None:
        return {
            "success": False,
            "colour_name": None,
            "rgb": None,
            "hex_code": None,
            "segmentation_method": None,
            "warning": (
                "Unable to segment the person or clothing region from the background. "
                "Please upload a clearer photo showing the person's upper-body clothing."
            ),
        }

    clothing_mask = _extract_torso_clothing_mask(bgr_image, person_mask, face_box)
    clothing_pixel_count = int(np.count_nonzero(clothing_mask))

    # Require at least 1.5% of image area (and at least 250 pixels) to be valid clothing
    min_required_pixels = max(250, int(total_pixels * 0.015))
    if clothing_pixel_count < min_required_pixels:
        return {
            "success": False,
            "colour_name": None,
            "rgb": None,
            "hex_code": None,
            "segmentation_method": segmentation_method,
            "warning": (
                "Clothing region is too small or obstructed to reliably detect dress colour. "
                "Please use an image where the person's outfit is clearly visible."
            ),
        }

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    clothing_pixels = rgb_image[clothing_mask > 0]

    dominant_rgb = _dominant_colour_kmeans(clothing_pixels, k=3)
    colour_name = rgb_to_colour_name(dominant_rgb)
    hex_code = "#{:02X}{:02X}{:02X}".format(*dominant_rgb)

    return {
        "success": True,
        "colour_name": colour_name,
        "rgb": dominant_rgb,
        "hex_code": hex_code,
        "segmentation_method": segmentation_method,
        "warning": None,
    }
