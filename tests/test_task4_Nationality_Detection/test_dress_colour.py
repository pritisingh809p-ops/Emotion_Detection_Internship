"""Unit tests for dress colour segmentation and classification module."""

from unittest.mock import patch
import numpy as np
import cv2
from modules.dress_colour import detect_dress_colour, rgb_to_colour_name


def test_rgb_to_colour_name_mapping():
    assert rgb_to_colour_name((205, 25, 25)) == "Red"
    assert rgb_to_colour_name((20, 45, 110)) == "Navy Blue"
    assert rgb_to_colour_name((242, 242, 242)) == "White"


def test_detect_dress_colour_with_synthetic_person_and_blue_shirt():
    # Create a 300x300 synthetic image: white background, face box at top-center, solid blue shirt below
    img_rgb = np.full((300, 300, 3), 245, dtype=np.uint8)
    # Paint a bright blue torso rectangle (x: 80..220, y: 120..280)
    img_rgb[120:280, 80:220] = (30, 90, 210)
    bgr_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Create a binary person mask covering the central upper body
    person_mask = np.zeros((300, 300), dtype=np.uint8)
    person_mask[40:285, 80:220] = 255

    with patch("modules.dress_colour.detect_face_region", return_value=(115, 35, 70, 70)),          patch("modules.dress_colour._segment_person_mediapipe", return_value=person_mask):
        result = detect_dress_colour(bgr_img)

    assert result["success"] is True
    assert result["colour_name"] == "Blue"
    assert result["hex_code"].startswith("#")
    assert result["warning"] is None


def test_detect_dress_colour_warns_on_tight_headshot():
    bgr_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Face extends down to 92% of image height -> no clothing visible
    with patch("modules.dress_colour.detect_face_region", return_value=(20, 10, 160, 175)):
        result = detect_dress_colour(bgr_img)

    assert result["success"] is False
    assert result["colour_name"] is None
    assert "Clothing is not clearly visible" in result["warning"]


def test_detect_dress_colour_warns_when_segmentation_fails():
    bgr_img = np.zeros((200, 200, 3), dtype=np.uint8)
    with patch("modules.dress_colour.detect_face_region", return_value=None),          patch("modules.dress_colour._segment_person_mediapipe", return_value=None),          patch("modules.dress_colour._segment_person_grabcut", return_value=None):
        result = detect_dress_colour(bgr_img)

    assert result["success"] is False
    assert "Unable to segment" in result["warning"]
