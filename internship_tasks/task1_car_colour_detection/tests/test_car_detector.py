import numpy as np
from car_detector import estimate_car_colour, draw_detections

def test_estimate_blue_colour():
    # RGB blue image crop
    crop = np.zeros((30, 30, 3), dtype=np.uint8)
    crop[:, :] = (0, 0, 255)
    assert estimate_car_colour(crop) == "Blue"

def test_estimate_red_colour():
    crop = np.zeros((30, 30, 3), dtype=np.uint8)
    crop[:, :] = (255, 0, 0)
    assert estimate_car_colour(crop) == "Red"

def test_draw_returns_same_image_dimensions():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [{"label": "Car", "colour": "Blue", "confidence": 0.9,
                   "box": (10, 10, 60, 60)}]
    result = draw_detections(image, detections)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
