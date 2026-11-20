"""Vehicle/person detection with YOLO and a lightweight HSV car-colour estimator."""
from functools import lru_cache
import cv2
import numpy as np

@lru_cache(maxsize=1)
def load_model():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    # Ultralytics downloads the small COCO pretrained weights on first use.
    return YOLO("yolov8n.pt")

def estimate_car_colour(crop):
    """Estimate a dominant practical colour label from an RGB or BGR crop."""
    if crop is None or crop.size == 0:
        return "Unknown"
    # OpenCV HSV conversion expects BGR.
    if crop.ndim == 3 and crop.shape[2] == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    else:
        return "Unknown"
    h, s, v = cv2.split(hsv)
    valid = (v > 35)
    if not np.any(valid):
        return "Black"
    hue = h[valid]
    sat = s[valid]
    val = v[valid]
    sat_median = float(np.median(sat))
    val_median = float(np.median(val))
    if val_median < 65:
        return "Black"
    if sat_median < 35:
        return "White" if val_median > 190 else "Grey"
    # OpenCV hue range is 0..179.
    hue_median = float(np.median(hue))
    if 90 <= hue_median <= 135:
        return "Blue"
    if hue_median < 10 or hue_median >= 170:
        return "Red"
    if 10 <= hue_median < 25:
        return "Orange"
    if 25 <= hue_median < 38:
        return "Yellow"
    if 38 <= hue_median < 85:
        return "Green"
    if 135 < hue_median < 170:
        return "Purple"
    return "Other"

def detect_cars_and_people(image_rgb, confidence=0.35):
    """Return car/person detections and summary counts for an RGB image."""
    model = load_model()
    height, width = image_rgb.shape[:2]
    results = model.predict(source=image_rgb, conf=float(confidence), verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in (0, 2):  # COCO: person=0, car=2
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            score = float(box.conf.item())
            if class_id == 2:
                colour = estimate_car_colour(image_rgb[y1:y2, x1:x2])
                detections.append({"label": "Car", "colour": colour,
                                   "confidence": score, "box": (x1, y1, x2, y2)})
            else:
                detections.append({"label": "Person", "confidence": score,
                                   "box": (x1, y1, x2, y2)})
    cars = [d for d in detections if d["label"] == "Car"]
    people = [d for d in detections if d["label"] == "Person"]
    counts = {"cars": len(cars), "people": len(people),
              "blue_cars": sum(d.get("colour") == "Blue" for d in cars)}
    return detections, counts

def draw_detections(image_rgb, detections):
    """Draw blue cars with red rectangles; all other cars/persons with blue rectangles."""
    output = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_blue_car = det["label"] == "Car" and det.get("colour") == "Blue"
        colour = (255, 0, 0) if is_blue_car else (0, 90, 255)  # RGB red / blue
        label = det["label"]
        if det["label"] == "Car":
            label += f' - {det.get("colour", "Unknown")}'
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(output, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return output
