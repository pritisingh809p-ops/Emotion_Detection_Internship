from pathlib import Path
from collections import Counter
import cv2
from ultralytics import YOLO

# ----------------------------
# Load YOLO Model
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

model = YOLO(str(MODEL_PATH))

# ----------------------------
# Animal classes from COCO
# ----------------------------
ANIMAL_CLASSES = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]

# ----------------------------
# Carnivores
# ----------------------------
CARNIVORES = [
    "cat",
    "dog",
    "bear"
]

# ----------------------------
# Animal Detection Function
# ----------------------------
def detect_animals(image):

    results = model(image)

    output = image.copy()

    total_animals = 0
    carnivore_count = 0
    detected_animals = []

    detection_details = []
    confidence_scores = []
    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in ANIMAL_CLASSES:
                continue

            confidence = float(box.conf[0])
            confidence_scores.append(confidence)
            total_animals += 1
            detected_animals.append(label)

            detection_details.append({
                "Animal": label.capitalize(),
                "Confidence (%)": round(confidence * 100, 2)
            })

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in CARNIVORES:
                color = (0, 0, 255)
                carnivore_count += 1
            else:
                color = (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                output,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    animal_count = Counter(detected_animals)

    return (
        output,
        total_animals,
        carnivore_count,
        detected_animals,
        detection_details,
        animal_count,
        confidence_scores
        
    )