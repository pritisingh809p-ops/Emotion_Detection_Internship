import cv2
import os
import sys
import matplotlib.pyplot as plt

# --------------------------------------------------
# Add Task 2 folder to Python path
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "Internship_tasks",
            "task2_animal_detection"
        )
    )
)

from animal_detection import detect_animals

# --------------------------------------------------
# Image Path
# --------------------------------------------------

IMAGE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "sample_images",
        "task2",
        "dog.jpg"
    )
)

print("Image Path:", IMAGE_PATH)
print("Exists:", os.path.exists(IMAGE_PATH))

# --------------------------------------------------
# Read Image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

print("Image Variable:", image)

if image is None:
    print("❌ OpenCV could not read the image.")
    exit()

print("✅ Image loaded successfully.")

# --------------------------------------------------
# Detect
# --------------------------------------------------

output, animals = detect_animals(image)

print("Detected Animals:", animals)

# --------------------------------------------------
# Show
# --------------------------------------------------

cv2.imshow("Animal Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()