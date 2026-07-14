import numpy as np
from app import detect_face


class MockDetector:

    def detect_faces(self, image):
        return []


def test_no_face_detection():

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    detector = MockDetector()

    result = detect_face(
        image,
        detector
    )

    assert result is None
    