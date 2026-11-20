import numpy as np
from PIL import Image
from src.app import preprocess_image


def test_preprocess_image_shape():
    image = Image.new("RGB", (200, 200))

    processed = preprocess_image(
        image,
        target_size=(100, 100)
    )

    assert processed.shape == (1, 100, 100, 3)


def test_image_normalization():
    image = Image.new(
        "RGB",
        (100, 100),
        color="white"
    )

    processed = preprocess_image(image)

    assert processed.min() >= 0.0
    assert processed.max() <= 1.0