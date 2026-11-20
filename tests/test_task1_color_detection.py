import numpy as np

from internship_tasks.task1_car_colour_detection.color_detection import (
    is_blue_car
)


def test_blue_car_detection():

    blue_image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    blue_image[:] = (255, 0, 0)

    assert is_blue_car(blue_image) is True


def test_non_blue_car_detection():

    red_image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    red_image[:] = (0, 0, 255)

    assert is_blue_car(red_image) is False


def test_empty_image():

    empty_image = np.array([])

    assert is_blue_car(empty_image) is False