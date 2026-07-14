import numpy as np


EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]


def test_emotion_classes():
    assert len(EMOTIONS) == 7


def test_prediction_logic():
    prediction = np.array(
        [0.01, 0.02, 0.01, 0.90, 0.02, 0.02, 0.02]
    )

    predicted_class = np.argmax(prediction)

    emotion = EMOTIONS[predicted_class]

    assert emotion == "Happy"