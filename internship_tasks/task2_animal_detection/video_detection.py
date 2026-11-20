import cv2
import os
from animal_detection import detect_animals


def process_video(input_video_path, output_video_path):
    """
    Process a video frame by frame and detect animals.

    Args:
        input_video_path (str): Path of uploaded video
        output_video_path (str): Path to save processed video

    Returns:
        output_video_path
    """

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Unable to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frame, total_animals, carnivore_count, detected_animals = detect_animals(frame)

        cv2.putText(
            processed_frame,
            f"Animals : {total_animals}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            processed_frame,
            f"Carnivores : {carnivore_count}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        out.write(processed_frame)

    cap.release()
    out.release()

    return output_video_path