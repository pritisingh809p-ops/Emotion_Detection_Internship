import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from PIL import Image

from animal_detection import detect_animals
import animal_detection

st.write(animal_detection.__file__)
from video_detection import process_video
from utils import load_image, convert_to_rgb
from carnivore_counter import generate_summary, get_alert_message

st.set_page_config(
    page_title="Animal Detection Model",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Animal Detection Model")

st.markdown("""
This application detects multiple animals from **images** and **videos**.

### Features

- Multiple Animal Detection
- Carnivores highlighted in Red
- Other Animals highlighted in Green
- Detection Summary
- Animal Count
- Confidence Score
- Download Processed Result
""")

mode = st.sidebar.radio(
    "Choose Detection Mode",
    ["Image Detection", "Video Detection"]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Supported Animals

🐻 Bear

🐱 Cat

🐶 Dog

🐄 Cow

🐎 Horse

🐘 Elephant

🦓 Zebra

🐦 Bird
"""
)
# ======================================================
# IMAGE DETECTION
# ======================================================
# ======================================================
# IMAGE DETECTION
# ======================================================

if mode == "Image Detection":

    uploaded_image = st.file_uploader(
        "Upload an Animal Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = load_image(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                convert_to_rgb(image),
                use_container_width=True
            )

        if st.button("Detect Animals", use_container_width=True):

            with st.spinner("Detecting animals..."):

                (
                    output,
                    total_animals,
                    carnivore_count,
                    detected_animals,
                    detection_details,
                    animal_count,
                    confidence_scores
                ) = detect_animals(image)

                if confidence_scores:
                    avg_confidence = (
                        sum(confidence_scores) / len(confidence_scores)
                    ) * 100
                else:
                    avg_confidence = 0

            with col2:
                st.subheader("Detection Result")
                st.image(
                    convert_to_rgb(output),
                    use_container_width=True
                )

            st.success("Detection Completed Successfully!")

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Total Animals", total_animals)

            with c2:
                st.metric("Carnivores", carnivore_count)

            with c3:
                st.metric(
                    "Other Animals",
                    total_animals - carnivore_count
                )

            with c4:
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.2f}%"
                )

            st.divider()

            summary = generate_summary(
                total_animals,
                carnivore_count
            )

            st.success(summary)

            alert = get_alert_message(carnivore_count)

            if carnivore_count > 0:
                st.error(alert)
            else:
                st.success(alert)

            st.divider()

            st.subheader("Detected Animals")

            if len(detected_animals) == 0:

                st.warning("No animals detected.")

            else:

                df = pd.DataFrame(detection_details)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Animal Count")

                count_df = pd.DataFrame(
                    list(animal_count.items()),
                    columns=["Animal", "Count"]
                )

                st.dataframe(
                    count_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            save_path = "detected_animals.jpg"

            cv2.imwrite(save_path, output)

            with open(save_path, "rb") as file:

                st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name="animal_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
# ======================================================
# VIDEO DETECTION
# ======================================================

elif mode == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload Animal Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button("Detect Animals in Video", use_container_width=True):

            with st.spinner("Processing video..."):

                temp_input = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                temp_input.write(uploaded_video.read())
                temp_input.close()

                output_video = "processed_video.mp4"

                process_video(
                    temp_input.name,
                    output_video
                )

            st.success("Video Detection Completed Successfully!")

            st.divider()

            st.subheader("Processed Video")

            st.video(output_video)

            with open(output_video, "rb") as file:

                st.download_button(

                    label="Download Processed Video",

                    data=file,

                    file_name="animal_detection_video.mp4",

                    mime="video/mp4",

                    use_container_width=True

                )

            os.remove(temp_input.name)