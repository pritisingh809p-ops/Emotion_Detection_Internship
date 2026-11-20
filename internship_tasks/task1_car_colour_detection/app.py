import streamlit as st
from PIL import Image
import numpy as np
import cv2
from car_detector import detect_cars_and_people, draw_detections

st.set_page_config(page_title="Car Colour Detection", page_icon="🚦", layout="wide")
st.title("🚦 Car Colour Detection System")
st.write("Upload a traffic-signal image to detect cars, estimate their colours, count cars and people, and highlight blue cars in red.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    st.caption("The first run downloads the YOLOv8n object-detection weights. An internet connection is required.")

uploaded = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    if st.button("Detect cars and people", type="primary"):
        with st.spinner("Detecting vehicles and people..."):
            try:
                detections, counts = detect_cars_and_people(image_np, confidence)
                result = draw_detections(image_np, detections)
                with right:
                    st.subheader("Detection preview")
                    st.image(result, use_container_width=True)
                st.success("Detection completed.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Cars detected", counts["cars"])
                m2.metric("People detected", counts["people"])
                m3.metric("Blue cars", counts["blue_cars"])
                st.info(f"Number of people at the traffic signal: {counts['people']}")
                if counts["blue_cars"]:
                    st.warning(f"Detected {counts['blue_cars']} blue car(s). Blue cars are highlighted with red rectangles.")
                else:
                    st.info("No blue cars were detected.")
                if detections:
                    st.subheader("Detection details")
                    st.dataframe(
                        [{"Object": d["label"], "Colour": d.get("colour", "—"),
                          "Confidence": f'{d["confidence"]:.1%}'} for d in detections],
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.warning("No cars or people were detected. Try another image or lower the confidence threshold.")
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                if ok:
                    st.download_button("Download annotated image", encoded.tobytes(),
                                       file_name="car_colour_detection_result.jpg",
                                       mime="image/jpeg")
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.caption("Check your internet connection for the first model download, then try again.")
else:
    st.info("Choose an image above to begin. The application displays the original and annotated image previews.")
st.divider()
st.caption("Educational computer-vision project. Colour estimates can be affected by lighting, shadows, reflections and image quality.")
