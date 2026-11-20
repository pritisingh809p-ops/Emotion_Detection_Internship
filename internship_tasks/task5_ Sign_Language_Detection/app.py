"""
Sign Language Detection System - Professional Streamlit Web Application.

Features:
- Strict Time Window Enforcement (Operational 6:00 PM to 10:00 PM)
- Dual Mode: Image Upload & Real-Time Webcam Stream
- Rich Vocabulary Recognition (HELLO, THANK_YOU, YES, NO, PLEASE, HELP, GOODBYE, DISAGREE)
- Modern CSS Styling & Interactive Analytics Dashboard

Developer: Priti Vijay Singh (B.E. Computer Science Engineering - Data Science)
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from modules.time_validator import is_operational_time
from modules.image_processing import validate_uploaded_file, ImageValidationError
from modules.sign_recognition import recognize_sign, SUPPORTED_SIGNS


def inject_custom_css() -> None:
    """Inject polished custom CSS for an attractive UI."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #ff4b4b;
        }
        .badge-active {
            background-color: #d4edda;
            color: #155724;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        .badge-locked {
            background-color: #f8d7da;
            color: #721c24;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Sign Language Detection System",
        page_icon="🖐️",
        layout="wide",
    )
    inject_custom_css()

    # Header Banner
    st.title("🖐️ Sign Language Detection & Recognition System")
    st.markdown(
        """
        An advanced machine learning web application designed to recognize known sign language words 
        (**HELLO**, **THANK_YOU**, **YES**, **NO**, **PLEASE**, **HELP**, **GOODBYE**, **DISAGREE**) from static images 
        and real-time camera streams.
        """
    )
    st.divider()

    # Time Window Validation
    is_active, time_msg = is_operational_time()

    with st.sidebar:
        st.header("🕒 System Status")
        if is_active:
            st.markdown('<span class="badge-active">🟢 SYSTEM ACTIVE (6 PM – 10 PM)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-locked">🔴 SYSTEM LOCKED (Outside Hours)</span>', unsafe_allow_html=True)
        
        st.divider()
        st.header("⚙️ Navigation & Mode")
        mode = st.radio(
            "Select Input Method",
            options=["Upload Image", "Real-Time Video / Webcam", "Vocabulary Reference"],
            index=0,
        )

        st.divider()
        st.caption("**Developer:** Priti Vijay Singh \nB.E. Computer Science Engineering – Data Science")

    if not is_active:
        st.error(f"🔒 {time_msg}")
        st.warning(
            "The system is currently outside its authorized operational window (6:00 PM to 10:00 PM). "
            "Please return during operational hours or check unit tests for evaluation."
        )
        with st.expander("📚 View Authorized Sign Vocabulary"):
            for word, desc in SUPPORTED_SIGNS.items():
                st.write(f"- **{word}**: {desc}")
        return

    if mode == "Vocabulary Reference":
        st.subheader("📚 Recognized Sign Vocabulary & Descriptions")
        st.markdown("The model recognizes the following core sign language gestures and words:")
        
        cols = st.columns(2)
        for i, (word, desc) in enumerate(SUPPORTED_SIGNS.items()):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div style="background: white; padding: 16px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-top: 3px solid #0066cc;">
                        <h4 style="margin: 0 0 8px 0; color: #0066cc;">📌 {word}</h4>
                        <p style="margin: 0; color: #555;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    elif mode == "Upload Image":
        st.subheader("🖼️ Upload Sign Language Image")
        st.markdown("Upload a clear upper-body or hand sign photo (`JPG`, `JPEG`, `PNG`) for instant inference.")

        uploaded_file = st.file_uploader(
            "Choose a sign image file",
            type=["jpg", "jpeg", "png"],
            help="Maximum file size: 10 MB.",
        )

        col_img, col_res = st.columns(2, gap="large")

        with col_img:
            st.markdown("#### 📷 Input Preview")
            if uploaded_file is not None:
                try:
                    pil_img, bgr_img = validate_uploaded_file(uploaded_file)
                    st.image(pil_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    
                    if st.button("🚀 Analyze & Recognize Sign", type="primary", use_container_width=True):
                        with st.spinner("Processing hand landmarks and classifying sign..."):
                            result = recognize_sign(bgr_img)
                            st.session_state.upload_result = result
                except Exception as exc:
                    st.error(f"Validation Error: {exc}")
            else:
                st.info("Upload an image file using the uploader above.")

        with col_res:
            st.markdown("#### 📊 Prediction Dashboard")
            if "upload_result" in st.session_state and st.session_state.upload_result:
                res = st.session_state.upload_result
                if res["success"]:
                    st.metric(label="Predicted Sign / Word", value=res["sign"])
                    st.metric(label="Confidence Score", value=f"{res['confidence']:.2f}%")
                    st.success(f"**Gesture Description:** {res['description']}")
                    st.progress(min(max(int(res['confidence']), 0), 100))
                else:
                    st.error(res["error"])
            else:
                st.info("Upload an image and click **Analyze & Recognize Sign** to view predictions here.")

    elif mode == "Real-Time Video / Webcam":
        st.subheader("📹 Real-Time Video & Webcam Stream")
        st.markdown(
            "Capture live video frames from your connected webcam to test real-time sign language detection."
        )

        enable_cam = st.toggle("🔴 Turn On Camera Feed", value=False)

        if enable_cam:
            cam_file = st.camera_input("Capture frame for classification")

            if cam_file is not None:
                file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1, cv2.IMREAD_COLOR)

                if frame is not None:
                    col_v1, col_v2 = st.columns(2, gap="large")
                    with col_v1:
                        st.markdown("#### Captured Frame")
                        st.image(frame, channels="BGR", use_container_width=True)
                    with col_v2:
                        st.markdown("#### Live Recognition Result")
                        with st.spinner("Classifying live frame..."):
                            res = recognize_sign(frame)
                            if res["success"]:
                                st.metric(label="Detected Sign", value=res["sign"])
                                st.metric(label="Confidence", value=f"{res['confidence']:.2f}%")
                                st.success(res["description"])
                            else:
                                st.warning(res["error"])
        else:
            st.info("Toggle **Turn On Camera Feed** above to activate webcam capture.")


if __name__ == "__main__":
    main()
