"""
Drowsiness Detection Model - Professional Streamlit Web Application.

Features:
- Multi-person detection in vehicle images or video frames.
- Identifies total people count and sleeping count.
- Marks sleeping individuals in RED bounding boxes and predicts their age.
- Pops up an alert message displaying the number of sleeping people and their ages.
- Image and video preview interface.

Developer: Priti Vijay Singh (B.E. Computer Science Engineering - Data Science)
"""

import streamlit as st
import cv2
import numpy as np
from modules.image_processing import validate_uploaded_image
from modules.drowsiness_logic import analyze_vehicle_occupants


def main() -> None:
    st.set_page_config(
        page_title="Vehicle Drowsiness Detection System",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 Vehicle Drowsiness Detection Model")
    st.markdown(
        """
        Monitor vehicle cabin safety by detecting multiple occupants, determining sleep/awake status, 
        marking sleeping individuals in **Red**, estimating their age, and triggering instant alert pop-ups.
        """
    )
    st.divider()

    # Sidebar Controls
    st.sidebar.header("⚙️ Operation Mode")
    mode = st.sidebar.radio(
        "Choose Input Type",
        options=["Upload Vehicle Image", "Video Frame / Webcam Analysis"],
    )

    with st.sidebar:
        st.divider()
        st.caption("**Developer:** Priti Vijay Singh\nB.E. Computer Science Engineering – Data Science")

    if mode == "Upload Vehicle Image":
        st.subheader("🖼️ Vehicle Cabin Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload cabin/driver photo (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        col_prev, col_res = st.columns(2, gap="large")

        with col_prev:
            st.markdown("### 📷 Input Image Preview")
            if uploaded_file is not None:
                try:
                    pil_img, bgr_img = validate_uploaded_image(uploaded_file)
                    st.image(pil_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

                    if st.button("🚀 Analyze Occupants & Detect Drowsiness", type="primary", use_container_width=True):
                        with st.spinner("Analyzing vehicle cabin occupants..."):
                            result = analyze_vehicle_occupants(bgr_img)
                            st.session_state.analysis_result = result
                except Exception as exc:
                    st.error(f"Validation Error: {exc}")
            else:
                st.info("Please upload an image of the vehicle cabin.")

        with col_res:
            st.markdown("### 📊 Detection & Alert Dashboard")
            if "analysis_result" in st.session_state and st.session_state.analysis_result:
                res = st.session_state.analysis_result
                if res["success"]:
                    # POP-UP ALERT MESSAGE (Task Requirement)
                    if res["sleeping_count"] > 0:
                        st.error(f"🚨 POP-UP ALERT: {res['alert_message']}")
                    else:
                        st.success(f"🟢 POP-UP NOTIFICATION: {res['alert_message']}")

                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric(label="Total People Detected", value=res["total_people"])
                    col_m2.metric(label="Sleeping Occupants", value=res["sleeping_count"])

                    st.markdown("#### Annotated Cabin View (Red = Sleeping)")
                    st.image(res["annotated_image"], channels="RGB", use_container_width=True)

                    with st.expander("View Occupant Details"):
                        for occ in res["occupants"]:
                            status_emoji = "🔴 Sleeping" if occ["status"] == "Sleeping" else "🟢 Awake"
                            st.write(f"- **Occupant {occ['id']}**: {status_emoji} | Estimated Age: **{occ['age']} yrs**")
                else:
                    st.error("Analysis failed.")
            else:
                st.info("Upload an image and click **Analyze Occupants & Detect Drowsiness** to view results.")

    elif mode == "Video Frame / Webcam Analysis":
        st.subheader("📹 Live Video / Webcam Drowsiness Monitoring")
        st.markdown("Capture a live frame from your vehicle camera or webcam to inspect occupant alertness.")

        use_cam = st.toggle("🔴 Enable Camera Feed", value=False)

        if use_cam:
            cam_file = st.camera_input("Capture vehicle cabin frame")

            if cam_file is not None:
                file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1, cv2.IMREAD_COLOR)

                if frame is not None:
                    col_v1, col_v2 = st.columns(2, gap="large")
                    with col_v1:
                        st.markdown("### Captured Frame Preview")
                        st.image(frame, channels="BGR", use_container_width=True)
                    with col_v2:
                        st.markdown("### Drowsiness Assessment")
                        with st.spinner("Processing video frame..."):
                            res = analyze_vehicle_occupants(frame)
                            if res["sleeping_count"] > 0:
                                st.error(f"🚨 POP-UP ALERT: {res['alert_message']}")
                            else:
                                st.success(f"🟢 POP-UP NOTIFICATION: {res['alert_message']}")
                            
                            st.metric(label="Total Occupants", value=res["total_people"])
                            st.metric(label="Sleeping Count", value=res["sleeping_count"])
                            st.image(res["annotated_image"], channels="RGB", use_container_width=True)
        else:
            st.info("Toggle **Enable Camera Feed** above to test live frame capture.")


if __name__ == "__main__":
    main()
