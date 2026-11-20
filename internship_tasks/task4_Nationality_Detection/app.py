"""
Nationality-Based Image Attribute Detection System - Streamlit Web Application.

Developer: Priti Vijay Singh (B.E. Computer Science Engineering - Data Science)
"""

import streamlit as st
from internship_tasks.task4_Nationality_Detection.modules.image_processing import (
    validate_uploaded_file,
    resize_for_inference,
    ImageValidationError,
)
from internship_tasks.task4_Nationality_Detection.modules.nationality_logic import (
    SUPPORTED_CATEGORIES,
    CATEGORY_INDIAN,
    CATEGORY_US,
    CATEGORY_AFRICAN,
    CATEGORY_OTHER,
    run_attribute_pipeline,
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables for clean reset functionality."""
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "error_message" not in st.session_state:
        st.session_state.error_message = None


def reset_application() -> None:
    """Clear uploaded file widget, cached results, and error states."""
    st.session_state.uploader_key += 1
    st.session_state.analysis_result = None
    st.session_state.error_message = None


def render_category_requirements_hint(category: str) -> None:
    """Display which attributes will be analyzed for the chosen category."""
    hints = {
        CATEGORY_INDIAN: "Outputs for **Indian**: Estimated Age • Facial Emotion • Dress Colour",
        CATEGORY_US: "Outputs for **United States**: Estimated Age • Facial Emotion",
        CATEGORY_AFRICAN: "Outputs for **African country/region**: Facial Emotion • Dress Colour",
        CATEGORY_OTHER: "Outputs for **Other**: Declared Nationality • Facial Emotion",
    }
    st.info(hints.get(category, ""))


def render_results_panel(result_payload: dict) -> None:
    """Render conditional prediction cards cleanly in the right-hand column."""
    outputs = result_payload.get("outputs", {})
    category = result_payload.get("category", "")

    st.success(f"Analysis complete for category: **{category}**")

    # 1. Declared Nationality (Only for 'Other')
    if "declared_nationality" in outputs:
        nat_data = outputs["declared_nationality"]
        st.markdown("### 🌍 User-Declared Nationality")
        st.metric(label="Declared Nationality", value=nat_data["value"])
        st.divider()

    # 2. Estimated Age (Only for 'Indian' and 'United States')
    if "age" in outputs:
        age_data = outputs["age"]
        st.markdown("### 🎂 Apparent Age Estimation")
        if age_data["success"]:
            st.metric(
                label="Estimated Age (Approximate)",
                value=f"{age_data['estimated_age']} years",
            )
            st.caption(
                "Note: This value is an automated visual estimate from a pretrained model, "
                "not an exact biological age."
            )
        else:
            st.error(age_data.get("error", "Unable to estimate age."))
        st.divider()

    # 3. Facial Emotion (Included in all 4 categories)
    if "emotion" in outputs:
        emo_data = outputs["emotion"]
        st.markdown("### 😊 Facial Emotion Detection")
        if emo_data["success"]:
            conf = emo_data.get("confidence")
            conf_str = f"{conf:.1f}%" if conf is not None else "N/A"
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.metric(label="Dominant Emotion", value=emo_data["dominant_emotion"])
            with col_e2:
                st.metric(label="Model Confidence", value=conf_str)

            all_scores = emo_data.get("all_emotions", {})
            if all_scores:
                with st.expander("View Emotion Probability Breakdown"):
                    for emo_label, score in sorted(
                        all_scores.items(), key=lambda x: x[1], reverse=True
                    ):
                        st.write(f"**{emo_label}**: {score:.2f}%")
                        st.progress(min(max(int(round(score)), 0), 100))
        else:
            st.error(emo_data.get("error", "Unable to detect facial emotion."))
        st.divider()

    # 4. Dress Colour (Only for 'Indian' and 'African country/region')
    if "dress_colour" in outputs:
        dress_data = outputs["dress_colour"]
        st.markdown("### 👕 Dress Colour Detection")
        if dress_data["success"]:
            colour_name = dress_data["colour_name"]
            hex_code = dress_data["hex_code"]
            rgb = dress_data["rgb"]
            method = dress_data.get("segmentation_method", "Segmentation")

            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                st.metric(label="Detected Dominant Colour", value=colour_name)
                st.caption(f"RGB: {rgb} | HEX: {hex_code} | Method: {method}")
            with col_d2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {hex_code};
                        width: 100%;
                        height: 68px;
                        border-radius: 10px;
                        border: 2px solid #cccccc;
                        margin-top: 6px;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                dress_data.get(
                    "warning",
                    "Clothing could not be segmented clearly to determine dress colour.",
                )
            )


def main() -> None:
    st.set_page_config(
        page_title="Nationality-Based Image Attribute Detection",
        page_icon="🔍",
        layout="wide",
    )
    init_session_state()

    # Header Section
    st.title("🔍 Nationality-Based Image Attribute Detection System")
    st.markdown(
        """
        Upload a portrait or upper-body photo of a person and select a nationality/regional category 
        to run conditional computer vision analysis (Age Estimation, Facial Emotion Recognition, 
        and Clothing Colour Detection).

        > **Ethical AI Design Notice:** This system **never** predicts or infers nationality, race, 
        or ethnicity from facial appearance. Nationality or regional category is strictly self-declared 
        by the user.
        """
    )
    st.divider()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration & Inputs")

        uploaded_file = st.file_uploader(
            "1. Upload Person Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"image_uploader_{st.session_state.uploader_key}",
            help="Maximum file size: 10 MB. Clear front-facing upper-body images work best.",
        )

        selected_category = st.selectbox(
            "2. Select Nationality / Category",
            options=SUPPORTED_CATEGORIES,
            index=0,
            help="Choose the user-declared category to determine which attributes are analyzed.",
        )

        declared_nationality = ""
        if selected_category == CATEGORY_OTHER:
            declared_nationality = st.text_input(
                "3. Enter Declared Nationality *",
                placeholder="e.g., Japanese, Brazilian, German, Canadian",
                max_chars=80,
                help="Required when 'Other' is selected.",
            )

        render_category_requirements_hint(selected_category)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_clicked = st.button(
                "🚀 Analyze Image",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            reset_clicked = st.button(
                "🔄 Reset",
                use_container_width=True,
            )

        if reset_clicked:
            reset_application()
            st.rerun()

        st.divider()
        st.caption(
            "**Developer:** Priti Vijay Singh  \n"
            "B.E. Computer Science Engineering – Data Science"
        )

    # Main Two-Column Workspace
    left_col, right_col = st.columns([1, 1], gap="large")

    validated_pil = None
    validated_bgr = None

    with left_col:
        st.subheader("🖼️ Input Image Preview")
        if uploaded_file is not None:
            try:
                validated_pil, validated_bgr = validate_uploaded_file(uploaded_file)
                st.image(
                    validated_pil,
                    caption=f"Uploaded: {uploaded_file.name} ({validated_pil.size[0]}×{validated_pil.size[1]} px)",
                    use_container_width=True,
                )
            except ImageValidationError as exc:
                st.session_state.error_message = str(exc)
                st.session_state.analysis_result = None
                st.error(f"Image Validation Error: {exc}")
        else:
            st.info("Upload a JPG, JPEG, or PNG image from the left sidebar to begin.")

    # Execute Pipeline on Button Click
    if analyze_clicked:
        st.session_state.error_message = None
        st.session_state.analysis_result = None

        if uploaded_file is None or validated_bgr is None:
            st.session_state.error_message = (
                "Please upload a valid JPG, JPEG, or PNG image before clicking 'Analyze Image'."
            )
        else:
            try:
                with st.spinner("Analyzing image attributes... Please wait."):
                    processed_bgr = resize_for_inference(validated_bgr, max_dim=1024)
                    result = run_attribute_pipeline(
                        bgr_image=processed_bgr,
                        category=selected_category,
                        declared_nationality=declared_nationality,
                    )
                    st.session_state.analysis_result = result
            except ValueError as val_err:
                st.session_state.error_message = str(val_err)
            except Exception as exc:
                st.session_state.error_message = f"Unexpected processing error: {exc}"

    with right_col:
        st.subheader("📊 Prediction Results")
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        elif st.session_state.analysis_result is not None:
            render_results_panel(st.session_state.analysis_result)
        else:
            st.info(
                "Select a category, upload an image, and click **Analyze Image** "
                "to view conditional attribute predictions here."
            )


if __name__ == "__main__":
    main()