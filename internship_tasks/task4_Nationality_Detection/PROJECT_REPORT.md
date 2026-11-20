# Project Report: Nationality-Based Image Attribute Detection System

**Submitted by:** Priti Vijay Singh  
**Program:** B.E. Computer Science Engineering – Data Science  

---

## 1. Abstract
This report documents the design, implementation, and verification of the **Nationality-Based Image Attribute Detection System**, an interactive Python and Streamlit computer vision application. The system accepts an uploaded image of a person alongside a user-selected nationality or regional category (`Indian`, `United States`, `African country/region`, or `Other`) and executes conditional attribute analysis. Depending on the selected category, the software predicts facial emotion, estimates apparent age, and identifies dominant upper-body dress colour while strictly avoiding any automated inference of nationality, race, or ethnicity from facial features.

## 2. Introduction
Modern computer vision libraries make it straightforward to extract visual cues such as facial expressions, apparent age, and clothing colour. However, deploying multiple models indiscriminately increases computational latency and raises ethical concerns when demographic traits are inferred without user consent. This internship project addresses both challenges by building a modular, conditionally routed inference pipeline wrapped in an accessible Streamlit graphical user interface.

## 3. Problem Statement
Design and implement an executable, beginner-friendly Python machine learning application that:
1. Accepts a portrait or upper-body image (`JPG`, `JPEG`, `PNG`) and a user-declared category.
2. Never attempts to classify or guess nationality, race, or ethnicity from facial appearance.
3. Executes only the specific detection modules mapped to each user-selected category.
4. Segments the upper-body clothing region accurately for dress colour detection rather than sampling background pixels.
5. Handles invalid uploads, missing faces, and occluded clothing gracefully.

## 4. Objectives
- Develop clean, decoupled Python modules for image preprocessing, emotion recognition, age estimation, dress colour segmentation, and conditional category routing.
- Utilize established pretrained models (`DeepFace` and `MediaPipe`/`OpenCV`) without making unverified training claims.
- Provide clear visual feedback, approximate-value disclaimers, and error handling inside a two-column Streamlit GUI.
- Verify core logic and edge cases using automated `pytest` unit tests.

## 5. Proposed System
The proposed system decouples user-declared demographic context from visual attribute extraction:
- **User Input Layer:** Users upload an image and explicitly select `Indian`, `United States`, `African country/region`, or `Other` (entering custom text for `Other`).
- **Conditional Orchestration Layer:** A rule-based dispatcher inspects the selected category and invokes only the required inference modules.
- **Vision Inference Layer:** Independent modules perform facial emotion classification, apparent age estimation, and torso clothing segmentation + dominant colour naming.

## 6. System Architecture
```text
+-------------------------------------------------------------------+
|                     Streamlit Frontend (app.py)                   |
|  [File Uploader]   [Category Dropdown]   [Declared Nationality]   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|           Image Validation & Preprocessing (image_processing.py)  |
|  - Extension/Size/Integrity Check  - RGB/BGR Conversion & Resize  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|          Conditional Category Router (nationality_logic.py)       |
|  - Indian:                Age + Emotion + Dress Colour            |
|  - United States:         Age + Emotion                           |
|  - African country/region: Emotion + Dress Colour                 |
|  - Other:                 Declared Nationality + Emotion          |
+----------------+----------------+----------------+----------------+
                 |                |                |
                 v                v                v
      +------------------+ +--------------+ +--------------------+
      | age_detection.py | |emotion_det.py| |  dress_colour.py   |
      | (DeepFace Age)   | |(DeepFace Emo)| | (MediaPipe/GrabCut |
      +------------------+ +--------------+ |  + LAB K-Means)    |
                                            +--------------------+
```

## 7. Technologies Used
- **Python (3.10–3.11):** Core programming language.
- **Streamlit:** Interactive web GUI framework with session state management.
- **DeepFace & TensorFlow / tf-keras:** Pretrained facial emotion and apparent age inference.
- **MediaPipe & OpenCV:** Person foreground segmentation, Haar cascade face detection, GrabCut fallback segmentation, YCrCb skin filtering, and K-Means clustering.
- **NumPy & Pillow:** Matrix operations and image decoding/validation.
- **Pytest:** Automated unit testing framework with `unittest.mock`.

## 8. Dataset Description
No custom dataset was collected or used to train models in this project. Instead, inference relies on pretrained weights distributed by upstream open-source frameworks:
- **Emotion Recognition:** Pretrained on FER-2013 (7 facial expression classes).
- **Age Estimation:** Pretrained on IMDB-WIKI facial age imagery.
- **Person Segmentation:** Pretrained MediaPipe Selfie Segmentation model.
Detailed documentation is maintained in `DATASET.md`.

## 9. Image Preprocessing
Implemented in `modules/image_processing.py`:
1. **File Validation:** Verifies non-empty byte streams, allowed extensions (`.jpg`, `.jpeg`, `.png`), maximum file size (`10 MB`), image header integrity via `PIL.Image.verify()`, and minimum resolution (`64x64` pixels).
2. **Colour Space Standardization:** Converts uploaded images to standard 3-channel RGB and OpenCV BGR NumPy arrays.
3. **Proportional Resizing:** Scales images exceeding `1024 px` along their longest edge using `cv2.INTER_AREA` to ensure fast, memory-safe inference.

## 10. Emotion Detection
Implemented in `modules/emotion_detection.py`:
- Invokes `DeepFace.analyze(..., actions=["emotion"], detector_backend="opencv", enforce_detection=True)`.
- Extracts the dominant emotion label, normalizes confidence percentages, and returns the full probability breakdown across all seven emotion classes.
- Catches `ValueError` exceptions when no face is detectable and returns a structured, user-friendly error message without crashing the GUI.

## 11. Age Estimation
Implemented in `modules/age_detection.py`:
- Invokes `DeepFace.analyze(..., actions=["age"], detector_backend="opencv", enforce_detection=True)`.
- Executes **only** when `Indian` or `United States` is selected.
- Formats output explicitly as `~<N> years (Estimated, not exact)` so users understand the figure is a visual approximation rather than ground-truth age.

## 12. Dress Colour Detection
Implemented in `modules/dress_colour.py`:
1. **Face-Guided Torso Localization:** Detects the frontal face bounding box via OpenCV Haar Cascade to anchor the expected upper-body clothing zone below the chin.
2. **Person Segmentation:** Generates a foreground silhouette mask using `MediaPipe SelfieSegmentation` (with automatic fallback to `OpenCV GrabCut` if MediaPipe is unavailable).
3. **Skin Exclusion:** Filters out neck/hand skin pixels using standard YCrCb skin-tone thresholds (`Cr: 133–173, Cb: 77–127`).
4. **Visibility Verification:** If the face occupies nearly the entire frame or fewer than 1.5% of image pixels belong to valid clothing, the module aborts and returns a clear warning rather than guessing a colour.
5. **Dominant Colour Extraction:** Runs 3-cluster K-Means on the remaining RGB clothing pixels and maps the largest cluster centroid to the nearest named colour in perceptual CIE LAB space.

## 13. Nationality Selection Logic
Implemented in `modules/nationality_logic.py`:
- Enforces the exact conditional matrix:
  - `Indian` $ightarrow$ `["age", "emotion", "dress_colour"]`
  - `United States` $ightarrow$ `["age", "emotion"]`
  - `African country/region` $ightarrow$ `["emotion", "dress_colour"]`
  - `Other` $ightarrow$ `["declared_nationality", "emotion"]`
- Validates that `Other` includes a non-empty user-declared nationality string (2–80 characters).

## 14. GUI Implementation
Implemented in `app.py`:
- **Sidebar Controls:** File uploader (`JPG`, `JPEG`, `PNG`), category dropdown, conditional text input for `Other`, category requirement preview banner, **Analyze Image** button, and **Reset** button.
- **Two-Column Layout:** Left column renders the uploaded image preview; right column renders conditional prediction cards, confidence progress bars, a visual HEX colour swatch for dress colour, and contextual alerts (`st.error` / `st.warning`).

## 15. Testing Methodology
Automated unit tests are organized under `tests/`:
- `test_image_processing.py`: Tests valid JPEG/PNG decoding, missing files, invalid extensions, corrupted bytes, undersized images, and proportional resizing.
- `test_emotion.py`: Tests successful emotion parsing, missing-face error handling, invalid inputs, and missing library handling via `unittest.mock`.
- `test_age.py`: Tests age rounding/labeling, missing-face handling, and empty inputs.
- `test_dress_colour.py`: Tests LAB colour name mapping, synthetic blue-shirt segmentation, tight-headshot warnings, and segmentation failure warnings.
- `test_nationality_logic.py`: Verifies that each of the four categories triggers **only** its designated models and skips unnecessary calls.

## 16. Results
- All modular components, conditional routing rules, image validation checks, and colour segmentation algorithms were verified via automated unit tests and direct syntax/import validation.
- Because pretrained DeepFace weights are downloaded dynamically on first execution in the target environment, end-to-end live inference accuracy metrics are intentionally not fabricated in this report.

## 17. Limitations
- Single-subject assumption: When multiple people appear in an image, only the primary/largest detected face is analyzed.
- Lighting sensitivity: Heavy colour casts (e.g., warm indoor tungsten lighting) can shift perceived RGB clothing tones.
- Pretrained model constraints: Apparent age and emotion predictions reflect the statistical patterns and biases of their upstream training datasets.

## 18. Ethical Considerations
- **No Automated Nationality or Race Inference:** Inferring nationality or ethnicity from facial geometry or skin tone is scientifically invalid and reinforces harmful stereotyping. This application strictly uses self-declared categories.
- **Continental Distinction:** Recognizing that Africa is a continent comprising 54 sovereign nations rather than a single nationality, the category is explicitly labeled `African country/region`.
- **Transparency:** Age predictions are explicitly labeled as estimates, and confidence scores are exposed alongside emotion predictions.

## 19. Future Scope
- Integrate multi-face selection so users can pick a specific person in group photos.
- Add multi-colour pattern detection (primary + secondary garment colours).
- Support batch image analysis with downloadable CSV summaries.

## 20. Conclusion
The project successfully implements a complete, modular, and ethically grounded Nationality-Based Image Attribute Detection System suitable for academic and internship evaluation.

## 21. References
1. Serengil, S. I., & Ozpinar, A. (2020). *LightFace: A Hybrid Deep Face Recognition Framework*. IEEE Innovations in Intelligent Systems and Applications Conference (ASYU).
2. Goodfellow, I. J., et al. (2013). *Challenges in Representation Learning: A Report on Three Machine Learning Contests* (FER-2013).
3. Rothe, R., Timofte, R., & Van Gool, L. (2015). *DEX: Deep Expectation of Apparent Age from a Single Image*. IEEE ICCV Workshops.
4. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. Google Research.
5. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
