# Nationality-Based Image Attribute Detection System

## 1. Project Title
**Nationality-Based Image Attribute Detection System**

## 2. Introduction
This project is an interactive computer vision web application built with **Python**, **Streamlit**, **DeepFace**, **MediaPipe**, and **OpenCV**. Given a photo of a person and a **user-selected** nationality or regional category, the application conditionally executes only the relevant machine learning and image-processing pipelines—such as facial emotion recognition, apparent age estimation, and upper-body dress colour detection.

## 3. Project Overview
In real-world human-centered AI systems, predicting sensitive attributes like nationality, race, or ethnicity from facial appearance is scientifically unreliable and ethically harmful. This system enforces an **ethical-by-design** architecture: the user explicitly selects their category (or declares their nationality), and the software routes the image to the exact attribute predictors specified for that category without running redundant models.

## 4. Objectives
- Build a modular, beginner-friendly Python computer vision pipeline using pretrained deep learning models.
- Respect user agency by requiring explicit user selection for nationality/regional categories rather than inferring nationality from facial features.
- Implement conditional execution logic so unnecessary neural networks are skipped when not required.
- Detect upper-body clothing colour accurately by segmenting the person's torso region rather than sampling random background pixels.
- Deliver a responsive, well-validated Streamlit GUI and an automated Pytest unit test suite.

## 5. Features
- **Ethical Nationality Handling:** Supports four user-selected categories (`Indian`, `United States`, `African country/region`, and `Other` with custom text declaration).
- **Conditional Attribute Routing:**
  - **Indian:** Estimated Age + Facial Emotion + Dress Colour
  - **United States:** Estimated Age + Facial Emotion
  - **African country/region:** Facial Emotion + Dress Colour
  - **Other:** User-Declared Nationality + Facial Emotion
- **Facial Emotion Detection:** Uses DeepFace pretrained emotion recognition to report dominant emotion, confidence percentage, and full class probability bars.
- **Apparent Age Estimation:** Uses DeepFace age estimation and clearly labels outputs as approximate estimates rather than exact biological age.
- **Clothing-Aware Dress Colour Detection:** Combines face-anchored torso region-of-interest (ROI) extraction, MediaPipe Selfie Segmentation (with OpenCV GrabCut fallback), skin-pixel filtering in YCrCb space, and K-Means colour clustering in perceptual LAB space.
- **Robust Input Validation & Reset:** Validates file extensions (`JPG`, `JPEG`, `PNG`), file size (up to 10 MB), minimum resolution, missing faces, and obstructed clothing.

## 6. Technologies Used
- **Language:** Python 3.10 – 3.11
- **Web GUI:** Streamlit (`1.41.1`)
- **Deep Learning & Facial Analysis:** DeepFace (`0.0.93`), TensorFlow (`2.18.0`), tf-keras (`2.18.0`)
- **Computer Vision & Segmentation:** OpenCV (`4.10.0.84`), MediaPipe (`0.10.18`)
- **Numerical & Image Utilities:** NumPy (`1.26.4`), Pillow (`11.0.0`)
- **Automated Testing:** Pytest (`8.3.4`)

## 7. Dataset Information
No local training dataset is required because the system uses pretrained inference weights (`DeepFace` FER-2013 emotion weights, `DeepFace` IMDB-WIKI age weights, and `MediaPipe` Selfie Segmentation). See [`DATASET.md`](DATASET.md) for complete details on model sources, licenses, and automatic weight caching.

## 8. System Workflow
1. **Image Upload & Validation:** User uploads a `JPG`, `JPEG`, or `PNG` image. `image_processing.py` verifies integrity, format, and dimensions.
2. **Category Selection:** User selects one of the four categories from the sidebar dropdown (and types their declared nationality if `Other` is selected).
3. **Conditional Routing:** `nationality_logic.py` inspects the selected category and triggers only the required modules.
4. **Attribute Inference:**
   - `age_detection.py` estimates apparent age in years (when required).
   - `emotion_detection.py` detects dominant facial emotion and confidence.
   - `dress_colour.py` isolates torso clothing pixels and extracts the dominant colour name and HEX swatch (when required).
5. **Result Display:** Streamlit renders only the applicable prediction cards alongside the input image preview.

## 9. Project Folder Structure
```text
Nationality_Detection/
│
├── app.py
│
├── modules/
│   ├── __init__.py
│   ├── emotion_detection.py
│   ├── age_detection.py
│   ├── dress_colour.py
│   ├── image_processing.py
│   └── nationality_logic.py
│
├── models/
│   └── README.md
│
├── tests/
│   ├── __init__.py
│   ├── test_emotion.py
│   ├── test_age.py
│   ├── test_dress_colour.py
│   ├── test_image_processing.py
│   └── test_nationality_logic.py
│
├── screenshots/
│   └── README.md
│
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── DATASET.md
├── .gitignore
└── LICENSE
```

## 10. Installation Instructions
1. Extract `Nationality_Detection_Task4.zip` to your local workspace.
2. Open a terminal / command prompt and navigate into the project root folder:
   ```bash
   cd Nationality_Detection
   ```
3. Ensure **Python 3.10 or 3.11** (64-bit) is installed:
   ```bash
   python --version
   ```

## 11. Virtual Environment Setup
Create and activate an isolated virtual environment before installing packages:

- **Windows (Command Prompt / PowerShell):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

## 12. Dependency Installation
Upgrade `pip` and install all required libraries from `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 13. How to Run the Application
Start the Streamlit web server from inside the `Nationality_Detection` directory:
```bash
streamlit run app.py
```
Your default browser will open automatically at `http://localhost:8501`.

> **First-Run Note:** On the very first analysis, DeepFace will automatically download pretrained `.h5` weights (~50–500 MB depending on the active models) into your user cache (`~/.deepface/weights/`). Ensure an active internet connection for the initial run.

## 14. How to Run Tests
Run the automated Pytest suite (uses mocked model inference so tests execute rapidly offline):
```bash
pytest tests/ -v
```

## 15. Expected Output
- **Left Column:** Displays the validated input portrait/upper-body photo with filename and pixel resolution.
- **Right Column:** Displays a confirmation banner and only the cards matching the selected category:
  - *Indian:* Estimated Age (e.g., `26 years`, labeled as approximate), Dominant Emotion + Confidence %, and Dress Colour name + colour swatch.
  - *United States:* Estimated Age and Dominant Emotion + Confidence %.
  - *African country/region:* Dominant Emotion + Confidence % and Dress Colour name + colour swatch.
  - *Other:* User-Declared Nationality and Dominant Emotion + Confidence %.

## 16. Applications
- Ethical human-computer interaction (HCI) demos and computer vision educational labs.
- Retail/fashion analytics tools where users opt in to wardrobe colour feedback.
- Sentiment and customer-experience kiosk prototypes that respect demographic privacy.

## 17. Limitations
- Apparent age and facial emotion estimates depend heavily on lighting, camera angle, occlusion (masks/sunglasses), and image resolution.
- Pretrained models may reflect demographic biases present in public benchmark datasets (FER-2013, IMDB-WIKI).
- Multi-coloured or heavily patterned garments are summarized by their single most dominant clustered colour.
- Close-up headshots without visible upper-body attire cannot yield a dress colour and will display a warning.

## 18. Future Scope
- Support multi-person detection with individual bounding-box selection.
- Add top-2 secondary colour extraction for patterned or traditional attire.
- Export structured JSON/CSV analysis reports directly from the Streamlit interface.
- Containerize the application with Docker for cloud deployment.

## 19. Conclusion
The Nationality-Based Image Attribute Detection System demonstrates how modular computer vision pipelines can combine facial attribute analysis and clothing segmentation inside an intuitive Streamlit interface while upholding strict ethical boundaries against automated nationality or race inference.

## 20. Developer Information
- **Developer:** Priti Vijay Singh
- **Program:** B.E. Computer Science Engineering – Data Science
