# Dataset Documentation

## 1. Is a Local Training Dataset Required?
**No.** This application performs inference using **pretrained computer vision models** (`DeepFace` and `MediaPipe` / `OpenCV`). You do not need to download or store any large image dataset to run the Streamlit application or execute the automated test suite.

## 2. Pretrained Models & Underlying Training Datasets
Although no local dataset is bundled, the pretrained models used during inference were originally trained by their respective authors on public academic benchmarks:

| Component | Model / Backend | Original Training Dataset | Weight Retrieval | License |
| :--- | :--- | :--- | :--- | :--- |
| **Facial Emotion Detection** | DeepFace Emotion CNN | **FER-2013** (Facial Expression Recognition 2013) | Automatically downloaded to `~/.deepface/weights/` on first inference | MIT (DeepFace) |
| **Apparent Age Estimation** | DeepFace Age Model (DEX / VGG-Face backbone) | **IMDB-WIKI** (Celebrity face images with age labels) | Automatically downloaded to `~/.deepface/weights/` on first inference | MIT (DeepFace) |
| **Person / Clothing Segmentation** | MediaPipe Selfie Segmentation (Landscape model) | Google internal / public portrait segmentation benchmarks | Bundled inside the `mediapipe` package | Apache 2.0 |
| **Face Localization Helper** | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) | OpenCV positive/negative frontal face samples | Bundled inside `opencv-python-headless` | Apache 2.0 |

## 3. Ethical Handling of Nationality Data
- **No Facial Ethnicity/Nationality Dataset Used:** This project deliberately avoids datasets or classifiers that attempt to predict race, ethnicity, or nationality from facial features.
- **User-Declared Input:** Nationality/regional category (`Indian`, `United States`, `African country/region`, or `Other` with custom text) is supplied directly by the user at runtime.

## 4. How to Add a Custom Dataset for Future Fine-Tuning
If you extend this internship project in the future to fine-tune a custom clothing classifier or emotion recognizer:
1. Create a local directory `data/raw/` (add `data/` to `.gitignore` so large image files are not committed to Git).
2. Organize images into class subfolders (e.g., `data/raw/happy/`, `data/raw/neutral/`).
3. Only use datasets with explicit open licenses (e.g., Creative Commons or MIT/Apache academic releases) and obtain informed consent for any newly captured portraits.
