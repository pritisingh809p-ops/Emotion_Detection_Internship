# Task 3 – Voice Emotion Detection

## Project Overview

This project is developed as part of the Emotion Detection internship project.

The objective of this task is to develop a voice-based emotion detection system that can:

- Accept uploaded voice notes.
- Record voice directly through the GUI.
- Verify whether the uploaded/recorded voice is female.
- Reject non-female voices.
- Detect emotions from female voices.
- Display the predicted emotion and confidence.
- Identify a happy voice as "Happy".

The application uses Machine Learning and Deep Learning techniques for audio classification.

---

## Main Features

### 1. Voice Upload

Users can upload a voice/audio file through the Streamlit GUI.

Supported audio formats include:

- WAV
- MP3
- M4A
- OGG
- FLAC

### 2. Voice Recording

The application also supports direct voice recording through the GUI.

### 3. Female Voice Verification

Before emotion detection, the system checks the voice category.

The system is designed to work only with female voices.

If a male/non-female voice is detected, the application displays a warning asking the user to upload or record a female voice.

### 4. Emotion Detection

If the voice is detected as female, the audio is passed to the emotion detection model.

The system predicts the emotion and displays the confidence.

### 5. Happy Voice Detection

If the predicted emotion is happy, the GUI displays a special happy-result message.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Streamlit
- Pydub
- FFmpeg
- Scikit-learn

---

## Project Structure

```text
task3_voice_emotion_detection/
│
├── app.py
├── voice_emotion_detection.py
├── audio_preprocessing.py
├── female_voice_detector.py
├── utils.py
├── README.md
├── requirements.txt
└── __init__.py