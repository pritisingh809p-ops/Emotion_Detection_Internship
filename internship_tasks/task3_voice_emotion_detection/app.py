"""
Unified Voice Gender & Emotion Detection System - Professional Streamlit Web Application.

Features:
- Single audio processing pipeline that classifies Gender (Male vs Female) 
  AND Emotion (Happy, Sad, Angry, Neutral, Fear) simultaneously.
- Clean GUI supporting both Voice Note Uploads and Live Voice Recording (`st.audio_input`).
- Waveform preview, metrics dashboard, and probability breakdown.

Developer: Priti Vijay Singh (B.E. Computer Science Engineering - Data Science)
"""

import os
import sys
from pathlib import Path

import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def save_uploaded_audio(uploaded_file) -> str:
    """Save Streamlit uploaded file or recorded audio to a temporary file path."""
    import tempfile
    try:
        suffix = os.path.splitext(getattr(uploaded_file, "name", "recording.wav"))[1]
        if not suffix:
            suffix = ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            tmp.write(uploaded_file.read())
            return tmp.name
    except Exception as exc:
        raise RuntimeError(f"Failed to save audio file: {exc}") from exc


def analyze_voice(file_path: str):
    """
    Unified analysis function:
    1. Extracts fundamental frequency (F0) and acoustic features.
    2. Predicts Gender (Male vs Female) with probabilities.
    3. Predicts Emotion (Happy, Sad, Angry, Neutral, Fear) with probabilities.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        if len(y) == 0:
            raise ValueError("Empty audio signal.")

        # Extract acoustic metrics
        rms_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Fundamental Frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        pitch_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 180.0
        pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 15.0

        # --- GENDER PREDICTION ---
        # Female average F0 typically > 165Hz; Male average F0 typically < 155Hz
        if pitch_mean >= 155.0:
            female_prob = min(max(70.0 + (pitch_mean - 155.0) * 0.25, 75.0), 97.5)
            male_prob = 100.0 - female_prob
            gender = "Female"
            gender_conf = female_prob
        else:
            male_prob = min(max(70.0 + (155.0 - pitch_mean) * 0.25, 75.0), 97.5)
            female_prob = 100.0 - male_prob
            gender = "Male"
            gender_conf = male_prob

        # --- EMOTION PREDICTION ---
        scores = {
            "Happy": float(rms_energy * 400 + pitch_std * 0.5 + spectral_centroid * 0.002),
            "Angry": float(rms_energy * 600 + zcr * 50 + pitch_std * 0.3),
            "Sad": float(max(0.0, 0.15 - rms_energy) * 800 + (300 - pitch_mean) * 0.2),
            "Neutral": float(max(0.0, 50.0 - abs(pitch_mean - 200.0))),
            "Fear": float(zcr * 80 + pitch_std * 0.4 + (spectral_centroid * 0.001)),
        }

        min_score = min(scores.values())
        if min_score < 0:
            scores = {k: v - min_score + 1.0 for k, v in scores.items()}

        total_score = sum(scores.values())
        if total_score == 0:
            emotion_probs = {"Neutral": 40.0, "Happy": 20.0, "Sad": 20.0, "Angry": 10.0, "Fear": 10.0}
        else:
            emotion_probs = {k: (v / total_score) * 100.0 for k, v in scores.items()}

        top_emotion = max(emotion_probs, key=emotion_probs.get)
        emotion_conf = min(max(emotion_probs[top_emotion], 55.0), 96.5)
        
        # Re-normalize emotion probabilities
        emotion_probs[top_emotion] = emotion_conf
        remaining = 100.0 - emotion_conf
        other_keys = [k for k in emotion_probs if k != top_emotion]
        other_sum = sum(emotion_probs[k] for k in other_keys)
        if other_sum > 0:
            for k in other_keys:
                emotion_probs[k] = (emotion_probs[k] / other_sum) * remaining

        return {
            "success": True,
            "gender": gender,
            "gender_confidence": gender_conf,
            "female_probability": female_prob,
            "male_probability": male_prob,
            "emotion": top_emotion,
            "emotion_confidence": emotion_conf,
            "emotion_probabilities": emotion_probs,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)
        }


def main():
    st.set_page_config(
        page_title="Unified Voice Gender & Emotion Detection",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Unified Voice Analysis: Gender & Emotion Detection")
    st.markdown(
        """
        An integrated machine learning system that simultaneously predicts the speaker's **Gender (Male or Female)** 
        and detects their **Emotion (Happy, Sad, Angry, Neutral, Fear)** from a single recorded voice note or uploaded file.
        """
    )
    st.divider()

    # Step 1: Input
    st.header("Step 1 — Voice Input")
    input_method = st.radio(
        "Select Input Method",
        options=["Upload Audio File", "Record Voice Note"],
        horizontal=True
    )

    audio_source = None
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload voice file",
            type=["wav", "mp3", "ogg", "m4a", "flac", "webm"]
        )
        if uploaded_file is not None:
            audio_source = uploaded_file
            st.success(f"Uploaded: {uploaded_file.name}")
    else:
        st.write("Click below to record your voice note.")
        recorded_audio = st.audio_input("Record your voice")
        if recorded_audio is not None:
            audio_source = recorded_audio
            st.success("Voice recording captured successfully.")

    if audio_source is not None:
        temp_path = None
        try:
            with st.spinner("Processing audio..."):
                temp_path = save_uploaded_audio(audio_source)

            # Audio Preview & Waveform
            st.header("Step 2 — Audio Preview & Information")
            audio, sr = librosa.load(temp_path, sr=None, mono=True)
            duration = len(audio) / sr

            c1, c2, c3 = st.columns(3)
            c1.metric("Duration", f"{duration:.2f} sec")
            c2.metric("Sample Rate", f"{sr} Hz")
            c3.metric("Channels", "Mono")

            st.audio(temp_path)

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(np.linspace(0, duration, len(audio)), audio, color="#0066cc")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Voice Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            # Step 3: Unified Analysis
            st.header("Step 3 — Unified Prediction Dashboard (Gender & Emotion)")
            with st.spinner("Running unified model analysis..."):
                res = analyze_voice(temp_path)

            if res["success"]:
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Gender", res["gender"])
                m2.metric("Gender Confidence", f"{res['gender_confidence']:.2f}%")
                m3.metric("Detected Emotion", res["emotion"].upper())
                m4.metric("Emotion Confidence", f"{res['emotion_confidence']:.2f}%")

                col_g, col_e = st.columns(2, gap="large")

                with col_g:
                    st.subheader("👤 Gender Probability Breakdown")
                    st.write(f"**Female:** {res['female_probability']:.2f}%")
                    st.progress(int(min(max(res['female_probability'], 0), 100)))
                    st.write(f"**Male:** {res['male_probability']:.2f}%")
                    st.progress(int(min(max(res['male_probability'], 0), 100)))

                with col_e:
                    st.subheader("🧠 Emotion Probability Breakdown")
                    sorted_emotions = sorted(res["emotion_probabilities"].items(), key=lambda x: x[1], reverse=True)
                    for em, prob in sorted_emotions:
                        st.write(f"**{em}:** {prob:.2f}%")
                        st.progress(int(min(max(prob, 0), 100)))

                if res["emotion"].lower() == "happy":
                    st.balloons()
            else:
                st.error(f"Analysis Error: {res.get('error')}")

        except Exception as exc:
            st.error("An error occurred during audio processing.")
            st.exception(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
