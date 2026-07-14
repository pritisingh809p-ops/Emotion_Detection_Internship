# Facial Emotion Detection System

A Deep Learning-based Facial Emotion Detection System developed using Convolutional Neural Networks (CNN), MTCNN, and Streamlit.

The application detects a human face from an uploaded image, processes the detected facial region, and predicts the facial expression across seven emotion categories.

## Project Overview

Facial Emotion Recognition is an application of Artificial Intelligence and Computer Vision that enables computer systems to analyze facial expressions and classify emotion-related expression patterns.

This project uses a CNN-based deep learning model for emotion classification and MTCNN (Multi-task Cascaded Convolutional Networks) for face detection.

An interactive Streamlit web application allows users to upload an image and view the predicted emotion, confidence score, top emotion predictions, and emotion probability visualization.

## Project Objectives

The main objectives of this project are:

- Detect human faces from uploaded images.
- Automatically crop the detected facial region.
- Preprocess facial images for model prediction.
- Classify facial expressions using a CNN model.
- Predict one of seven emotion categories.
- Display the prediction confidence score.
- Show the top three emotion predictions.
- Visualize emotion probability scores.
- Provide a simple and interactive web interface.

## Features

- Face detection using MTCNN
- Automatic face cropping
- CNN-based emotion classification
- Seven emotion categories
- Emotion confidence score
- Top three emotion predictions
- Emotion probability visualization
- Image upload support
- Interactive Streamlit web application
- Automated facial image preprocessing

## Emotion Classes

The model predicts the following seven facial emotions:

| Class | Emotion |
|------:|---------|
| 1 | Surprise |
| 2 | Fear |
| 3 | Disgust |
| 4 | Happy |
| 5 | Sad |
| 6 | Angry |
| 7 | Neutral |

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| CNN | Facial emotion classification |
| MTCNN | Face detection |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pillow | Image handling |
| Streamlit | Web application interface |
| Jupyter Notebook | Model development and experimentation |

## Dataset

The project uses the RAF-DB (Real-world Affective Faces Database) facial expression dataset.

RAF-DB contains facial images representing different expression categories and is commonly used for facial expression recognition research.

### Emotion Categories

- Surprise
- Fear
- Disgust
- Happy
- Sad
- Angry
- Neutral

### Dataset Usage

The dataset is used for:

1. Facial image collection
2. Emotion label processing
3. Image preprocessing
4. CNN model training
5. Model validation
6. Emotion classification

## Model Workflow

```text
Input Facial Image
        |
        v
Image Upload
        |
        v
Face Detection using MTCNN
        |
        v
Detected Face Cropping
        |
        v
Resize Image to 100 x 100 Pixels
        |
        v
Image Normalization
        |
        v
CNN Model
        |
        v
Emotion Probability Prediction
        |
        v
Highest Probability Emotion Selection
        |
        v
Confidence Score and Top Three Predictions
        |
        v
Emotion Result Visualization
```
## Model Architecture

The project uses a Convolutional Neural Network (CNN) for facial emotion classification.

CNN models are suitable for image classification because they automatically learn important visual features from images.

The model processes facial images and generates probability scores for seven emotion classes.

The emotion class with the highest probability is selected as the final prediction.

### Model Processing Steps

1. Input facial image
2. Image resizing
3. Pixel normalization
4. Feature extraction using convolution layers
5. Feature reduction
6. Emotion classification
7. Probability generation
8. Final emotion prediction

## Face Detection

The project uses MTCNN (Multi-task Cascaded Convolutional Networks) for facial detection.

MTCNN identifies the face location in an uploaded image.

After detecting the face:

1. Face coordinates are extracted.
2. The facial region is cropped.
3. The cropped image is resized.
4. The image is normalized.
5. The processed facial image is passed to the CNN model.

This process allows the classification model to focus primarily on facial features.

## Streamlit Web Application

The project includes an interactive web application developed using Streamlit.

The application allows users to:

- Upload a facial image.
- View the uploaded image.
- Detect the facial region.
- Predict the facial emotion.
- View the confidence score.
- View the top three predicted emotions.
- Analyze emotion probability scores.

## Project Structure

```text
Emotion_Detection/
|
|-- app.py
|   Streamlit emotion detection application
|
|-- model_creation.ipynb
|   CNN model development and training notebook
|
|-- model_a1.json
|   Model-related JSON file
|
|-- requirements.txt
|   Required Python dependencies
|
|-- Application Screenshots.pdf
|   Application output screenshots
|
|-- README.md
|   Project documentation
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pritisingh809p-ops/Emotion_Detection.git
```

### 2. Open the Project Directory

```bash
cd Emotion_Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux or macOS:

```bash
source venv/bin/activate
```

### 5. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The Streamlit application should automatically open in the default web browser.

## Required Dependencies

The main project dependencies include:

```text
streamlit
tensorflow
numpy
opencv-python
Pillow
mtcnn
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```


## System Testing

The application workflow is tested by providing facial images and verifying the complete emotion detection process.

### Functional Test Cases

| Test Case | Expected Result |
|-----------|-----------------|
| Upload facial image | Image should be accepted |
| Detect face | Facial region should be detected |
| Crop face | Detected face should be extracted |
| Preprocess image | Image should be resized and normalized |
| Emotion prediction | Model should return an emotion class |
| Confidence score | Prediction confidence should be displayed |
| Top three predictions | Three highest predictions should be shown |
| Probability visualization | Emotion probabilities should be displayed |

## Application Output

The application displays:

- Uploaded facial image
- Detected facial region
- Predicted emotion
- Prediction confidence
- Top three emotion predictions
- Emotion probability visualization

Google Drive:
https://drive.google.com/file/d/13uVbsAOo1LFRdryj_xi-Q1Gaim-4COZj/view?usp=sharing

Canva:
https://canva.link/3cwn19b6k6e0qiu

## Applications

Facial emotion detection can be applied in:

- Smart education systems
- Human-computer interaction
- Customer experience analysis
- AI-based assistants
- Driver monitoring systems
- Entertainment applications
- User experience research
- Emotion-aware applications

## Limitations

- Prediction performance depends on image quality.
- Poor lighting may affect face detection.
- Partially covered faces may reduce prediction quality.
- Extreme face angles may affect face detection.
- Emotion classification is based only on facial expression patterns.
- Facial expressions may not always represent a person's actual internal emotional state.

## Future Improvements

Future enhancements of the project include:

- Real-time webcam emotion detection
- Video-based emotion analysis
- Improved CNN model performance
- Data augmentation
- Transfer learning
- Cloud deployment
- Mobile application integration
- Voice emotion detection
- Multimodal emotion recognition

## Conclusion

The Facial Emotion Detection System demonstrates the practical application of Deep Learning and Computer Vision for facial expression classification.

The project combines CNN-based emotion classification, MTCNN face detection, image preprocessing, and Streamlit application development to create an interactive emotion recognition system.

The system detects facial regions from uploaded images and predicts emotion categories while displaying confidence and probability information.

This project demonstrates practical knowledge of Artificial Intelligence, Deep Learning, Computer Vision, image processing, and web application development.

## Developer

**Priti Vijay Singh**

B.E. Computer Science Engineering  
Specialization in Data Science

## Disclaimer

This project is developed for educational and internship purposes.

The emotion predictions are based on facial expression patterns identified by the trained model and should not be interpreted as psychological or medical assessments.