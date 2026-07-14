Emotion Detection Using Deep Learning

Project Report

Submitted By: Priti Vijay Singh
Branch: Computer Science Engineering – Data Science
Project: Emotion Detection System

1. Abstract

Emotion detection is an important application of Artificial Intelligence, Deep Learning, and Computer Vision. This project presents an Emotion Detection System designed to recognize human facial emotions from uploaded images.

The system uses a Convolutional Neural Network (CNN) for emotion classification and Multi-task Cascaded Convolutional Networks (MTCNN) for facial detection. The RAF-DB facial expression dataset is used for model development and training.

The detected face is preprocessed and passed to the trained CNN model. The model classifies the facial expression into one of seven emotion categories: Surprise, Fear, Disgust, Happy, Sad, Angry, and Neutral.

A user-friendly web application is developed using Streamlit. The application allows users to upload facial images and view the predicted emotion, confidence score, and top emotion predictions.

The project demonstrates the practical application of deep learning in facial expression recognition and human-computer interaction.

2. Introduction

Human emotions play an important role in communication and decision-making. Facial expressions are one of the most common ways through which people express their emotions.

Emotion Detection, also known as Facial Expression Recognition, is a computer vision technique used to automatically identify emotions from facial images.

Recent advancements in deep learning have significantly improved image classification and facial expression recognition. Convolutional Neural Networks are particularly effective for extracting facial features from images.

This project develops an automated Emotion Detection System using CNN and MTCNN. MTCNN detects the human face from an image, while the CNN model analyzes facial features and predicts the corresponding emotion.

The system is integrated into a Streamlit web application to provide an interactive and simple user interface.

3. Problem Statement

Manual identification of human emotions can be subjective and may vary depending on individual interpretation.

Traditional computer systems cannot naturally understand human facial expressions without the use of Artificial Intelligence and Computer Vision.

Therefore, there is a need for an automated system capable of detecting human faces and classifying facial expressions into predefined emotion categories.

The proposed system aims to analyze facial images and automatically predict the most probable emotion using deep learning techniques.

4. Project Objectives

The main objectives of this project are:

To develop an automated facial emotion detection system.
To detect human faces from uploaded images using MTCNN.
To preprocess facial images for deep learning-based classification.
To develop a CNN-based emotion classification model.
To classify facial expressions into seven emotion categories.
To display prediction confidence scores.
To display the top predicted emotions.
To develop an interactive web application using Streamlit.
To implement testing for important project components.
To study the application of deep learning in facial expression recognition.
5. Dataset Description – RAF-DB

The Real-world Affective Faces Database, commonly known as RAF-DB, is a facial expression dataset used for emotion recognition research.

The dataset contains facial images representing different human expressions under real-world conditions.

The project uses seven basic emotion categories:

Class	Emotion
1	Surprise
2	Fear
3	Disgust
4	Happy
5	Sad
6	Angry
7	Neutral

RAF-DB contains variations in facial appearance, lighting conditions, head pose, age, and expression intensity.

These variations make the dataset suitable for developing facial emotion recognition models.

The dataset is stored separately from the GitHub repository because of its size and dataset distribution considerations.

6. Data Preprocessing

Data preprocessing is an important step in the emotion detection pipeline.

The following preprocessing operations are performed:

Face Detection

The MTCNN face detector identifies the facial region from the input image.

Face Cropping

The detected facial region is extracted from the original image.

Image Resizing

The cropped facial image is resized to:

100 × 100 pixels

This ensures that all input images have the same dimensions.

Pixel Normalization

Image pixel values are normalized to improve model processing and numerical stability.

The normalized pixel value can be represented as:

Normalized Pixel = Pixel Value / 255

Therefore, the pixel values are converted from the range 0–255 to 0–1.

Array Transformation

The processed image is converted into a NumPy array and reshaped into the format expected by the CNN model.

The preprocessing process ensures that all facial images are provided to the model in a consistent format.

7. CNN Architecture

A Convolutional Neural Network is used for facial emotion classification.

CNNs are deep learning models specifically designed for processing image data.

The CNN model learns important facial features such as:

Eye shape
Eyebrow position
Mouth movement
Facial muscle patterns
Expression intensity

The CNN architecture consists of different layers.

Convolutional Layers

Convolutional layers extract important features from facial images.

These layers learn visual patterns such as edges, curves, textures, and facial structures.

Activation Function

Activation functions introduce non-linearity into the neural network.

The ReLU activation function is commonly used in hidden layers.

ReLU(x) = max(0, x)
Pooling Layers

Pooling layers reduce the spatial dimensions of feature maps.

This reduces computational complexity and helps the model focus on important features.

Flatten Layer

The extracted feature maps are converted into a one-dimensional feature vector.

Dense Layers

Fully connected dense layers process the extracted features and perform emotion classification.

Output Layer

The final output layer contains seven neurons representing the seven emotion categories.

The model generates probability scores for each emotion.

The emotion with the highest probability is selected as the final prediction.

8. MTCNN Face Detection

MTCNN stands for Multi-task Cascaded Convolutional Networks.

It is a deep learning-based face detection technique used to identify human faces in images.

MTCNN uses a cascaded architecture consisting of three networks:

P-Net

The Proposal Network generates possible facial regions.

R-Net

The Refine Network removes incorrect facial regions and improves face detection.

O-Net

The Output Network produces the final face bounding box and facial landmark information.

In this project, MTCNN is used before emotion classification.

The workflow is:

Input Image
     ↓
MTCNN Face Detection
     ↓
Face Cropping
     ↓
Image Resizing
     ↓
CNN Model
     ↓
Emotion Prediction

Using face detection helps the CNN model focus on the facial region instead of the complete image.

9. Model Training

The CNN model is trained using facial images from the RAF-DB dataset.

The general model training process includes:

Dataset Loading
       ↓
Image Preprocessing
       ↓
Training Data Preparation
       ↓
CNN Model Creation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Saving

During training, the model processes facial images and compares predicted emotions with actual emotion labels.

The model calculates the prediction error using a loss function.

The optimization algorithm updates model parameters to reduce the prediction error.

The training process is repeated for multiple epochs.

After training, the model architecture and required model components are saved for use in the Streamlit application.

10. Emotion Classification

The trained CNN model performs emotion classification.

The system supports seven emotions:

Surprise
Fear
Disgust
Happy
Sad
Angry
Neutral

The model generates a probability score for every emotion.

Example:

Happy      : 85%
Neutral    : 7%
Surprise   : 4%
Sad        : 2%
Angry      : 1%
Fear       : 0.5%
Disgust    : 0.5%

The emotion with the highest probability is selected as the predicted emotion.

In the above example, the detected emotion is:

Happy

The application also displays prediction confidence information to help users understand the model output.

11. Application Development Using Streamlit

Streamlit is used to develop the web-based user interface for the Emotion Detection System.

The application provides a simple interface where users can upload facial images.

The application workflow is:

User Opens Application
        ↓
Uploads Facial Image
        ↓
Image Validation
        ↓
MTCNN Detects Face
        ↓
Face Preprocessing
        ↓
CNN Emotion Prediction
        ↓
Display Result

The application displays:

Uploaded image
Detected emotion
Confidence score
Top emotion predictions

Streamlit makes the machine learning application interactive and accessible through a web interface.

12. Testing Strategy

Testing is performed to verify the reliability and correctness of important project components.

The testing strategy focuses on:

Image preprocessing
Image dimensions
Pixel normalization
Face detection handling
Emotion classes
Prediction logic
Model output structure

Automated testing can be implemented using the Pytest framework.

The testing process ensures that individual project functions behave as expected.

13. Test Cases
Test Case	Description	Expected Result
TC01	Upload valid facial image	Image accepted
TC02	Preprocess facial image	Image resized to 100 × 100
TC03	Check normalization	Pixel values between 0 and 1
TC04	Detect face	Facial region detected
TC05	Image without face	Appropriate handling
TC06	Check emotion classes	Seven emotion classes available
TC07	CNN prediction	Probability output generated
TC08	Emotion selection	Highest probability emotion selected
TC09	Confidence display	Prediction confidence displayed
TC10	Streamlit application	Application loads successfully
Automated Test Example
def test_emotion_classes():
    emotions = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happy",
        "Sad",
        "Angry",
        "Neutral"
    ]

    assert len(emotions) == 7

The test verifies that the application contains seven emotion classes.

14. Results and Evaluation

The developed system successfully integrates facial detection and emotion classification into a web-based application.

The MTCNN component detects facial regions from uploaded images.

The detected facial image is processed and passed to the CNN model.

The CNN model generates probability scores for seven emotion categories.

The Streamlit application displays the predicted emotion and confidence information.

The system demonstrates the successful integration of:

Computer Vision
+
Deep Learning
+
Face Detection
+
Emotion Classification
+
Web Application
Model Evaluation

The model should be evaluated using metrics such as:

Training Accuracy
Validation Accuracy
Test Accuracy
Training Loss
Validation Loss
Confusion Matrix

Important: Add your actual model accuracy and evaluation values from model_creation.ipynb. Do not add fake accuracy values.

Example format:

Training Accuracy   : [Add Actual Value]
Validation Accuracy : [Add Actual Value]
Test Accuracy       : [Add Actual Value]
15. Application Screenshots

Application screenshots demonstrate the working of the Emotion Detection System.

The screenshots may include:

Streamlit application home page.
Image upload interface.
Uploaded facial image.
Detected emotion result.
Confidence score.
Top three emotion predictions.
Different emotion detection examples.

The project repository contains an Application Screenshots.pdf file containing application output screenshots.

16. Limitations

The current system has the following limitations:

Emotion prediction depends on facial image quality.
Poor lighting may affect prediction.
Extreme head poses may reduce model performance.
Partially covered faces may be difficult to analyze.
Emotion expressions may vary between individuals.
Similar emotions may be difficult to distinguish.
The system analyzes facial expressions and cannot determine a person's actual internal emotional or mental state.
Model performance depends on the training dataset.
17. Future Scope

The project can be improved in several ways.

Future improvements include:

Real-time webcam-based emotion detection.
Mobile application development.
Cloud deployment.
Improved CNN architecture.
Transfer learning using advanced deep learning models.
Voice emotion recognition.
Text emotion analysis.
Multimodal emotion detection.
Multiple-face emotion detection.
Improved model accuracy using larger datasets.

A multimodal emotion recognition system could combine:

Facial Expression
       +
Voice Analysis
       +
Text Analysis
       ↓
Advanced Emotion Recognition
18. Conclusion

The Emotion Detection System demonstrates the practical application of Artificial Intelligence, Deep Learning, and Computer Vision in facial expression recognition.

The project uses MTCNN to detect human faces and a Convolutional Neural Network to classify facial expressions into seven emotion categories.

The Streamlit framework provides an interactive interface for uploading facial images and viewing emotion predictions.

The project demonstrates a complete machine learning workflow involving dataset preparation, image preprocessing, deep learning model development, face detection, emotion classification, testing, and application development.

The system can be further enhanced using advanced deep learning architectures and multimodal emotion recognition techniques.

19. References
RAF-DB – Real-world Affective Faces Database.
TensorFlow Documentation.
Keras Deep Learning Documentation.
OpenCV Documentation.
MTCNN Face Detection.
Streamlit Documentation.
NumPy Documentation.
Convolutional Neural Networks for Image Classification.
Facial Expression Recognition using Deep Learning.