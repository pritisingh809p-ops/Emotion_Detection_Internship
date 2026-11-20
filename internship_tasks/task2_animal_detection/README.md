# Animal Detection and Classification System

## 1. Project Overview

The **Animal Detection and Classification System** is a machine learning and computer vision project developed using Python. It is designed to detect and classify different animals in images and videos.

The system can identify multiple animals in a single frame, distinguish between different animal species, and highlight carnivorous animals using red bounding boxes.

It also displays a pop-up message showing the total number of detected carnivorous animals.

The project provides a Graphical User Interface (GUI) that allows users to upload images or videos and view the detection results.

## 2. Objectives

- To develop a machine learning model for animal detection.
- To identify and classify different animal species.
- To detect multiple animals in a single image or video frame.
- To highlight carnivorous animals with red bounding boxes.
- To display the number of detected carnivorous animals.
- To provide a user-friendly GUI for image and video detection.
- To display the input image or video along with the detection results.

## 3. Features

- **Image Detection:** Upload an image and detect animals present in it.
- **Video Detection:** Upload a video and detect animals in its frames.
- **Multiple Animal Detection:** Identify multiple animals within one frame.
- **Species Classification:** Classify detected animals into different species.
- **Carnivore Highlighting:** Highlight detected carnivorous animals using red bounding boxes.
- **Carnivore Count:** Display a pop-up message showing the number of detected carnivorous animals.
- **Preview:** Display the image or video after detection.
- **GUI:** Simple interface for uploading images and videos.

## 4. Technologies Used

- Python
- OpenCV
- NumPy
- Machine Learning / Deep Learning
- TensorFlow or PyTorch (depending on the model used)
- Streamlit or Tkinter (depending on the GUI used)

## 5. Project Structure

```text
Animal_Detection/
│
├── app.py
├── model_training.py
├── animal_detection.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── train/
│   └── validation/
│
├── models/
│   └── trained_model
│
├── images/
│   └── sample_animal.jpg
│
├── videos/
│   └── sample_video.mp4
│
└── screenshots/
```

*Note: The folder structure is an example. Modify it according to your actual project files.*

## 6. Working Procedure

1. The user opens the application.
2. The user selects image detection or video detection.
3. The user uploads an image or video.
4. The system processes the uploaded file using the trained model.
5. The model detects animals and identifies their species.
6. Multiple animals are detected within the same frame, if present.
7. Carnivorous animals are highlighted using red bounding boxes.
8. The system counts the detected carnivorous animals.
9. A pop-up message displays the carnivore count.
10. The final detection result is displayed in the GUI.

## 7. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open the Project Folder

```bash
cd Animal_Detection
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

If the project uses Streamlit:

```bash
streamlit run app.py
```

If the project uses Tkinter or a normal Python application:

```bash
python app.py
```

## 8. Input

The system accepts:

- Images: JPG, JPEG, PNG
- Videos: MP4, AVI, and other supported video formats

## 9. Expected Output

The application displays:

- Uploaded image or video preview.
- Detected animals with bounding boxes.
- Names of detected animal species.
- Carnivorous animals highlighted in red.
- Total number of detected carnivorous animals.
- Final processed image or video preview.

## 10. Applications

- Wildlife monitoring.
- Forest and animal conservation.
- Zoo monitoring systems.
- Wildlife research.
- Animal behaviour analysis.
- Automated animal identification.

## 11. Future Scope

- Improve animal detection accuracy.
- Support more animal species.
- Enable real-time animal detection using a webcam.
- Improve detection in low-light conditions.
- Add animal tracking in videos.
- Develop a mobile application.

## 12. Author

**Priti Vijay Singh**

B.E. Computer Science and Engineering (Data Science)

**Internship:** InAmigos Foundation

## 13. License

This project is developed for educational and internship purposes.
