# Car Colour Detection System

A Python computer-vision application that detects cars and people in an uploaded traffic image, estimates each car's dominant colour, counts detected cars and people, and highlights blue cars with red rectangles. Other detected cars are highlighted with blue rectangles. The application includes an interactive Streamlit GUI and an annotated-image preview.

## Features
- Detects cars and people in traffic images using YOLOv8n (COCO-pretrained object detector).
- Estimates the dominant visible car colour using HSV colour analysis.
- Draws **red rectangles around blue cars** and **blue rectangles around other cars**.
- Displays total car count, blue-car count and people count.
- Shows original and annotated image previews.
- Displays confidence values and provides an annotated-image download.

## Technology
Python, Streamlit, Ultralytics YOLO, OpenCV, NumPy, Pillow and Pytest.

## Project workflow
1. Upload a traffic image.
2. YOLO detects cars and people.
3. For each detected car, the application crops the bounding box and estimates its dominant colour.
4. Blue cars receive red bounding boxes; other cars receive blue bounding boxes. People receive blue bounding boxes.
5. The GUI displays the annotated preview and counts.

## Requirements
- Windows 10/11, Linux or macOS
- Python 3.9 or newer (Python 3.10/3.11 recommended)
- Internet access on first launch to download `yolov8n.pt` automatically

## Setup (Windows PowerShell)
Open a terminal in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

If PowerShell blocks environment activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run tests
```powershell
pytest -q
```

## Project structure
```text
Car_Colour_Detection_Project/
├── app.py
├── car_detector.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── DATASET_AND_MODEL.md
├── .gitignore
├── setup.bat
├── run_app.bat
└── tests/
    └── test_car_detector.py
```

## Important implementation note
This version uses the publicly available YOLOv8n model pretrained on the COCO dataset for detecting the `car` and `person` classes. It does **not** train a new object-detection model from scratch. Car colour is estimated separately with an HSV-based computer-vision heuristic, not a learned colour classifier. This makes the project lightweight and easier to run for a demonstration. If the internship rubric strictly requires training your own model, a separately trained and evaluated model should be added and documented.

## Limitations
Colour estimation is approximate and can be affected by sunlight, shadows, reflections, tinted windows, vehicle paint, small bounding boxes and camera white balance. The detector may miss distant or occluded cars/people. Counts represent detections in the uploaded still image, not unique vehicles over time.

## Developer
Priti Vijay Singh  
B.E. Computer Science Engineering (Data Science)

## Disclaimer
For educational and demonstration purposes. Detection and colour labels are estimates and should not be used for traffic enforcement or safety-critical decisions.
