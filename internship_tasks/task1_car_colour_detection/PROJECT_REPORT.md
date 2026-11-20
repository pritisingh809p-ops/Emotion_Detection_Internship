# Project Report: Car Colour Detection System

## 1. Abstract
The Car Colour Detection System is a computer-vision application designed to analyse a traffic image, detect cars and people, estimate the visible colour of each detected car, and display object counts. Blue cars are highlighted with red rectangles, while other cars are highlighted with blue rectangles. The application provides an interactive graphical interface and an annotated preview.

## 2. Problem Statement
Manually identifying and counting cars by colour in traffic images is time-consuming. This project demonstrates an automated approach to detecting cars and people and summarising the results from an image.

## 3. Objectives
- Detect multiple cars in one traffic image.
- Detect people visible in the traffic image.
- Estimate the dominant colour of each detected car.
- Highlight blue cars using red rectangles.
- Highlight other cars using blue rectangles.
- Display the total number of detected cars and people.
- Provide a GUI with original and processed image previews.

## 4. Tools and Technologies
Python, Streamlit, Ultralytics YOLOv8n, OpenCV, NumPy, Pillow and Pytest.

## 5. Methodology
The user uploads an image through the Streamlit interface. YOLOv8n identifies objects belonging to the COCO `car` and `person` classes. For every detected car, the bounding-box region is cropped and passed to an HSV-based colour estimator. The system then draws the required coloured rectangle and label. Finally, the interface displays the annotated image and counts.

## 6. Colour Highlighting Logic
- Blue car: red rectangle.
- Any other estimated car colour: blue rectangle.
- Person: blue rectangle.

The colour estimate is based on the dominant HSV values in the detected car crop and is an approximation.

## 7. Output
The application reports:
- Total detected cars.
- Total detected people.
- Number of detected blue cars.
- Estimated colour and confidence for each detected car.
- Annotated image preview and downloadable result.

## 8. Testing
Unit tests check blue and red colour estimation and verify that drawing detections preserves the image dimensions. Functional testing should also be performed with different traffic images, lighting conditions, vehicle colours, and image resolutions.

## 9. Limitations
The detector uses pretrained YOLOv8n weights and does not train a new detector from scratch. Colour classification uses an HSV heuristic rather than a separately trained machine-learning colour model. Performance can be affected by shadows, reflections, occlusion, small objects and image quality.

## 10. Future Scope
- Add a custom-labelled traffic dataset and train/fine-tune a detector.
- Train a dedicated car-colour classification model.
- Add video and live-camera processing.
- Improve colour estimation using vehicle-region segmentation.
- Evaluate precision, recall, mAP and colour-classification accuracy on a labelled test set.

## 11. Conclusion
The project demonstrates an end-to-end image-based traffic analysis workflow using object detection, colour estimation, annotation and a Streamlit GUI. It provides car and people counts and the specified blue-car highlighting behaviour.
