# Dataset and Model Information

## Object detector
This project uses Ultralytics YOLOv8n with pretrained COCO weights (`yolov8n.pt`). The model is downloaded automatically by Ultralytics on first execution and cached locally. The COCO class IDs used are:
- `0`: person
- `2`: car

## Car colour estimation
The detected car crop is converted from RGB to HSV. The program estimates a dominant hue/saturation/value and maps it to a practical colour label such as blue, red, green, yellow, orange, purple, white, grey or black. This is a heuristic and not a separately trained machine-learning model.

## Dataset
No private or custom dataset is bundled. The detector uses pretrained COCO weights. For a custom-trained version, collect and label traffic images with bounding boxes for cars and people, split them into training/validation/test sets, and document the source and licence of each dataset.

## Reproducibility
Record the Python version, package versions, model weights version, confidence threshold and test image sources when reporting evaluation results. Do not claim accuracy or training results unless measured on a labelled test set.
