# Dataset Documentation - Sign Language Detection

## 1. Overview
This project uses **MediaPipe Hands** landmark extraction combined with heuristic and machine learning classification models to recognize known sign language vocabulary (`HELLO`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `HELP`, `GOODBYE`). No massive local dataset is required to run inference.

## 2. Model Sources & Weights
- **MediaPipe Hands:** Pretrained palm detection and hand landmark regression pipeline provided by Google MediaPipe under the Apache 2.0 License.
- **Vocabulary Classification:** Rule-based and lightweight spatial geometry classifiers mapped against standard sign language lexicons.
