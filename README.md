## Overview
This repository contains the implementation of an automatic license plate detection system that leverages YOLOv8 for object detection and PyTesseract for OCR. The model has been fine-tuned on a custom dataset to detect car license plates and extract the corresponding number plates. The trained model is deployed via a Streamlit interface to enable real-time detection and recognition.

## Key Steps:
Dataset Preprocessing: Conversion of pre-annotated XML files to YOLO format.
Model Training: Fine-tuning a pre-trained YOLOv8 nano model.
OCR Integration: Using PyTesseract for text extraction from detected license plates.
Deployment: Real-time inference on images and video using Streamlit.

## Features
Real-time license plate detection using YOLOv8, optimized for fast inference.
OCR-powered text extraction from detected license plates using PyTesseract.
Fine-tuned YOLOv8 model with custom dataset and high accuracy metrics.
Streamlit-based deployment for interactive and real-time inference on images and video.
Mean Average Precision (mAP) evaluation for model performance assessment.

## Tech Stack
YOLOv8 (PyTorch): Object detection framework for training the model and detecting license plates.
PyTesseract: Optical Character Recognition (OCR) library for extracting text from detected license plates.
Streamlit: Web application framework for deploying real-time detection and OCR.
OpenCV: Library for image processing and video handling.
Numpy, Matplotlib: For data manipulation and visualization.

## Dataset
Dataset: 433 pre-annotated images containing vehicle license plates.
Annotations: Pre-annotated XML files containing bounding box coordinates and class labels.
Format Conversion: The dataset was processed and converted from XML format to YOLOv8-compatible text files containing bounding box coordinates and labels.

## Metrics
Mean Average Precision (mAP):
  mAP@50: 0.907
  mAP@50-95: 0.539
These metrics were obtained after training for 100 epochs on the custom dataset.
