
# SignSense - Real-Time Indian Sign Language Recognition System - Year 2024

Welcome to the Real-Time Indian Sign Language Recognition System! This project aims to recognize Indian sign language gestures using advanced machine learning techniques. The system is developed using Python, TensorFlow Object Detection API, SSD model, and MobileNetV2.

## Table of Contents

- [Introduction](#introduction)
- [Tools and Technologies](#tools-and-technologies)
- [Project Flow](#project-flow)
- [Results](#results)

## Introduction

This project focuses on developing a real-time sign language recognition system capable of recognizing specific Indian sign language gestures. The recognized gestures include: A, B, C, D, E, G, and T. The system uses a machine learning model trained with TensorFlow Object Detection API, utilizing an SSD model with MobileNetV2 as the feature extractor.

## Tools and Technologies

- **Programming Language:** Python
- **Framework:** TensorFlow
- **Object Detection API:** TensorFlow Object Detection API
- **Model:** SSD (Single Shot MultiBox Detector)
- **Feature Extractor:** MobileNetV2
- **Image Labeling:** LabelImg
- **Computer Vision:** OpenCV

## Project Flow

1. **Image Collection:**
   - Captured images using an OpenCV script to gather raw data for various hand gestures.

2. **Image Labeling:**
   - Used LabelImg to label the hand signals in the captured images, creating annotated datasets for training.

3. **Model Training:**
   - Trained the model using the annotated dataset. The TensorFlow Object Detection API with SSD and MobileNetV2 was employed for this purpose.

4. **Detection and Recognition:**
   - Developed an output script to use the trained model for real-time detection and recognition of hand signals. The script processes live video feed from a webcam to identify and classify gestures.


## Results

The system achieved a high recognition accuracy for the selected hand gestures, demonstrating its effectiveness in recognizing Indian sign language in real-time.

![Results 1](.images/results1.png)
![Results 2](.images/results2.png)
![Results 3](.images/results3.png)
