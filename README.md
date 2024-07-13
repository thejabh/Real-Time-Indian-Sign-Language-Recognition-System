
# SignSense - Real-Time Indian Sign Language Recognition System - Year 2024

Welcome to the Real-Time Indian Sign Language Recognition System! This project aims to recognize Indian sign language gestures using advanced machine learning techniques. The system is developed using Python, TensorFlow Object Detection API, SSD model, and MobileNetV2.

## Table of Contents

- [Introduction](#introduction)
- [Tools and Technologies](#tools-and-technologies)
- [Project Flow](#project-flow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Indian-Sign-Language-Recognition.git
   cd Indian-Sign-Language-Recognition
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and set up the TensorFlow Object Detection API:**
   Follow the instructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

5. **Download the pre-trained model and configure the training pipeline:**
   Configure the `pipeline.config` file with appropriate paths and parameters.

## Usage

1. **Label Images:**
   - Use LabelImg to annotate the images with the correct labels for each hand gesture.

2. **Train the Model:**
   - Run the training script to train the model using the annotated dataset.
   ```bash
   python train.py --pipeline_config_path=path/to/pipeline.config --model_dir=path/to/model_dir --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr
   ```

3. **Evaluate the Model:**
   - Evaluate the model performance on a separate validation dataset.
   ```bash
   python eval.py --pipeline_config_path=path/to/pipeline.config --model_dir=path/to/model_dir --checkpoint_dir=path/to/model_dir --eval_dir=path/to/eval_dir
   ```

4. **Run the Detection Script:**
   - Use the detection script to recognize hand gestures in real-time.
   ```bash
   python detect.py --model_dir=path/to/model_dir --label_map_path=path/to/label_map.pbtxt --num_classes=7
   ```

## Results

The system achieved a high recognition accuracy for the selected hand gestures, demonstrating its effectiveness in recognizing Indian sign language in real-time.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or additions you would like to see.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or feedback, please contact:

- Name: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]

Thank you for your interest in the Real-Time Indian Sign Language Recognition System! We hope this project can contribute to better communication for individuals using Indian sign language.
