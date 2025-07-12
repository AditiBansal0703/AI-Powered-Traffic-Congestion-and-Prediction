# AI-powered-traffic-congestion-detection-and-prediction
This project leverages Computer Vision and Deep Learning to automatically detect traffic congestion levels and accidents from video footage. It combines YOLOv8 segmentation, occupancy ratio logic, and image classification models to assist urban planners, law enforcement, and citizens in improving road safety and traffic flow.

![WhatsApp Image 2025-07-11 at 02 27 33_2d24f18b](https://github.com/user-attachments/assets/e88418d6-a267-4e68-9688-c3db56225007)

##  Features

- Vehicle detection and segmentation using *YOLOv8*
- Occupancy ratio calculation to measure congestion
- Classification of congestion level: *Low, **Moderate, **High, **Unusual*
- Accident detection using a *CNN-based classifier (MobileNetV2)*
- Web-based UI built with *Flask*
- Upload traffic videos and get instant predictions

  ##  Project Objective

- To *automatically detect* traffic congestion levels using AI-based video analysis.
- To *classify and highlight* accident-prone or anomalous traffic conditions.
- To build a *deployed and interactive web system* for real-time traffic monitoring.

###  Web & Backend
- *Python 3.10+*
- *Flask 3.1*
##  Demo

*Upload a traffic video to instantly detect congestion levels and identify possible accidents using our AI-powered system.*

![WhatsApp Image 2025-07-11 at 02 33 06_a876306b](https://github.com/user-attachments/assets/dbf3c3b1-2a26-4906-81b8-82fddedd96cb)

*Predicted congestion level and accident status based on real-time AI analysis of the uploaded traffic footage.*

![WhatsApp Image 2025-07-11 at 02 34 57_7f9ac8ed](https://github.com/user-attachments/assets/092bb107-b337-435a-8766-baaa4a6d0970)

## Model Summary

### 1. Congestion Detection
- Trained a custom YOLOv8 model for vehicle segmentation.
- Calculated occupancy ratio of vehicle pixels to road pixels.
- Random Forest Classifier trained on these features.

### 2. Accident Detection
- MobileNetV2 used as base CNN for accident classification.
- Dataset organized as Accident and Non-Accident images.
- Predictions made on video frames (1 frame every second approx.).

---
##  Technologies Used

| Category         | Tools / Libraries                       |
|------------------|------------------------------------------|
| Language         | Python 3.10+                             |
| Deep Learning    | YOLOv8 (Ultralytics), TensorFlow 2.19    |
| ML Classifier    | Random Forest (for congestion prediction)|
| Framework        | Flask 3.1, Gunicorn                      |
| Image Processing | OpenCV 4.12                              |
| Training         | Google Colab, Jupyter Notebook           |
---


## ▶ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/traffic-congestion-detector.git
cd traffic-congestion-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
