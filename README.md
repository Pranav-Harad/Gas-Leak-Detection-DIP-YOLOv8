# Hybrid Gas Leak Detection using DIP + YOLOv8
### Real-time AI System for Industrial Gas Leak Monitoring and Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv" />
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-orange?logo=ultralytics" />
  <img src="https://img.shields.io/badge/Roboflow-Dataset%20Integration-purple?logo=roboflow" />
  <img src="https://img.shields.io/badge/Google%20Colab-GPU%20Runtime-yellow?logo=googlecolab" />
</p>

---

## Overview

Gas leaks pose severe threats to **human safety, environmental health, and industrial operations**.  
This project introduces a **hybrid AI-based framework** that combines **Digital Image Processing (DIP)** and **YOLOv8 deep learning** for **accurate, real-time gas leak detection**.

The system:
- Enhances input images using classical DIP techniques
- Uses YOLOv8 to detect and localize gas plumes with bounding boxes
- Leverages a high-quality annotated dataset from **Roboflow**

---

## Methodology

### 1. Data Acquisition
Infrared (IR) and RGB gas leak images collected and annotated using **Roboflow**.


from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("gas-leak-x7van-osejz")
version = project.version(1)
dataset = version.download("yolov8")
