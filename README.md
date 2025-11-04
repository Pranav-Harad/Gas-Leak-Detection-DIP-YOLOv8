Hybrid Gas Leak Detection using DIP + YOLOv8

🚨 Real-time AI system for industrial gas leak monitoring and detection
🌍 Overview

Gas leaks pose severe threats to human safety, environmental health, and industrial operations.
This project presents a hybrid detection framework combining Digital Image Processing (DIP) techniques and YOLOv8 deep learning to detect gas leaks from image or video data accurately.

Leveraging Roboflow datasets, filter-based preprocessing, and real-time inference, the system enhances image quality, extracts features, and classifies potential leaks with improved accuracy and localization.

🧩 Methodology

1️⃣ Data Acquisition

Infrared (IR) and RGB images of gas leaks were sourced and annotated using Roboflow.

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("gas-leak-x7van-osejz")
version = project.version(1)
dataset = version.download("yolov8")

2️⃣ Preprocessing (DIP Techniques)

Before detection, images undergo:

Gaussian Filtering → Noise removal

Bilateral Filtering → Edge-preserving smoothing

Adaptive Histogram Equalization → Enhanced contrast

Edge & Contour Detection → Shape extraction of plumes

3️⃣ Model Training

Using YOLOv8 for object detection:

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/Gas-Leak-1/data.yaml", epochs=30, imgsz=640)

4️⃣ Detection & Prediction

Test and visualize gas leak detection:

results = model.predict("/content/Gas-Leak-1/test/images/sample.jpg", conf=0.4)
results.show()

⚙️ Features

✅ Hybrid approach (DIP + Deep Learning)
✅ Real-time detection on GPU runtime
✅ Noise reduction and contrast enhancement
✅ Object localization with bounding boxes
✅ Scalable dataset integration with Roboflow

📊 Evaluation

Confidence Score (e.g., Gas: 0.52) → Probability of correct detection

mAP (Mean Average Precision) → Model accuracy metric after training

🚀 Tech Stack

Python, OpenCV, YOLOv8 (Ultralytics)

NumPy, Matplotlib, Roboflow API

Google Colab GPU Runtime



🧠 Future Scope

Integrate thermal IR sensors for multi-spectrum analysis

Real-time IoT alert systems for industrial safety

Deploy model as web or mobile app for field monitoring

🤝 Contributors

Pranav Harad – Developer & Researcher
