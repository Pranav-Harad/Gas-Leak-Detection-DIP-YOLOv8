# 🚀 **Hybrid Gas Leak Detection**  
### 🔥 **Real-time AI Guardian for Industrial Safety**  
#### *Powered by DIP + YOLOv8*  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.11-FFD43B?logo=python&logoColor=3776AB&style=for-the-badge" />
  <img src="https://img.shields.io/badge/OpenCV-5.0+-green?logo=opencv&style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=ultralytics&style=for-the-badge" />
  <img src="https://img.shields.io/badge/Roboflow-Dataset-purple?logo=roboflow&style=for-the-badge" />
  <img src="https://img.shields.io/badge/Google%20Colab-GPU%20Magic-yellow?logo=googlecolab&style=for-the-badge" />
</p>

> **"When gas leaks silently, AI sees it clearly."**  

---

## 🌪️ **The Mission: Stop Invisible Threats**

Gas leaks are **silent killers** — odorless, colorless, and deadly.  
This project fuses **Digital Image Processing (DIP)** with **YOLOv8 Deep Learning** to **detect, localize, and alert** gas leaks in **real time** using visual data.

---

## 🧬 **How It Works — The AI Pipeline**

```mermaid
flowchart TD
    A[📸 Input Image or Video Frame]
    --> B{🔍 Preprocess using DIP}
    B --> C[✨ Gaussian Blur]
    B --> D[🖼️ Bilateral Filter]
    B --> E[🌈 Adaptive Histogram Equalization]
    B --> F[✂️ Edge & Contour Detection]
    F --> G[🤖 YOLOv8 Model Inference]
    G --> H[🚨 Bounding Box + Confidence Score]
    H --> I[📊 Output Visualization & Alerts]

    style A fill:#FF6B6B,stroke:#333
    style B fill:#FFD93D,stroke:#333
    style F fill:#6BCB77,stroke:#333
    style G fill:#4D96FF,stroke:#333
    style H fill:#9D4EDD,stroke:#333
    style I fill:#00BFA6,stroke:#333


🧩 Methodology
1️⃣ Data Acquisition
Images of gas leaks (IR and RGB) were collected and annotated using Roboflow.

2️⃣ Preprocessing (DIP Techniques)
Enhances image quality for model training and inference:

Gaussian Filter – Removes random noise

Bilateral Filter – Preserves edges during smoothing

Adaptive Histogram Equalization (CLAHE) – Boosts contrast

Edge & Contour Detection – Highlights plume boundaries

3️⃣ Model Training
YOLOv8 is trained for robust gas leak detection:


from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/Gas-Leak-1/data.yaml", epochs=30, imgsz=640)
4️⃣ Detection & Prediction
Run inference on test images or videos:


results = model.predict("/content/Gas-Leak-1/test/images/sample.jpg", conf=0.4)
results.show()
⚙️ Key Features
✅ Hybrid DIP + Deep Learning pipeline
✅ Real-time detection using YOLOv8
✅ GPU acceleration (Google Colab compatible)
✅ Edge-preserving noise reduction
✅ Accurate localization of gas plumes
✅ Seamless Roboflow dataset integration

📊 Model Evaluation
Confidence Score (e.g., Gas: 0.52) → Probability of a detected object being a gas leak

mAP (Mean Average Precision) → Measures detection accuracy across test samples

🚀 Tech Stack
Python 3.12.11

OpenCV (DIP)

Ultralytics YOLOv8

NumPy, Matplotlib

Roboflow API

Google Colab (GPU Runtime)

🧠 Future Scope
Integrate Thermal IR cameras for enhanced plume detection

Add IoT sensor alerts for industrial deployment

Develop a web or mobile interface for live monitoring

Enable real-time video stream inference

🤝 Contributor
👨‍💻 Pranav Harad
Developer 
