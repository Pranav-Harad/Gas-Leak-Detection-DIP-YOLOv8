# 🚀 Hybrid Gas Leak Detection  
### 🔥 Real-time AI Guardian for Industrial Safety  
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

## 🌪️ The Mission: Stop Invisible Threats

Gas leaks are **silent killers** — odorless, colorless, and dangerous.  
This project fuses **Digital Image Processing (DIP)** with **YOLOv8 Deep Learning** to **detect, localize, and alert** gas leaks in **real time** using visual data.

---

## 🧩 Methodology: How the AI Sees the Invisible

This hybrid system combines **classic Digital Image Processing (DIP)** with **YOLOv8 deep learning** to detect and localize gas leaks in real time.

### 1️⃣ Data Acquisition

Gas leak images (IR and RGB) were collected and annotated using **Roboflow**.

- **Sources:** RGB and IR cameras  
- **Purpose:** Provide high-quality, labeled datasets for training  

### 2️⃣ Image Preprocessing (DIP Magic)

Enhances the visual information before feeding into the model:

- **Gaussian Blur** – Smooths noise while preserving structure  
- **Bilateral Filter** – Keeps edges sharp while reducing noise  
- **Adaptive Histogram Equalization (CLAHE)** – Boosts contrast for better visibility  
- **Edge & Contour Detection** – Highlights the gas plume for accurate recognition  

### 3️⃣ Model Training (YOLOv8 Deep Learning)

- Uses **YOLOv8 pretrained weights**  
- Fine-tunes on the processed dataset  
- Learns to detect and localize gas plumes accurately  

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/content/Gas-Leak-1/data.yaml", epochs=30, imgsz=640)
``` 

### 4️⃣ Detection & Prediction

Once trained, the model predicts gas leak regions in new images or video frames:

```python
results = model.predict("/content/Gas-Leak-1/test/images/sample.jpg", conf=0.4)
results.show()
``` 

### ✅ Key Highlights

- Hybrid pipeline: DIP filters enhance input, YOLOv8 detects gas  
- Real-time performance with GPU support  
- Accurate plume localization with confidence scores

### ⚙️ Key Features

- ✅ Hybrid DIP + Deep Learning pipeline  
- ✅ Real-time detection using YOLOv8  
- ✅ GPU acceleration (Google Colab compatible)  
- ✅ Edge-preserving noise reduction  
- ✅ Accurate localization of gas plumes  
- ✅ Roboflow dataset integration  

---

### 📊 Model Evaluation

- **Confidence Score (e.g., Gas: 0.52)** → Probability of detected object being a gas leak  
- **mAP (Mean Average Precision)** → Measures detection accuracy across test samples

### 🚀 Tech Stack

- **Python 3.12.11**  
- **OpenCV (DIP)**  
- **Ultralytics YOLOv8**  
- **NumPy, Matplotlib**  
- **Roboflow API**  
- **Google Colab (GPU Runtime)**  

---

### 🧠 Future Scope

- Integrate **Thermal IR cameras** for enhanced plume detection  
- Add **IoT alert system** for industrial safety  
- Develop **web or mobile interface** for live monitoring  
- Enable **real-time video stream inference**  

---

## Author 👨‍💻
Made with ❤️ by **Pranav**  
📧 Email: [pranavharad64@gmail.com](mailto:pranavharad64@gmail.com)  
🔗 LinkedIn: [www.linkedin.com/in/pranav-harad-667070268](https://www.linkedin.com/in/pranav-harad-667070268)
