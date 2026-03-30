
# 🦁 Real-Time Animal Classification Diagnostics with ConvNeXt-V2

## 📌 Project Overview
This project is an **AI-powered real-time animal classification and diagnostics system** developed using **ConvNeXt-V2**, **YOLOv8**, and **Streamlit**.

The system can:
- Identify the **animal species** from an uploaded image
- Show the **confidence score**
- Generate **Grad-CAM heatmaps** to explain predictions
- Count the **number of animals present** in the image
- Provide **performance metrics** like confusion matrix and classification report
- Display **dataset archive analysis**

This project is useful for **wildlife monitoring**, **zoo management**, **animal dataset analysis**, and **AI-based image recognition learning**.

---

## 🚀 Features

### 🔍 1. Animal Classification
- Upload an animal image
- Predicts the animal class using **ConvNeXt-V2**
- Displays prediction confidence

### 📊 2. Confidence Score Visualization
- Shows confidence level using a progress bar
- Detects low-confidence predictions

### 🧠 3. Grad-CAM Explainability
- Highlights the image regions the AI model used to make its decision
- Helps in understanding model behavior

### 🐾 4. Animal Counting
- Uses **YOLOv8** object detection
- Detects and counts how many animals are present in the uploaded image

### 📈 5. Performance Diagnostics
- Confusion Matrix
- Classification Report
- Accuracy insights using test dataset

### 📂 6. Dataset Archive Analysis
- Shows class distribution
- Displays example images from dataset folders

---

## 🧠 Where AI is Used in This Project

AI is used in **two main parts** of this system:

### 1. **Animal Species Classification**
A deep learning model (**ConvNeXt-V2**) is used to classify uploaded images into one of the supported animal categories.

### 2. **Animal Counting**
An object detection model (**YOLOv8**) is used to detect animals in the image and count how many are present.

### 3. **Prediction Explainability**
**Grad-CAM** is used to visualize the important regions of the image that influenced the model’s prediction.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **PyTorch**
- **Torchvision**
- **timm**
- **Ultralytics YOLOv8**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Plotly**
- **scikit-learn**
- **Pillow**
- **Grad-CAM**

---

## 🗂️ Project Structure

```bash
Real-Time-Animal-Classification-Diagnostics-with-ConvNeXt-V2/
│
├── app.py                  # Main Streamlit application
├── train_zoo.py           # Model training script
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
├── TEST_REPORT.md         # Testing documentation
├── zoo_bundle.pth         # Trained ConvNeXt-V2 model weights
├── yolov8n.pt             # YOLOv8 model (optional / auto-download)
│
├── raw-img/               # Dataset folder
├── Test Images/           # Sample test images
├── uploads/               # Uploaded images during runtime
├── tests/                 # Test-related files
└── venv/                  # Virtual environment (ignored in Git)
