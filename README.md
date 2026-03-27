# 🦺 Harness Detection — YOLO11s

A custom-trained object detection model using **YOLO11s** to detect harnesses in images. Trained on a labeled dataset of 582 training images and 146 validation images using Ultralytics on Kaggle (Tesla T4 GPU).

---

## 📁 Repository Structure

```
├── weights/
│   ├── best.pt          # Best checkpoint (use this for inference)
│   └── last.pt          # Last epoch checkpoint
├── harness-train.ipynb  # Training notebook
└── README.md
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLO11s |
| Parameters | 9,428,566 |
| Layers | 182 |
| GFLOPs | 21.6 |
| Input Size | 800 × 800 |
| Epochs | 100 |
| Dataset | harness_dataset |
| Train Images | 582 |
| Val Images | 146 |
| Framework | Ultralytics 8.4.30 |
| Hardware | Kaggle Tesla T4 (14GB) |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install ultralytics --upgrade
```

### 2. Run inference

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("weights/best.pt")

# Run inference on an image
results = model("path/to/your/image.jpg")
results[0].show()

# Save results
results = model("path/to/your/image.jpg", save=True)
```

### 3. Batch inference on a folder

```python
results = model("path/to/images/folder/", save=True)
```

---

## 🏋️ Training

The model was trained on Kaggle using the following command:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="harness_dataset/data.yaml",
    epochs=100,
    imgsz=800,
    plots=True
)
```

### Dataset structure

```
harness_dataset/
├── data.yaml
├── images/
│   ├── train/    # 582 images
│   └── val/      # 146 images
└── labels/
    ├── train/
    └── val/
```

---

## 📊 Evaluation

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
metrics = model.val(data="harness_dataset/data.yaml")

print(metrics.box.map)    # mAP50-95
print(metrics.box.map50)  # mAP50
```

---

## 📦 Export

Export the model to other formats (ONNX, TensorRT, etc.):

```python
model.export(format="onnx")   # ONNX
model.export(format="engine") # TensorRT
```

---

## 🛠️ Requirements

- Python 3.8+
- `ultralytics >= 8.4.0`
- PyTorch
- CUDA (optional, for GPU inference)

---
