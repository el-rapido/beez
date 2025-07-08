# 🐝 Lithuanian Bee Detection Model

## Overview

A pre-trained YOLOv8 model for automated bee detection in beehive entrance footage. This lightweight, high-performance model can detect and count individual bees with **97.8% accuracy** and runs efficiently on standard hardware.

## 🎯 What This Model Does

- **Detects individual bees** in beehive entrance videos and images
- **Counts bee activity** automatically for hive monitoring  
- **Processes footage in real-time** (12.5ms per frame)
- **Achieves 97.8% accuracy** with excellent precision and recall
- **Works on standard hardware** - no GPU required
- **Ready to use** - no training or setup needed

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy (mAP50)** | **97.8%** |
| **Precision** | **97.7%** |
| **Recall** | **94.3%** |
| **Model Size** | 17MB |
| **Inference Speed** | 12.5ms per frame |
| **Input Size** | 640x640 pixels |
| **Framework** | YOLOv8 (Ultralytics) |

## 🚀 Quick Start

### Installation
```bash
pip install ultralytics opencv-python
```

### Basic Usage
```python
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('best.pt')

# Detect bees in an image
results = model('beehive_image.jpg')

# Count detections
num_bees = len(results[0].boxes)
print(f"🐝 Detected {num_bees} bees")

# Show results with 97.8% accuracy confidence
results[0].show()
```

### Batch Processing
```python
# Process multiple images
image_folder = "test_images/"
results = model([f"{image_folder}image1.jpg", 
                f"{image_folder}image2.jpg"])

for i, result in enumerate(results):
    bee_count = len(result.boxes)
    print(f"Image {i+1}: {bee_count} bees detected")
```

## 📁 Repository Contents

```
bee-detection-model/
├── best.pt                 # Pre-trained YOLOv8 model (17MB)
├── test.py                 # Ready-to-use inference script
├── test_images/            # Sample beehive images
│   ├── hive1.jpg
│   ├── hive2.jpg
│   └── hive3.jpg
├── requirements.txt        # Dependencies
└── README.md              # This documentation
```

## 🛠️ Using the Inference Script

The included `test.py` script provides an easy way to test the model:

```bash
# Process test images folder
python test.py

# Or run custom inference
python test.py --source your_image.jpg
```

## 🎯 Applications

### 🔬 Research & Monitoring
- **Bee activity analysis**: Track hive productivity over time
- **Colony health assessment**: Monitor entrance activity patterns
- **Behavioral studies**: Quantify bee traffic for research

### 🚜 Practical Beekeeping
- **Automated monitoring**: Set up cameras for remote hive checking
- **Activity logging**: Track bee numbers throughout the day
- **Health indicators**: Detect unusual activity patterns

### 🤖 Integration Projects
- **Raspberry Pi monitoring**: Deploy on edge devices
- **Web applications**: Build dashboards for multiple hives
- **Mobile apps**: Create smartphone-based hive inspection tools

## 📊 Training Data

This model was trained on the **Lithuanian Beehive Dataset**:
- **7,200 annotated frames** from real beehive footage
- **8 different beehive setups** across Lithuania
- **High resolution**: 1920x1080 source footage
- **Real conditions**: Various lighting and weather
- **Peak season**: June-July 2023 data

*Dataset source: [Lithuanian Beehive Dataset](https://data.mendeley.com/datasets/8gb9r2yhfc/6)*

## ⚙️ Technical Details

### Model Specifications
- **Architecture**: YOLOv8 Nano
- **Input**: RGB images, 640x640 pixels
- **Output**: Bounding boxes with confidence scores
- **Classes**: Single class ("bee")
- **Format**: PyTorch (.pt)

### Performance Characteristics
- **CPU inference**: ~12.5ms per frame
- **GPU inference**: ~3-5ms per frame  
- **Memory usage**: ~100MB RAM
- **Minimum requirements**: Python 3.8+, 2GB RAM

## 🎮 Try It Yourself

1. **Download** the model and test images
2. **Install** dependencies: `pip install ultralytics`
3. **Run** the inference script: `python test.py`
4. **View** results with detected bees highlighted

## 📋 Output Format

The model returns:
- **Bounding boxes**: Coordinates of detected bees
- **Confidence scores**: Detection certainty (0-1)
- **Counts**: Total number of bees in each image
- **Visualizations**: Images with bounding boxes drawn

Example output:
```python
# Detection results
boxes: tensor([[x1, y1, x2, y2, confidence, class], ...])
count: 23 bees detected
avg_confidence: 0.847
```

## 🔧 Integration Examples

### Video Processing
```python
import cv2

cap = cv2.VideoCapture('hive_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        bee_count = len(results[0].boxes)
        print(f"Frame bees: {bee_count}")
```

### Real-time Camera
```python
# Use webcam or IP camera
cap = cv2.VideoCapture(0)  # or 'rtsp://camera-ip/stream'
while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('Bee Detection', annotated)
```

## 📚 Citation

If you use this model in research, please cite:

```bibtex
@misc{lithuanian_bee_detection_2025,
  title={Lithuanian Bee Detection Model - YOLOv8},
  author={Precious Thom},
  year={2025},
  url={https://github.com/el-rapido/beez}
}
```

## 🤝 Support & Feedback

- **Issues**: Report problems or ask questions in GitHub Issues
- **Improvements**: Suggestions for better performance welcome
- **Use cases**: Share how you're using the model!

## 📄 License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This model is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

### ✅ Permitted Uses
- **Research and academic use** - Free for universities and research institutions
- **Educational purposes** - Teaching and learning applications
- **Personal projects** - Non-commercial experimentation and development
- **Open-source projects** - Non-commercial software development

### ❌ Restrictions
- **Commercial use prohibited** without explicit written permission
- **No selling or monetizing** the model or its derivatives
- **No integration** into commercial products or services

### 💼 Commercial Licensing
For commercial applications, please contact us for licensing terms:
- 📧 **Commercial inquiries**: preciousekarithom@gmail.com
- 💰 **Licensing fees**: Reasonable rates for commercial use
- 🤝 **Custom agreements**: Tailored terms for specific use cases
- 🏢 **Enterprise solutions**: Volume licensing available

**Commercial use cases requiring permission:**
- Agricultural monitoring systems
- Smart beekeeping products
- Bee research services
- Integration into commercial software
- Government/municipal monitoring projects

---

**Ready to detect some bees? Download the model and start counting! 🐝**
