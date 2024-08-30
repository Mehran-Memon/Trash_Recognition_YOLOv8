# Garbage Detection Model Using YOLOv8

This repository contains the code and resources for training and deploying a YOLOv8 model to detect various types of garbage. The model is designed to assist in automated garbage classification and waste management.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Validation](#validation)
- [Prediction](#prediction)
  - [Batch Prediction](#batch-prediction)
  - [Single Image Prediction](#single-image-prediction)
  - [Real-Time Prediction](#real-time-prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project involves training a YOLOv8 model on a dataset of garbage images to develop an efficient garbage detection system. The model can detect and classify different types of waste, which is useful for various waste management applications.

## Dataset

The dataset used for training is obtained from Roboflow. The code to download the dataset is provided in the `import_data.py` script.

### import_data.py

```python
from roboflow import Roboflow

rf = Roboflow(api_key="T0wFToAaOYWZDBERldKM")
project = rf.workspace("federico-andrea-rizzi-g1fjs").project("dataset-itis-cardano-trash-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

## This script downloads the dataset and prepares it for training.

## Model Training

The model is trained using the `train_model.py` script. The training is performed on the YOLOv8 model with the following configuration:

- **Model:** YOLOv8 Medium (`yolov8m.pt`)
- **Image Size:** 640x640
- **Epochs:** 100
- **Batch Size:** 8
- **Optimizer:** SGD
- **Learning Rate:** 1e-3

### train_model.py

```python
from ultralytics import YOLO

# Specify the path to your data.yaml file
data_yaml_path = "D:\\Garbage Detection\\Trash-Detection-1\\data.yaml"

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    # Train the model
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=100,
        batch=8,
        optimizer="SGD",
        lr0=1e-3
    )
```
## Validation

To validate the trained model, use the `val.py` script. This script evaluates the model's performance on the validation set.

### val.py

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("D:\\Garbage Detection\\runs\\detect\\train4\\weights\\best.pt")

if __name__ == '__main__':
    # Validate on training data
    model.val()
```

## Prediction

The repository includes scripts for making predictions using the trained YOLOv8 model. You can perform batch predictions, single image predictions, or real-time predictions.

### Batch Prediction

The `predict.py` script allows you to run predictions on a batch of images and save the results.

```python
import cv2
from PIL import Image
from ultralytics import YOLO
import os

# Load the model
model = YOLO("D:\\Garbage Detection\\Training\\Yolov8s\\weights\\best.pt")

# Define the source for prediction
source = "D:\\Garbage Detection\\Trash-Detection-1\\test\\images"

# Define the directory where you want to save the predictions
save_dir = "C:\\Users\\msipc\\Desktop\\Garbage Detection\\predictions"

# Make sure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Perform prediction and save the results
results = model.predict(source=source, show=False, save=True, save_dir=save_dir)
```

### Single Image Prediction

Use the `single_predict.py` script to predict on a single image and visualize the result.

```python
import cv2
from PIL import Image
from ultralytics import YOLO
import os

# Load the model
model = YOLO("D:\\Garbage Detection\\runs\\detect\\train4\\weights\\best.pt")

# Define the source for prediction
source = "D:\\Garbage Detection\\Taco_v1.1-4\\test\\images\\batch_2_000007_JPG.rf.790c35d798f96e6f80176b4e19568e6f.jpg"

# Perform prediction and save the results
results = model.predict(source=source, show=True, save=True)
```

### Real-Time Prediction

The `realtime.py` script enables real-time garbage detection using your webcam.

```python
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("D:\\Garbage Detection\\runs\\detect\\train5\\weights\\best.pt")

if __name__ == '__main__':
    # Perform inference with stream=True to avoid accumulating results in RAM
    results = model.predict(source=0, show=True, conf=0.1, stream=True)
    
    for r in results:
        boxes = r.boxes  # Get bounding boxes
        masks = r.masks  # Get segmentation masks (if any)
        probs = r.probs  # Get class probabilities (for classification tasks)
```

## Contributing

Contributions to this project are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
