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
