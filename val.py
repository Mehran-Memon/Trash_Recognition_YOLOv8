from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("D:\\Garbage Detection\\runs\\detect\\train4\\weights\\best.pt")

if __name__ == '__main__':
    # Validate on training data
    model.val()