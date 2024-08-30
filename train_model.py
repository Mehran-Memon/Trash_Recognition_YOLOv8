from ultralytics import YOLO
import ultralytics as ut

# Specify the path to your data.yaml file
data_yaml_path = "D:\\Garbage Detection\\Trash-Detection-1\\data.yaml"

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    # Train the model
    results = model.train(
        data=data_yaml_path,  # Corrected variable name
        imgsz=640,
        epochs=100,
        batch=8,
        optimizer="SGD",
        lr0=1e-3
    )
