import cv2
from PIL import Image
from ultralytics import YOLO
import os

# Load the model
model = YOLO("D:\\Garbage Detection\\runs\\detect\\train4\\weights\\best.pt")

# Define the source for prediction
source = "D:\\Garbage Detection\\Trash-Detection-1\\test\\images"

# Define the directory where you want to save the predictions
save_dir = "C:\\Users\\msipc\\Desktop\\Garbage Detection\\predictions"

# Make sure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Perform prediction and save the results
results = model.predict(source=source, show=False, save=True, save_dir=save_dir)
