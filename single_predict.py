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
