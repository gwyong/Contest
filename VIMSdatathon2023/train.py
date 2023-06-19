from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO("yolov8m.yaml")
model = YOLO("yolov8x.pt")

# Use the model
results = model.train(data="./datasets/data.yaml", epochs=200, imgsz=640,
                      optimizer="AdamW", seed=0, lr0=1e-3, dropout=0.2, patience=30)  # train the model
metrics = model.val()