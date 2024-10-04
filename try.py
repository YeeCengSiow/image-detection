import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load your own image (same for both im1 and im2)
image_path = "/Users/user/Desktop/ty/processed/ADELINE LEE XI YEAN_page_1_img_1.jpeg"  # Replace with your image path

im1 = Image.open(image_path)  # PIL image
im2 = cv2.imread(image_path)[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1, im2], size=640)  # batch of images (same image)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # im1 predictions (tensor)
results.pandas().xyxy[0]  # im1 predictions (pandas)
