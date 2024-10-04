# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# import uvicorn
# import sys
# import os
# import cv2
# import numpy as np
# from io import BytesIO
# sys.path.append('/Users/user/Desktop/ty/yolov5')  # Adjust path to your YOLOv5 installation

# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages
# from utils.general import non_max_suppression, scale_boxes
# import torch

# app = FastAPI()

# # Load model (Assuming the model and its weights are accessible and configured correctly)
# model = DetectMultiBackend('/Users/user/Desktop/ty/best.pt', device='cpu')  # or 'cuda' for GPU

# def detect_image(image_bytes):
#     stride, names = model.stride, model.names
#     imgsz = (640, 640)  # Define default img size

#     # Convert bytes to numpy array
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     im0s = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Prepare image for YOLOv5
#     img = cv2.resize(im0s, imgsz)
#     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).to('cpu')
#     img = img.float()  # uint8 to fp16/32
#     img /= 255  # normalize to 0-1 range
#     img = img[None]  # add batch dimension

#     pred = model(img, augment=False, visualize=False)
#     pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)

#     # Process detections
#     results = []
#     for i, det in enumerate(pred):  # detections per image
#         if len(det):
#             # Rescale boxes from img size to im0 size
#             det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 results.append({
#                     "class": names[int(cls)],
#                     "confidence": f"{conf:.2f}",
#                     "bbox": [int(x) for x in xyxy]
#                 })
    
#     # Draw bounding boxes on the image
#     for result in results:
#         bbox = result['bbox']
#         label = f"{result['class']} {result['confidence']}"
#         color = (0, 255, 0)  # Green color for the box
#         cv2.rectangle(im0s, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#         cv2.putText(im0s, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return results, im0s

# @app.post("/detect/")
# async def detect(file: UploadFile = File(...)):
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
#     contents = await file.read()
#     results, img_with_boxes = detect_image(contents)
#     print("Detection Results:", results)  # Print results to the server console
    
#     # Convert the image to bytes
#     is_success, buffer = cv2.imencode(".jpg", img_with_boxes)
#     if not is_success:
#         raise HTTPException(status_code=500, detail="Failed to encode image.")
    
#     io_buf = BytesIO(buffer)
    
#     return StreamingResponse(io_buf, media_type="image/jpeg")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import sys
import os
import cv2
import numpy as np
from io import BytesIO
import torch
sys.path.append('/Users/user/Desktop/ty/yolov5')  # Adjust path to your YOLOv5 installation

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes

import torch

app = FastAPI()

# Load YOLOv5 model (Assuming the model and its weights are accessible and configured correctly)
model = DetectMultiBackend('/Users/user/Desktop/ty/best2.pt', device='cpu')  # Change to 'cuda' if using GPU

# Image processing function (e.g., HSV, Gaussian blur, Canny)
def process_image(image, hmin, hmax, smin, smax, vmin, vmax, 
                  gaussian_blur, canny_low, canny_high, 
                  kernel_d, kernel_e, dilation, erosion):

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply HSV thresholding
    lower_bound = np.array([hmin, smin, vmin])
    upper_bound = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Apply Gaussian Blur
    if gaussian_blur > 0:
        result = cv2.GaussianBlur(result, (gaussian_blur*2+1, gaussian_blur*2+1), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(result, canny_low, canny_high)

    # Create kernels for dilation and erosion
    kernel_d = np.ones((kernel_d, kernel_d), np.uint8)
    kernel_e = np.ones((kernel_e, kernel_e), np.uint8)

    # Apply Dilation
    if dilation > 0:
        edges = cv2.dilate(edges, kernel_d, iterations=dilation)

    # Apply Erosion
    if erosion > 0:
        edges = cv2.erode(edges, kernel_e, iterations=erosion)

    return edges

# Function to detect objects using the YOLOv5 model and apply image processing
def detect_and_process_image(image_bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    im0s = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Image processing settings
    hmin, hmax = 0, 111
    smin, smax = 0, 213
    vmin, vmax = 129, 239
    gaussian_blur = 2
    canny_low, canny_high = 251, 179
    kernel_d, kernel_e = 9, 12
    dilation, erosion = 0, 0

    # Apply image processing
    processed_img = process_image(im0s, hmin, hmax, smin, smax, vmin, vmax, 
                                  gaussian_blur, canny_low, canny_high, 
                                  kernel_d, kernel_e, dilation, erosion)

    # Check if the image has 3 channels (color image), otherwise convert
    if len(processed_img.shape) == 2:
        # Grayscale image, add a channel dimension to match expected format
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    # Prepare image for YOLOv5 detection
    img = cv2.resize(processed_img, (640, 640))
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255  # normalize to 0-1 range
    img = img[None]  # add batch dimension

    # Get predictions from YOLOv5 model
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.5, max_det=1000)

    # Process detections
    results = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                results.append({
                    "class": model.names[int(cls)],
                    "confidence": f"{conf:.2f}",
                    "bbox": [int(x) for x in xyxy]
                })
    
    # Draw bounding boxes on the image
    for result in results:
        bbox = result['bbox']
        label = f"{result['class']} {result['confidence']}"
        color = (0, 255, 0)  # Green color for the box
        cv2.rectangle(im0s, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(im0s, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return results, im0s


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    contents = await file.read()
    results, img_with_boxes = detect_and_process_image(contents)
    print("Detection and Processing Results:", results)  # Print results to the server console
    
    # Convert the image to bytes
    is_success, buffer = cv2.imencode(".jpg", img_with_boxes)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    
    io_buf = BytesIO(buffer)
    
    return StreamingResponse(io_buf, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
