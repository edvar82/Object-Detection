from ultralytics import YOLO
import cv2

# Initialize model
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model('Images/2.png', show=True)
cv2.waitKey(0)