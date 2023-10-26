from ultralytics import YOLO
import cv2

# Initialize model
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model('../Images/1.png', show=True)
print(results[0])
cv2.waitKey(0)