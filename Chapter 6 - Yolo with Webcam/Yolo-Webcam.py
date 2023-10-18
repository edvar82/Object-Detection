from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../Yolo-Weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    cv2.imshow("Image", img)
    cv2.waitKey(1)