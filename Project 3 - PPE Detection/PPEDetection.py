from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 640)
# cap.set(4, 480)
cap = cv2.VideoCapture('../Videos/ppe-2.mp4')  # For Video

model = YOLO("ppe.pt")

# classes que o modelo reconhece
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

myColor = (0, 0, 255)

# Loop principal para processar o vídeo
while True:
    # Lê um quadro do vídeo
    success, img = cap.read()

    # Executa a detecção de objetos no quadro atual
    results = model(img, stream=True)

    # Loop para processar os resultados da detecção
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Obtem os bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])
            # Current class
            currentClass = classNames[cls]

            # Define a cor com base na classe
            if conf > 0.5:
                if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0, 255, 0)
                elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColor = (0, 0, 255)
                else:
                    myColor = (255, 0, 0)

                # Desenha o texto
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                   thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)

                # Desenha a caixa delimitadora
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)