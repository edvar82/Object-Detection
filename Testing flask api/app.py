from flask import Flask, request, send_file
from ultralytics import YOLO
import math
import cv2
import cvzone
import numpy as np
from PIL import Image
import io

model = YOLO("ppe.pt")

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def predict():
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    try:
        file = request.files['image'].read()  # lê o conteúdo do arquivo
        image = Image.open(io.BytesIO(file))

        image = image.convert('RGB')

        # Converter a imagem para um objeto numpy.array
        image_np = np.array(image)

        results = model(image, stream=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.3:
                    if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                        myColor = (0, 255, 0)
                    elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                        myColor = (0, 0, 255)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(image_np, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), myColor, 3)

        image_output = Image.fromarray(image_np)
        img_io = io.BytesIO()
        image_output.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return str(e), 400

@app.route('/infos', methods=['POST'])
def detect():
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    counts = {
        'Hardhat': 0,
        'Mask': 0,
        'NO-Hardhat': 0,
        'NO-Mask': 0,
        'NO-Safety Vest': 0,
        'Person': 0,
        'Safety Cone': 0,
        'Safety Vest': 0,
        'machinery': 0,
        'vehicle': 0
    }
    try:
        file = request.files['image'].read()  # lê o conteúdo do arquivo
        image = Image.open(io.BytesIO(file))

        image = image.convert('RGB')

        # Converter a imagem para um objeto numpy.array
        image_np = np.array(image)

        results = model(image, stream=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.3:
                    if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                        myColor = (0, 255, 0)
                    elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                        myColor = (0, 0, 255)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(image_np, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), myColor, 3)

                    counts[currentClass] += 1

        return counts
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
