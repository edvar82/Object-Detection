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
    # classes que o modelo reconhece
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    try:
        file = request.files['image'].read()  # lê o conteúdo do arquivo
        # Converte o conteúdo do arquivo para um objeto PIL.Image
        image = Image.open(io.BytesIO(file))

        # Converte a imagem para RGB
        image = image.convert('RGB')

        # Converter a imagem para um objeto numpy.array
        image_np = np.array(image)

        # Executa a detecção de objetos no quadro atual
        results = model(image, stream=False)

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
                current_class = classNames[cls]

                if conf > 0.3:
                    if current_class == 'Hardhat' or current_class == 'Mask' or current_class == 'Safety Vest':
                        myColor = (0, 255, 0)
                    elif current_class == 'NO-Hardhat' or current_class == 'NO-Mask' or current_class == 'NO-Safety Vest':
                        myColor = (0, 0, 255)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(image_np, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), myColor, 3)

        # Converte o objeto numpy.array para um objeto PIL.Image
        image_output = Image.fromarray(image_np)
        # Cria um objeto BytesIO
        img_io = io.BytesIO()
        # Salva a imagem no objeto BytesIO
        image_output.save(img_io, 'PNG')
        # Move o cursor para o início do objeto BytesIO
        img_io.seek(0)
        # Retorna a imagem
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return str(e), 400

@app.route('/infos', methods=['POST'])
def detect():
    # classes que o modelo reconhece
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
        # Converte o conteúdo do arquivo para um objeto PIL.Image
        image = Image.open(io.BytesIO(file))

        # Converte a imagem para RGB
        image = image.convert('RGB')

        # Converter a imagem para um objeto numpy.array
        image_np = np.array(image)

        # Executa a detecção de objetos no quadro atual
        results = model(image, stream=False)

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
                current_class = classNames[cls]

                if conf > 0.3:
                    if current_class == 'Hardhat' or current_class == 'Mask' or current_class == 'Safety Vest':
                        myColor = (0, 255, 0)
                    elif current_class == 'NO-Hardhat' or current_class == 'NO-Mask' or current_class == 'NO-Safety Vest':
                        myColor = (0, 0, 255)
                    else:
                        myColor = (255, 0, 0)
                    # Desenha o texto
                    cvzone.putTextRect(image_np, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                    # Desenha a caixa delimitadora
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), myColor, 3)

                    counts[current_class] += 1
        # Pega somente os valores que não são 0
        filtered_data = {k: v for k, v in counts.items() if v != 0}
        return filtered_data
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
