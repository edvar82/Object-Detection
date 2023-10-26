from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

model = YOLO("ppe.pt")

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def predict():
    try:
        file = request.files['image'].read()  # lê o conteúdo do arquivo
        file = io.BytesIO(file)  # cria um objeto de arquivo em memória
        return send_file(file, mimetype='image/gif')  # envia o arquivo
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
