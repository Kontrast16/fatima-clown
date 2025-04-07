import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
socketio = SocketIO(app)

# Загружаем модель для распознавания лиц
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Функция распознавания лиц
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, len(faceBoxes)

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Обработка кадров через WebSocket
@socketio.on('frame')
def handle_frame(data):
    print("Получен запрос на обработку кадра")
    img_data = base64.b64decode(data.split(',')[1])
    npimg = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    resultImg, faceCount = highlightFace(faceNet, frame)
    cv2.putText(resultImg, f"Faces detected: {faceCount}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', resultImg)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    print("Кадр обработан, отправляем результат")
    emit('response_frame', {'image': f'data:image/jpeg;base64,{img_base64}', 'count': faceCount})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)