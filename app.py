from flask import Flask, request 
from flask_cors import CORS 
import cv2 
import numpy as np 
 
app = Flask(__name__) 
CORS(app) 
 
# Model files 
faceProto = "opencv_face_detector.pbtxt" 
faceModel = "opencv_face_detector_uint8.pb" 
ageProto = "age_deploy.prototxt" 
ageModel = "age_net.caffemodel" 
genderProto = "gender_deploy.prototxt" 
genderModel = "gender_net.caffemodel" 
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
genderList = ['Male', 'Female'] 
 
try: 
    faceNet = cv2.dnn.readNet(faceModel, faceProto) 
    ageNet = cv2.dnn.readNet(ageModel, ageProto) 
    genderNet = cv2.dnn.readNet(genderModel, genderProto) 
    print(" Models loaded successfully") 
except Exception as e: 
    print(" Error loading models:", e) 
 
def detect_faces(net, image, threshold=0.5): 
    h, w = image.shape[:2] 
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False) 
    net.setInput(blob) 
    detections = net.forward() 
    faces = [] 
    for i in range(detections.shape[2]): 
        confidence = detections[0, 0, i, 2] 
        if confidence > threshold: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
            (x1, y1, x2, y2) = box.astype(int) 
            faces.append([x1, y1, x2, y2]) 
    return faces 
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    if 'image' not in request.files: 
        return "Error: No image uploaded", 400 
 
    file = request.files['image'] 
    npimg = np.frombuffer(file.read(), np.uint8) 
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR) 
 
    if image is None: 
        return "Error: Invalid image", 400 
 
    faces = detect_faces(faceNet, image) 
 
    if not faces: 
        return "No faces detected", 200 
 
    padding = 20 
    for (x1, y1, x2, y2) in faces: 
        face_img = image[max(0, y1 - padding):min(y2 + padding, image.shape[0] - 1), 
                         max(0, x1 - padding):min(x2 + padding, image.shape[1] - 1)] 
 
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, 
swapRB=False) 
 
        genderNet.setInput(blob) 
        gender_preds = genderNet.forward() 
        gender = genderList[gender_preds[0].argmax()] 
 
        ageNet.setInput(blob) 
        age_preds = ageNet.forward() 
        age = ageList[age_preds[0].argmax()] 
 
        result_text = f"Gender: {gender}\nAge   : {age[1:-1]}" 
        return result_text 
 
    return "No valid face found", 200 
 
if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True) 
 
 
 
 
 
