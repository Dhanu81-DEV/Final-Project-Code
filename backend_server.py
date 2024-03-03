from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_filename = "cnn_model.h5"
cnn_model = load_model(model_filename)

import base64

def enhance_and_predict(input_image_path):
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_img_resized = cv2.resize(enhanced_img, (100, 100))
    input_data = np.expand_dims(np.array(enhanced_img_resized) / 255.0, axis=0)
    prediction = cnn_model.predict(input_data)[0]
 
    _, processed_image_data = cv2.imencode('.bmp', enhanced_img_resized)
    processed_image_base64 = base64.b64encode(processed_image_data).decode('utf-8')

    return 'Leukemia (ALL)' if prediction[1] > 0.5 else 'Normal', processed_image_base64


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        file_path = 'temp.bmp' 
        file.save(file_path)
        result, processed_image = enhance_and_predict(file_path)
        return jsonify({'result': result, 'processed_image': processed_image})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
