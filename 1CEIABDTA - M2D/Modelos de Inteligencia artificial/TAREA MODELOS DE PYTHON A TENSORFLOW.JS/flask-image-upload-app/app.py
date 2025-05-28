from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import json

app = Flask(__name__)

# Cargar el modelo Keras (debe estar convertido a .h5)
MODEL_PATH = os.path.join('static', 'flower_classifier_tfjs', 'flower_classifier.h5')
CLASS_NAMES_PATH = os.path.join('static', 'flower_classifier_tfjs', 'class_names.json')
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))  # Ajusta al tamaño de entrada del modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_name = class_names[class_idx]
    confidence = float(preds[0][class_idx])
    return jsonify({'class': class_name, 'confidence': confidence})

@app.route('/upload')
def upload():
    return render_template('upload.html')