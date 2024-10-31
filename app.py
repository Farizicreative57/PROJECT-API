import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io

model = tf.keras.models.load_model(r'python-service/model_path.h5')

app = Flask(__name__)

className = {0: "PEREMPUAN", 1: "LAKI-LAKI"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    image_file = request.files['image']
    
    try:
        image = Image.open(image_file)
        image = image.resize((80, 60))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predictions_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        class_name = className[predictions_class]
        return jsonify({'prediction': class_name, 'accuracy': round(confidence, 2)})
    except Exception as e:
        print(f"error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)