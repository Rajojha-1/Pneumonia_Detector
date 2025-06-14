from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the model once
model = tf.keras.models.load_model('model/pneumonia_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save uploaded file to static for display
        file_path = os.path.join('static', 'uploaded.png')
        img = Image.open(file).convert('RGB')
        img.save(file_path)

        # Preprocess for model
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        pred_value = pred[0][0]

        prediction = 'Pneumonia' if pred_value > 0.5 else 'Normal'
        accuracy = pred_value * 100 if pred_value > 0.5 else (1 - pred_value) * 100

        return render_template('index.html',
                               prediction=prediction,
                               accuracy=round(accuracy, 2),
                               result_img='uploaded.png')

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
#added comment to trigger redeploy
