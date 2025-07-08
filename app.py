from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import pickle

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model and labels
model = tf.keras.models.load_model('brain_tumor_model.h5')
with open('labeling.pkl', 'rb') as f:
    labels = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            pred_class = np.argmax(prediction, axis=1)[0]
            
            if pred_class == 0:
                result = 'Glioma Tumor'
                result_color = 'bg-amber-100 border-amber-500 text-amber-800'
            elif pred_class == 1:
                result = 'No Tumor Detected'
                result_color = 'bg-emerald-100 border-emerald-500 text-emerald-800'
            elif pred_class == 2:
                result = 'Meningioma Tumor'
                result_color = 'bg-amber-100 border-amber-500 text-amber-800'
            else:
                result = 'Pituitary Tumor'
                result_color = 'bg-amber-100 border-amber-500 text-amber-800'
            
            confidence = float(np.max(prediction) * 100)
            
            return render_template('result.html', 
                                 result=result,
                                 result_color=result_color,
                                 confidence=round(confidence, 2),
                                 filename=filename)
    
    return render_template('index.html')

# Tambahkan route baru di app.py
@app.route('/preview', methods=['POST'])
def preview_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preview_' + filename)
        file.save(preview_path)
        return jsonify({
            'filename': 'preview_' + filename,
            'preview_url': url_for('static', filename='uploads/preview_' + filename)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/research')
def research():
    return render_template('research.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)