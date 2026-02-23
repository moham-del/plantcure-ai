from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import uuid

app = Flask(__name__)
app.secret_key = 'plantcure_secret_key_2024'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
print("🌿 Loading AI Model...")
model = tf.keras.models.load_model('model/plantcure_model.h5')
with open('model/class_names.json', 'r') as f:
    class_names = json.load(f)
print("✅ Model Loaded!")

# Disease Solutions Database
disease_solutions = {
    "Tomato_Early_blight": {
        "disease": "Tomato Early Blight",
        "severity": "Medium",
        "cause": "Fungal infection by Alternaria solani",
        "symptoms": "Dark brown spots with yellow rings on leaves",
        "solutions": [
            "Remove infected leaves immediately",
            "Apply copper-based fungicide",
            "Avoid overhead watering",
            "Ensure proper air circulation",
            "Rotate crops next season"
        ],
        "fertilizer": "NPK 10-10-10 balanced fertilizer",
        "organic": "Neem oil spray every 7 days",
        "recovery_days": "14-21 days"
    },
    "Tomato_Late_blight": {
        "disease": "Tomato Late Blight",
        "severity": "High",
        "cause": "Water mold Phytophthora infestans",
        "symptoms": "Dark water-soaked spots, white mold under leaves",
        "solutions": [
            "Remove and destroy infected plants",
            "Apply Mancozeb fungicide",
            "Improve drainage system",
            "Space plants for airflow",
            "Use resistant varieties"
        ],
        "fertilizer": "Potassium-rich fertilizer (K2O)",
        "organic": "Bordeaux mixture spray",
        "recovery_days": "21-30 days"
    },
    "Tomato_healthy": {
        "disease": "Healthy Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant looks healthy and green",
        "solutions": [
            "Continue regular watering",
            "Maintain proper sunlight",
            "Regular fertilization",
            "Monitor for pests weekly",
            "Keep soil pH 6.0-6.8"
        ],
        "fertilizer": "NPK 20-20-20 for maintenance",
        "organic": "Compost tea every 2 weeks",
        "recovery_days": "No treatment needed"
    }
}

def get_solution(class_name):
    for key in disease_solutions:
        if key.lower() in class_name.lower():
            return disease_solutions[key]
    return {
        "disease": class_name.replace('_', ' '),
        "severity": "Medium",
        "cause": "Fungal or bacterial infection",
        "symptoms": "Visible spots or discoloration on leaves",
        "solutions": [
            "Consult local agricultural expert",
            "Apply broad-spectrum fungicide",
            "Remove severely infected leaves",
            "Improve irrigation practices",
            "Check soil nutrient levels"
        ],
        "fertilizer": "NPK 15-15-15 balanced fertilizer",
        "organic": "Neem oil spray every 7 days",
        "recovery_days": "14-21 days"
    }

def predict_disease(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index]) * 100
    class_name = class_names[predicted_index]
    solution = get_solution(class_name)
    return class_name, confidence, solution

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/guest_login')
def guest_login():
    session['user'] = 'Guest User'
    session['email'] = 'guest@plantcure.ai'
    session['photo'] = ''
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return jsonify({'error': 'Please login first'})
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    class_name, confidence, solution = predict_disease(filepath)
    return jsonify({
        'class_name': class_name,
        'confidence': round(confidence, 2),
        'solution': solution,
        'image_path': filename
    })
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# (மேல இருக்கற imports-ஓட சேர்த்து இதை add பண்ணு)

@app.route('/google_login', methods=['POST'])
def google_login():
    data = request.get_json()
    session['user'] = data.get('name', 'User')
    session['email'] = data.get('email', '')
    session['photo'] = data.get('photo', '')
    session['uid'] = data.get('uid', '')
    return jsonify({'success': True})
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)