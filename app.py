from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from PIL import Image
import numpy as np
import json
import os
import uuid

app = Flask(__name__)
app.secret_key = 'plantcure_secret_key_2024'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load Model safely
model = None
class_names = []

try:
    import tensorflow as tf
    if os.path.exists('model/plantcure_model.h5'):
        model = tf.keras.models.load_model('model/plantcure_model.h5')
        with open('model/class_names.json', 'r') as f:
            class_names = json.load(f)
        print("✅ Model Loaded!")
    else:
        print("⚠️ Model not found - Running in Demo mode")
except Exception as e:
    print(f"⚠️ Model load error: {e} - Demo mode active")

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
    },
    "Potato_Late_blight": {
        "disease": "Potato Late Blight",
        "severity": "High",
        "cause": "Phytophthora infestans fungus",
        "symptoms": "Brown-black lesions on leaves and stems",
        "solutions": [
            "Remove infected plants immediately",
            "Apply fungicide spray",
            "Improve soil drainage",
            "Avoid overhead irrigation",
            "Use certified disease-free seeds"
        ],
        "fertilizer": "High potassium fertilizer",
        "organic": "Copper sulfate spray",
        "recovery_days": "21-28 days"
    },
    "Corn_Common_rust": {
        "disease": "Corn Common Rust",
        "severity": "Medium",
        "cause": "Puccinia sorghi fungus",
        "symptoms": "Small brown pustules on both leaf surfaces",
        "solutions": [
            "Apply fungicide at early stage",
            "Plant resistant varieties",
            "Remove heavily infected leaves",
            "Ensure good air circulation",
            "Monitor regularly"
        ],
        "fertilizer": "Nitrogen-rich fertilizer",
        "organic": "Sulfur-based spray",
        "recovery_days": "14-20 days"
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
    # Real model prediction
    if model is not None:
        try:
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index]) * 100
            class_name = class_names[predicted_index]
            solution = get_solution(class_name)
            return class_name, round(confidence, 2), solution
        except Exception as e:
            print(f"Prediction error: {e}")

    # Demo mode - random result for testing
    import random
    demo_diseases = list(disease_solutions.keys())
    class_name = random.choice(demo_diseases)
    confidence = round(random.uniform(75, 98), 2)
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
    return redirect(url_for('dashboard'))

@app.route('/google_login', methods=['POST'])
def google_login():
    data = request.get_json()
    session['user'] = data.get('name', 'User')
    session['email'] = data.get('email', '')
    session['photo'] = data.get('photo', '')
    return jsonify({'success': True})

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
        'confidence': confidence,
        'solution': solution
    })
from flask import send_from_directory

@app.route('/static/sw.js')
def sw():
    return send_from_directory('static', 'sw.js',
    mimetype='application/javascript')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
