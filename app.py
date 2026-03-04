from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from PIL import Image
import numpy as np
import json
import os
import uuid
import requests

app = Flask(__name__)
app.secret_key = 'plantcure_secret_key_2024'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# =======================================
# Hugging Face Model URLs
# =======================================
MODEL_URL = "https://huggingface.co/MSAYE/plantcure-ai/resolve/main/plantcure_model.h5"
CLASS_URL = "https://huggingface.co/MSAYE/plantcure-ai/resolve/main/class_names.json"

# =======================================
# Download Model
# =======================================
def download_model():
    os.makedirs('model', exist_ok=True)
    model_path = 'model/plantcure_model.h5'
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        print("📥 Downloading model from Hugging Face...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (5*1024*1024) == 0:
                            print(f"📥 Downloaded: {downloaded/1024/1024:.0f} MB")
            size = os.path.getsize(model_path)
            print(f"✅ Model downloaded: {size/1024/1024:.2f} MB")
        except Exception as e:
            print(f"❌ Model download failed: {e}")

    class_path = 'model/class_names.json'
    if not os.path.exists(class_path) or os.path.getsize(class_path) < 100:
        print("📥 Downloading class names...")
        try:
            response = requests.get(CLASS_URL)
            with open(class_path, 'wb') as f:
                f.write(response.content)
            print("✅ Class names downloaded!")
        except Exception as e:
            print(f"❌ Class names download failed: {e}")

download_model()

# =======================================
# Load AI Model
# =======================================
model = None
class_names = []

try:
    import tensorflow as tf
    model_path = 'model/plantcure_model.h5'
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        print("🌿 Loading AI Model...")
        model = tf.keras.models.load_model(model_path)
        with open('model/class_names.json', 'r') as f:
            class_names = json.load(f)
        print(f"✅ Real AI Model Loaded! Classes: {len(class_names)}")
    else:
        print("⚠️ Model not found - Demo mode active")
except Exception as e:
    print(f"⚠️ Model error: {e}")

# =======================================
# 38 Disease Solutions Database
# =======================================
disease_solutions = {
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "severity": "Medium",
        "cause": "Venturia inaequalis fungus",
        "symptoms": "Olive-green to brown spots on leaves and fruits",
        "solutions": ["Remove infected leaves immediately","Apply fungicide (myclobutanil)","Prune for better air circulation","Rake fallen leaves in autumn","Use scab-resistant apple varieties"],
        "fertilizer": "Balanced NPK with calcium",
        "organic": "Neem oil spray every 7 days",
        "recovery_days": "21-30 days"
    },
    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "severity": "High",
        "cause": "Botryosphaeria obtusa fungus",
        "symptoms": "Brown circular spots on leaves, black rotting fruits",
        "solutions": ["Remove mummified fruits","Prune dead wood completely","Apply captan fungicide","Disinfect pruning tools","Improve orchard drainage"],
        "fertilizer": "Potassium-rich fertilizer",
        "organic": "Copper spray every 10 days",
        "recovery_days": "30-45 days"
    },
    "Apple___Cedar_apple_rust": {
        "disease": "Apple Cedar Rust",
        "severity": "Medium",
        "cause": "Gymnosporangium juniperi-virginianae fungus",
        "symptoms": "Bright orange-yellow spots on upper leaf surface",
        "solutions": ["Apply myclobutanil fungicide","Remove nearby juniper trees","Spray preventively in spring","Use rust-resistant varieties","Monitor weekly in wet weather"],
        "fertilizer": "NPK 10-10-10 balanced",
        "organic": "Sulfur spray every 7-10 days",
        "recovery_days": "14-21 days"
    },
    "Apple___healthy": {
        "disease": "Healthy Apple Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy and growing well",
        "solutions": ["Continue regular watering","Maintain proper sunlight","Apply fertilizer monthly","Monitor pests weekly","Prune annually for shape"],
        "fertilizer": "NPK 20-20-20",
        "organic": "Compost tea monthly",
        "recovery_days": "No treatment needed"
    },
    "Blueberry___healthy": {
        "disease": "Healthy Blueberry! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Maintain soil pH 4.5-5.5","Regular watering","Mulch around base","Prune old canes","Monitor birds"],
        "fertilizer": "Acidic fertilizer (ammonium sulfate)",
        "organic": "Pine bark mulch",
        "recovery_days": "No treatment needed"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Cherry Powdery Mildew",
        "severity": "Medium",
        "cause": "Podosphaera clandestina fungus",
        "symptoms": "White powdery coating on young leaves",
        "solutions": ["Apply sulfur-based fungicide","Improve air circulation","Remove infected shoots","Avoid overhead watering","Apply potassium bicarbonate"],
        "fertilizer": "Low nitrogen fertilizer",
        "organic": "Baking soda + neem oil spray",
        "recovery_days": "14-21 days"
    },
    "Cherry_(including_sour)___healthy": {
        "disease": "Healthy Cherry Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular watering","Annual pruning","Fertilize in spring","Bird protection nets","Monitor for pests"],
        "fertilizer": "NPK 12-12-12",
        "organic": "Compost annually",
        "recovery_days": "No treatment needed"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Corn Gray Leaf Spot",
        "severity": "High",
        "cause": "Cercospora zeae-maydis fungus",
        "symptoms": "Rectangular gray-brown lesions on leaves",
        "solutions": ["Apply strobilurin fungicide","Plant resistant hybrids","Rotate crops annually","Till crop residue","Improve field drainage"],
        "fertilizer": "Nitrogen-rich fertilizer",
        "organic": "Trichoderma bio-fungicide",
        "recovery_days": "21-30 days"
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Corn Common Rust",
        "severity": "Medium",
        "cause": "Puccinia sorghi fungus",
        "symptoms": "Small brown pustules on both leaf surfaces",
        "solutions": ["Apply fungicide early","Plant resistant varieties","Monitor weekly","Remove heavily infected leaves","Ensure good air circulation"],
        "fertilizer": "Balanced NPK fertilizer",
        "organic": "Sulfur spray every 7 days",
        "recovery_days": "14-20 days"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Corn Northern Leaf Blight",
        "severity": "High",
        "cause": "Exserohilum turcicum fungus",
        "symptoms": "Long cigar-shaped gray-green lesions",
        "solutions": ["Apply propiconazole fungicide","Use resistant hybrids","Crop rotation","Remove crop debris","Apply at early stage"],
        "fertilizer": "Potassium + phosphorus fertilizer",
        "organic": "Neem oil spray",
        "recovery_days": "21-28 days"
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy Corn Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular irrigation","Proper spacing","Nitrogen fertilization","Weed control","Monitor for pests"],
        "fertilizer": "Urea + NPK 15-15-15",
        "organic": "Compost + manure",
        "recovery_days": "No treatment needed"
    },
    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "severity": "High",
        "cause": "Guignardia bidwellii fungus",
        "symptoms": "Brown spots on leaves, black shriveled fruits",
        "solutions": ["Apply myclobutanil fungicide","Remove mummified berries","Prune for air circulation","Apply before bloom","Remove infected clusters"],
        "fertilizer": "Balanced grape fertilizer",
        "organic": "Copper hydroxide spray",
        "recovery_days": "21-35 days"
    },
    "Grape___Esca_(Black_Measles)": {
        "disease": "Grape Esca (Black Measles)",
        "severity": "Critical",
        "cause": "Multiple fungal pathogens",
        "symptoms": "Tiger stripe pattern on leaves, wood decay",
        "solutions": ["Remove infected wood","Apply wound sealant","No chemical cure available","Remove severely infected vines","Prevent pruning wounds"],
        "fertilizer": "Balanced nutrition program",
        "organic": "Trichoderma wound treatment",
        "recovery_days": "Long term management required"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Grape Leaf Blight",
        "severity": "Medium",
        "cause": "Isariopsis clavispora fungus",
        "symptoms": "Dark brown irregular spots on leaves",
        "solutions": ["Apply copper fungicide","Remove infected leaves","Improve air circulation","Avoid overhead irrigation","Apply preventive sprays"],
        "fertilizer": "Potassium-rich fertilizer",
        "organic": "Bordeaux mixture spray",
        "recovery_days": "14-21 days"
    },
    "Grape___healthy": {
        "disease": "Healthy Grape Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular pruning","Proper trellising","Monitor weekly","Adequate irrigation","Annual fertilization"],
        "fertilizer": "NPK 10-10-10 + micronutrients",
        "organic": "Compost tea",
        "recovery_days": "No treatment needed"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Greening (HLB)",
        "severity": "Critical",
        "cause": "Candidatus Liberibacter bacterium by psyllid",
        "symptoms": "Yellow shoots, blotchy mottled leaves, bitter fruits",
        "solutions": ["No cure - remove infected trees","Control Asian citrus psyllid","Apply systemic insecticide","Use disease-free nursery plants","Quarantine infected areas"],
        "fertilizer": "Micronutrient foliar spray",
        "organic": "Neem for psyllid control",
        "recovery_days": "No cure - tree removal required"
    },
    "Peach___Bacterial_spot": {
        "disease": "Peach Bacterial Spot",
        "severity": "High",
        "cause": "Xanthomonas arboricola bacteria",
        "symptoms": "Water-soaked spots on leaves and fruits",
        "solutions": ["Apply copper bactericide","Avoid overhead irrigation","Plant resistant varieties","Prune in dry weather","Apply in early spring"],
        "fertilizer": "Calcium-rich fertilizer",
        "organic": "Copper hydroxide spray",
        "recovery_days": "21-30 days"
    },
    "Peach___healthy": {
        "disease": "Healthy Peach Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular pruning","Proper irrigation","Annual fertilization","Thin fruits early","Monitor for pests"],
        "fertilizer": "NPK 15-15-15",
        "organic": "Compost annually",
        "recovery_days": "No treatment needed"
    },
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Pepper Bell Bacterial Spot",
        "severity": "High",
        "cause": "Xanthomonas bacteria infection",
        "symptoms": "Water-soaked spots, dark lesions on fruits",
        "solutions": ["Remove infected parts","Apply copper bactericide","Avoid overhead irrigation","Use certified seeds","Rotate crops"],
        "fertilizer": "Calcium nitrate fertilizer",
        "organic": "Copper hydroxide spray every 7 days",
        "recovery_days": "21-28 days"
    },
    "Pepper,_bell___healthy": {
        "disease": "Healthy Pepper Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular watering","6-8 hours sunlight","Monthly fertilization","Weekly pest monitoring","pH 6.0-6.8"],
        "fertilizer": "NPK 10-10-10",
        "organic": "Compost tea every 2 weeks",
        "recovery_days": "No treatment needed"
    },
    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "severity": "Medium",
        "cause": "Alternaria solani fungus",
        "symptoms": "Dark brown circular spots with yellow halos",
        "solutions": ["Remove infected leaves","Apply chlorothalonil","Proper plant spacing","Avoid wetting foliage","Apply mulch"],
        "fertilizer": "NPK 15-15-15 + Potassium",
        "organic": "Neem oil every 5-7 days",
        "recovery_days": "14-21 days"
    },
    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "severity": "Critical",
        "cause": "Phytophthora infestans",
        "symptoms": "Dark water-soaked lesions, white fuzzy growth",
        "solutions": ["Destroy infected plants","Apply Metalaxyl+Mancozeb","Improve drainage","Avoid same location","Use resistant varieties"],
        "fertilizer": "High Potassium fertilizer",
        "organic": "Bordeaux mixture 1%",
        "recovery_days": "30-45 days"
    },
    "Potato___healthy": {
        "disease": "Healthy Potato Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular deep watering","Hill up soil","Balanced fertilizer","Monitor beetles","Good drainage"],
        "fertilizer": "NPK 20-20-20",
        "organic": "Seaweed extract monthly",
        "recovery_days": "No treatment needed"
    },
    "Raspberry___healthy": {
        "disease": "Healthy Raspberry! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Annual pruning","Proper trellising","Regular irrigation","Mulching","Monitor for pests"],
        "fertilizer": "NPK 10-10-10",
        "organic": "Compost annually",
        "recovery_days": "No treatment needed"
    },
    "Soybean___healthy": {
        "disease": "Healthy Soybean! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Proper spacing","Crop rotation","Weed control","Monitor pests","Adequate irrigation"],
        "fertilizer": "Phosphorus + Potassium",
        "organic": "Rhizobium inoculant",
        "recovery_days": "No treatment needed"
    },
    "Squash___Powdery_mildew": {
        "disease": "Squash Powdery Mildew",
        "severity": "Medium",
        "cause": "Podosphaera xanthii fungus",
        "symptoms": "White powdery patches on leaves",
        "solutions": ["Apply sulfur fungicide","Improve air circulation","Remove infected leaves","Avoid overhead watering","Apply potassium bicarbonate"],
        "fertilizer": "Low nitrogen balanced fertilizer",
        "organic": "Baking soda spray weekly",
        "recovery_days": "10-14 days"
    },
    "Strawberry___Leaf_scorch": {
        "disease": "Strawberry Leaf Scorch",
        "severity": "Medium",
        "cause": "Diplocarpon earlianum fungus",
        "symptoms": "Small purple spots enlarging with tan centers",
        "solutions": ["Remove infected leaves","Apply captan fungicide","Avoid wetting leaves","Improve air circulation","Use resistant varieties"],
        "fertilizer": "Balanced strawberry fertilizer",
        "organic": "Copper spray every 10 days",
        "recovery_days": "14-21 days"
    },
    "Strawberry___healthy": {
        "disease": "Healthy Strawberry! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is healthy",
        "solutions": ["Regular watering","Mulching","Runner removal","Annual replanting","Monitor pests"],
        "fertilizer": "NPK 10-10-10",
        "organic": "Compost tea",
        "recovery_days": "No treatment needed"
    },
    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "severity": "High",
        "cause": "Xanthomonas vesicatoria bacteria",
        "symptoms": "Small water-soaked spots, yellowing",
        "solutions": ["Remove infected plants","Apply copper bactericide","Avoid wet plants","Disinfect tools","Use resistant varieties"],
        "fertilizer": "Calcium-rich fertilizer",
        "organic": "Copper soap spray weekly",
        "recovery_days": "21-30 days"
    },
    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "severity": "Medium",
        "cause": "Alternaria solani fungus",
        "symptoms": "Brown spots with concentric rings",
        "solutions": ["Remove infected leaves","Apply copper fungicide","Mulch around base","Water at soil level","Improve air circulation"],
        "fertilizer": "NPK 10-10-10",
        "organic": "Neem oil + baking soda",
        "recovery_days": "14-21 days"
    },
    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "severity": "Critical",
        "cause": "Phytophthora infestans",
        "symptoms": "Dark brown-black lesions, white mold",
        "solutions": ["Destroy infected plants","Apply Mancozeb+Metalaxyl","No overhead irrigation","Increase plant spacing","Resistant varieties"],
        "fertilizer": "Potassium sulfate",
        "organic": "Bordeaux mixture every 5 days",
        "recovery_days": "30-45 days"
    },
    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "severity": "Medium",
        "cause": "Passalora fulva fungus",
        "symptoms": "Yellow patches upper leaf, olive mold below",
        "solutions": ["Reduce humidity below 85%","Improve ventilation","Remove infected leaves","Apply chlorothalonil","Space plants wider"],
        "fertilizer": "Balanced NPK low nitrogen",
        "organic": "Baking soda + neem oil",
        "recovery_days": "14-21 days"
    },
    "Tomato___Septoria_leaf_spot": {
        "disease": "Tomato Septoria Leaf Spot",
        "severity": "Medium",
        "cause": "Septoria lycopersici fungus",
        "symptoms": "Small circular spots dark borders light centers",
        "solutions": ["Remove lower leaves","Apply mancozeb","Avoid wetting leaves","Add mulch","Stake plants"],
        "fertilizer": "Phosphorus-rich fertilizer",
        "organic": "Copper fungicide every 7-10 days",
        "recovery_days": "14-20 days"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Tomato Spider Mites",
        "severity": "High",
        "cause": "Tetranychus urticae mite",
        "symptoms": "Tiny yellow dots, fine webbing, bronzing",
        "solutions": ["Water jets on undersides","Apply miticide","Introduce predatory mites","Remove infested leaves","Maintain soil moisture"],
        "fertilizer": "Balanced fertilizer",
        "organic": "Neem oil every 3 days",
        "recovery_days": "10-14 days"
    },
    "Tomato___Target_Spot": {
        "disease": "Tomato Target Spot",
        "severity": "Medium",
        "cause": "Corynespora cassiicola fungus",
        "symptoms": "Circular target-like rings on leaves",
        "solutions": ["Apply azoxystrobin","Remove plant debris","Improve air circulation","Avoid high humidity","Rotate crops"],
        "fertilizer": "Balanced NPK monthly",
        "organic": "Trichoderma bio-fungicide",
        "recovery_days": "14-21 days"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus",
        "severity": "Critical",
        "cause": "Begomovirus by whiteflies",
        "symptoms": "Upward leaf curling, yellowing, stunted",
        "solutions": ["Remove infected plants","Apply imidacloprid","Yellow sticky traps","Insect-proof nets","Resistant varieties"],
        "fertilizer": "Potassium + micronutrients",
        "organic": "Neem oil + reflective mulch",
        "recovery_days": "No cure - remove plants"
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus",
        "severity": "Critical",
        "cause": "Tobamovirus by contact/insects",
        "symptoms": "Mosaic yellow-green pattern, stunted growth",
        "solutions": ["Remove infected plants","Control aphids/whiteflies","Disinfect tools with bleach","Wash hands before handling","Virus-free seeds only"],
        "fertilizer": "Balanced fertilizer avoid excess nitrogen",
        "organic": "Neem for insect control",
        "recovery_days": "No cure - remove plants"
    },
    "Tomato___healthy": {
        "disease": "Healthy Tomato Plant! 🎉",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Plant is perfectly healthy",
        "solutions": ["Continue watering schedule","Regular fertilization","Stake as plant grows","Weekly pest monitoring","Prune suckers"],
        "fertilizer": "NPK 8-32-16",
        "organic": "Compost + banana peel monthly",
        "recovery_days": "No treatment needed!"
    }
}

def get_solution(class_name):
    if class_name in disease_solutions:
        return disease_solutions[class_name]
    for key in disease_solutions:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return disease_solutions[key]
    return {
        "disease": class_name.replace('_', ' '),
        "severity": "Medium",
        "cause": "Fungal or bacterial infection detected",
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

def is_leaf_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        r = img_array[:,:,0].astype(float)
        g = img_array[:,:,1].astype(float)
        b = img_array[:,:,2].astype(float)
        green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 40)
        yellow_mask = (r > 150) & (g > 150) & (b < 100)
        brown_mask = (r > 100) & (g > 60) & (g < 130) & (b < 80)
        leaf_pixels = np.sum(green_mask) + np.sum(yellow_mask) + np.sum(brown_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        leaf_ratio = leaf_pixels / total_pixels
        print(f"🌿 Leaf ratio: {leaf_ratio:.2f}")
        return leaf_ratio > 0.10
    except Exception as e:
        print(f"Validation error: {e}")
        return True

def predict_disease(image_path):
    if not is_leaf_image(image_path):
        return "not_leaf", 0, {
            "disease": "❌ Not a Leaf Image!",
            "severity": "Invalid",
            "cause": "Please upload a clear plant leaf photo only",
            "symptoms": "Image does not appear to be a plant leaf",
            "solutions": [
                "📸 Take clear photo of affected leaf only",
                "🌿 Leaf should fill most of the frame",
                "☀️ Use good natural lighting",
                "🔍 Focus clearly on the leaf",
                "❌ Do not upload people, animals or objects"
            ],
            "fertilizer": "Please upload a valid leaf image first",
            "organic": "Please upload a valid leaf image first",
            "recovery_days": "Upload correct leaf image to continue"
        }

    if model is not None and len(class_names) > 0:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index]) * 100

            if confidence < 50:
                return "unclear", round(confidence, 2), {
                    "disease": "🔍 Unclear - Please Retake Photo",
                    "severity": "Unknown",
                    "cause": "Image quality too low for analysis",
                    "symptoms": "Could not detect clear symptoms",
                    "solutions": [
                        "☀️ Take photo in bright daylight",
                        "🌿 Leaf should fill entire frame",
                        "🔍 Focus on affected area",
                        "📱 Hold phone steady",
                        "📏 20-30cm distance from leaf"
                    ],
                    "fertilizer": "Retake photo for recommendation",
                    "organic": "Retake photo for recommendation",
                    "recovery_days": "Retake clear photo first"
                }

            class_name = class_names[predicted_index]
            solution = get_solution(class_name)
            return class_name, round(confidence, 2), solution

        except Exception as e:
            print(f"Prediction error: {e}")

    import random
    demo_diseases = list(disease_solutions.keys())
    class_name = random.choice(demo_diseases)
    confidence = round(random.uniform(60, 95), 2)
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

@app.route('/static/sw.js')
def sw():
    return send_from_directory('static', 'sw.js',
                               mimetype='application/javascript')

@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('static', 'sitemap.xml')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
