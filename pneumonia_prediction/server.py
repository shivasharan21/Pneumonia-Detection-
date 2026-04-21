# server.py
import io, json, os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

DEVICE = torch.device('cpu')
MODEL_PATH = 'pneumonia_best.pth'

app = Flask(__name__, static_folder='static')
CORS(app)

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, 2)
    )
    return m

model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"'{MODEL_PATH}' not found. Place it in the same folder as server.py")
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✓ Model loaded from {MODEL_PATH}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_bytes):
    img    = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    idx        = probs.argmax().item()
    return {
        'label':      CLASS_NAMES[idx],
        'confidence': round(probs[idx].item() * 100, 2),
        'probabilities': {
            'NORMAL':    round(probs[0].item() * 100, 2),
            'PNEUMONIA': round(probs[1].item() * 100, 2),
        }
    }

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
    try:
        result = predict(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"\n⚠  {e}\n")
    print("\n🫁  Server running → http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)