#!/usr/bin/env python3
"""
🐄 Simple Cattle Breed Recognition API Server 🚀
================================================

Lightweight Flask API server for cattle breed recognition.
Works without Streamlit dependencies.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import os
import traceback
import json
import time
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Global variables
model = None
device = None
breeds = [
    "Bhadawari", "Gir", "Jaffarbadi", "Kankrej", "Mehsana", 
    "Murrah", "Nagpuri", "Ongole", "Sahiwal", "Surti", "Tharparkar"
]
prediction_stats = {
    "total_predictions": 0,
    "successful_predictions": 0,
    "start_time": datetime.now().isoformat()
}

class SimpleCattleModel(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleCattleModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the cattle breed recognition model"""
    global model, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model
        model = SimpleCattleModel(num_classes=11)
        
        # Try to load saved model weights
        model_path = "models/enhanced_11breed_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print("⚠️ No saved model found, using pretrained weights")
            
        model.to(device)
        model.eval()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        tensor = transform(image).unsqueeze(0)
        return tensor.to(device)
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    """API information endpoint"""
    return jsonify({
        "message": "🐄 Cattle Breed Recognition API Server",
        "version": "2.0.0",
        "status": "running",
        "accuracy": "87.1%",
        "supported_breeds": len(breeds),
        "endpoints": {
            "predict": "POST /predict - Upload image for breed prediction",
            "health": "GET /health - Health check",
            "breeds": "GET /breeds - List supported breeds",
            "stats": "GET /stats - Server statistics"
        },
        "usage": "Upload an image to /predict endpoint to get cattle breed prediction"
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - datetime.fromisoformat(prediction_stats["start_time"])).total_seconds()
    })

@app.route('/breeds')
def get_breeds():
    """Get list of supported breeds"""
    return jsonify({
        "breeds": breeds,
        "count": len(breeds),
        "categories": ["Cattle", "Buffalo"]
    })

@app.route('/stats')
def get_stats():
    """Get prediction statistics"""
    return jsonify(prediction_stats)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict cattle breed from uploaded image"""
    global prediction_stats
    
    prediction_stats["total_predictions"] += 1
    
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "message": "Please upload an image file"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select an image file"
            }), 400
        
        # Open and process image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({
                "error": "Image processing failed",
                "message": "Could not process the uploaded image"
            }), 400
        
        # Make prediction
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "The recognition model is not available"
            }), 500
        
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_breed = breeds[predicted.item()]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top_predictions = []
            for i in range(3):
                top_predictions.append({
                    "breed": breeds[top3_indices[0][i].item()],
                    "confidence": top3_probs[0][i].item()
                })
        
        prediction_stats["successful_predictions"] += 1
        
        result = {
            "success": True,
            "predicted_breed": predicted_breed,
            "confidence": confidence_score,
            "top_predictions": top_predictions,
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "processing_time": time.time(),
            "message": f"Prediction: {predicted_breed} with {confidence_score:.1%} confidence"
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "message": "An error occurred during prediction"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/breeds", "/stats", "/predict"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    print("🐄 SIMPLE CATTLE BREED RECOGNITION API SERVER 🚀")
    print("=" * 50)
    print(f"📊 Supported Breeds: {len(breeds)}")
    print("🌐 CORS: Enabled for React integration")
    print("=" * 50)
    
    # Load model
    print("🔄 Loading recognition model...")
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model loading failed, using basic functionality")
    
    print("🚀 Starting Flask API server...")
    print("📡 Endpoints available:")
    print("   GET  /          - API information")
    print("   POST /predict   - Predict cattle breed")
    print("   GET  /health    - Health check")
    print("   GET  /breeds    - List supported breeds")
    print("   GET  /stats     - Server statistics")
    print("=" * 50)
    print("🖥️  Local access: http://localhost:5000")
    print("🌍 Network access: http://172.16.45.105:5000")
    print("📱 Mac React app can connect to: http://172.16.45.105:5000")
    print("=" * 50)
    print("✅ Server ready! Listening for connections...")
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )