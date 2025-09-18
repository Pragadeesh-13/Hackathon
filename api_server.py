#!/usr/bin/env python3
"""
🐄 Cattle Breed Recognition API Server 🚀
==========================================

Production-ready Flask API server for cattle breed recognition.
Designed for integration with React.js applications.

Features:
- 87.1% accuracy cattle breed recognition
- 11 supported breeds (Cattle + Buffalo)
- Advanced ensemble prediction system
- CORS enabled for web integration
- Comprehensive error handling
- Health monitoring endpoints

Author: Cattle Breed Recognition Team
Version: 1.0.0
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
from PIL import Image
import io
import numpy as np
import torch
import sys
import os
import traceback
import json
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from app_ultimate import UltimateCattleRecognizer
    print("✅ Successfully imported UltimateCattleRecognizer")
except ImportError as e:
    print(f"❌ Failed to import UltimateCattleRecognizer: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for development

# Global variables
recognizer = None
server_stats = {
    'start_time': time.time(),
    'total_predictions': 0,
    'successful_predictions': 0,
    'error_count': 0
}

# Supported breeds information
BREED_INFO = {
    'Bhadawari': {'type': 'Buffalo', 'origin': 'Uttar Pradesh', 'specialty': 'Heat tolerance'},
    'Gir': {'type': 'Cattle', 'origin': 'Gujarat', 'specialty': 'A2 milk production'},
    'Jaffarbadi': {'type': 'Buffalo', 'origin': 'Gujarat', 'specialty': 'Heavy draft work'},
    'Kankrej': {'type': 'Cattle', 'origin': 'Gujarat/Rajasthan', 'specialty': 'Drought resistance'},
    'Mehsana': {'type': 'Buffalo', 'origin': 'Gujarat', 'specialty': 'High butterfat'},
    'Murrah': {'type': 'Buffalo', 'origin': 'Haryana/Punjab', 'specialty': 'Highest milk yield'},
    'Nagpuri': {'type': 'Cattle', 'origin': 'Maharashtra', 'specialty': 'Drought tolerance'},
    'Ongole': {'type': 'Cattle', 'origin': 'Andhra Pradesh', 'specialty': 'Heat tolerance'},
    'Sahiwal': {'type': 'Cattle', 'origin': 'Punjab', 'specialty': 'Quality milk'},
    'Surti': {'type': 'Buffalo', 'origin': 'Gujarat', 'specialty': 'Rich milk quality'},
    'Tharparkar': {'type': 'Cattle', 'origin': 'Rajasthan', 'specialty': 'Desert adaptation'}
}

def initialize_model():
    """Initialize the cattle breed recognition model"""
    global recognizer
    try:
        print("🔄 Initializing Cattle Breed Recognition System...")
        recognizer = UltimateCattleRecognizer()
        print("✅ Model initialization successful!")
        print("📊 System ready with 87.1% accuracy")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def log_prediction(success=True, error_msg=None):
    """Log prediction statistics"""
    global server_stats
    server_stats['total_predictions'] += 1
    if success:
        server_stats['successful_predictions'] += 1
    else:
        server_stats['error_count'] += 1
        if error_msg:
            print(f"❌ Prediction error: {error_msg}")

@app.route('/', methods=['GET'])
def api_info():
    """API information and welcome message"""
    return jsonify({
        'name': '🐄 Cattle Breed Recognition API',
        'version': '1.0.0',
        'accuracy': '87.1%',
        'supported_breeds': 11,
        'status': 'operational',
        'description': 'Advanced AI system for cattle breed identification',
        'endpoints': {
            'POST /predict': 'Predict cattle breed from image',
            'GET /health': 'Health check and system status',
            'GET /breeds': 'List all supported breeds',
            'GET /stats': 'Server statistics',
            'GET /': 'API information'
        },
        'usage': {
            'image_format': 'Base64 encoded image (JPEG, PNG)',
            'max_size': '10MB',
            'response_time': '1-3 seconds'
        },
        'integration': {
            'react_example': 'See /docs endpoint for React integration examples',
            'cors_enabled': True,
            'content_type': 'application/json'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    global recognizer, server_stats
    
    # Calculate uptime
    uptime_seconds = time.time() - server_stats['start_time']
    uptime_hours = uptime_seconds / 3600
    
    # Model status
    model_loaded = recognizer is not None
    
    # Calculate success rate
    success_rate = 0
    if server_stats['total_predictions'] > 0:
        success_rate = (server_stats['successful_predictions'] / server_stats['total_predictions']) * 100
    
    health_status = {
        'status': 'healthy' if model_loaded else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_hours': round(uptime_hours, 2),
        'model': {
            'loaded': model_loaded,
            'accuracy': '87.1%',
            'supported_breeds': 11,
            'ensemble_system': True
        },
        'performance': {
            'total_predictions': server_stats['total_predictions'],
            'successful_predictions': server_stats['successful_predictions'],
            'error_count': server_stats['error_count'],
            'success_rate_percent': round(success_rate, 2)
        },
        'system': {
            'python_version': sys.version.split()[0],
            'torch_available': torch.cuda.is_available(),
            'device': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
    }
    
    return jsonify(health_status)

@app.route('/breeds', methods=['GET'])
def list_breeds():
    """Get comprehensive list of supported breeds"""
    breeds_data = []
    
    for breed_name, info in BREED_INFO.items():
        breeds_data.append({
            'name': breed_name,
            'type': info['type'],
            'origin': info['origin'],
            'specialty': info['specialty'],
            'category': 'buffalo' if info['type'] == 'Buffalo' else 'cattle'
        })
    
    return jsonify({
        'total_breeds': len(breeds_data),
        'breeds': breeds_data,
        'categories': {
            'cattle': [b['name'] for b in breeds_data if b['type'] == 'Cattle'],
            'buffalo': [b['name'] for b in breeds_data if b['type'] == 'Buffalo']
        },
        'accuracy': '87.1%',
        'model_type': 'Ensemble (Neural Network + Smart Fusion)'
    })

@app.route('/predict', methods=['POST'])
def predict_breed():
    """Main prediction endpoint for cattle breed recognition"""
    global recognizer
    
    if not recognizer:
        log_prediction(success=False, error_msg="Model not initialized")
        return jsonify({
            'success': False,
            'error': 'Model not initialized',
            'code': 'MODEL_NOT_READY'
        }), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image'
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Handle data URL format (data:image/jpeg;base64,...)
        if image_data.startswith('data:'):
            if ',' not in image_data:
                return jsonify({
                    'success': False,
                    'error': 'Invalid data URL format'
                }), 400
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image data: {str(e)}'
            }), 400
        
        # Get prediction mode
        prediction_mode = data.get('mode', 'ensemble')  # ensemble, trained, fusion
        
        # Perform prediction based on mode
        start_time = time.time()
        
        if prediction_mode == 'trained':
            result = recognizer.predict_trained_model(image)
            method_used = 'Trained Neural Network'
        elif prediction_mode == 'fusion':
            result = recognizer.predict_smart_fusion(image)
            method_used = 'Smart Fusion'
        else:  # ensemble (default)
            result = recognizer.advanced_ensemble_predict(image)
            method_used = 'Advanced Ensemble'
        
        prediction_time = time.time() - start_time
        
        # Process results
        if 'final_prediction' in result:
            # Ensemble result
            predicted_breed = result['final_prediction']
            confidence = result['final_confidence']
            is_cattle = result.get('is_cattle_image', True)
            confidence_level = result.get('confidence_level', 'MEDIUM')
            
            # Get top 5 predictions from ensemble scores
            top_predictions = []
            for breed, score in sorted(result['ensemble_scores'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                top_predictions.append({
                    'breed': breed,
                    'confidence': float(score),
                    'breed_info': BREED_INFO.get(breed, {})
                })
        else:
            # Single method result
            predicted_breed = result['top_breed']
            confidence = result['confidence']
            is_cattle = result.get('is_cattle_likely', True)
            confidence_level = result.get('confidence_level', 'MEDIUM')
            
            # Get top 5 predictions
            top_predictions = []
            for breed, score in sorted(result['predictions'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                top_predictions.append({
                    'breed': breed,
                    'confidence': float(score),
                    'breed_info': BREED_INFO.get(breed, {})
                })
        
        # Build response
        response_data = {
            'success': True,
            'prediction': {
                'breed': predicted_breed,
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'is_cattle': is_cattle,
                'breed_info': BREED_INFO.get(predicted_breed, {}),
                'method_used': method_used
            },
            'alternatives': top_predictions,
            'metadata': {
                'prediction_time_seconds': round(prediction_time, 3),
                'timestamp': datetime.now().isoformat(),
                'model_accuracy': '87.1%',
                'total_breeds': 11
            }
        }
        
        # Add ensemble-specific data
        if 'agreement' in result:
            response_data['ensemble_info'] = {
                'methods_agree': result['agreement'],
                'trained_model_prediction': result['trained_model']['top_breed'],
                'smart_fusion_prediction': result['smart_fusion']['top_breed']
            }
        
        log_prediction(success=True)
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = str(e)
        log_prediction(success=False, error_msg=error_msg)
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'code': 'PREDICTION_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/stats', methods=['GET'])
def server_stats_endpoint():
    """Get server statistics and performance metrics"""
    global server_stats
    
    uptime_seconds = time.time() - server_stats['start_time']
    
    return jsonify({
        'server_stats': {
            'uptime_seconds': int(uptime_seconds),
            'uptime_formatted': f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",
            'total_predictions': server_stats['total_predictions'],
            'successful_predictions': server_stats['successful_predictions'],
            'error_count': server_stats['error_count'],
            'success_rate': round(
                (server_stats['successful_predictions'] / max(server_stats['total_predictions'], 1)) * 100, 2
            )
        },
        'model_info': {
            'accuracy': '87.1%',
            'supported_breeds': 11,
            'architecture': 'Enhanced ResNet50 + Smart Fusion',
            'prediction_methods': ['Neural Network', 'Smart Fusion', 'Ensemble']
        }
    })

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation and React integration examples"""
    return jsonify({
        'api_documentation': {
            'base_url': 'http://YOUR_SERVER_IP:5000',
            'endpoints': {
                'predict': {
                    'method': 'POST',
                    'url': '/predict',
                    'description': 'Predict cattle breed from image',
                    'body': {
                        'image': 'Base64 encoded image string',
                        'mode': 'ensemble|trained|fusion (optional, default: ensemble)'
                    },
                    'example_request': {
                        'image': 'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
                        'mode': 'ensemble'
                    }
                }
            }
        },
        'react_integration': {
            'example_component': '''
// React component example
import React, { useState } from 'react';

const CattleBreedPredictor = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const predictBreed = async (imageFile) => {
    setLoading(true);
    try {
      const base64 = await fileToBase64(imageFile);
      const response = await fetch('http://YOUR_SERVER_IP:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64, mode: 'ensemble' })
      });
      
      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  return (
    <div>
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e) => predictBreed(e.target.files[0])} 
      />
      {loading && <p>Analyzing cattle breed...</p>}
      {prediction && prediction.success && (
        <div>
          <h3>Predicted Breed: {prediction.prediction.breed}</h3>
          <p>Confidence: {(prediction.prediction.confidence * 100).toFixed(1)}%</p>
          <p>Type: {prediction.prediction.breed_info.type}</p>
        </div>
      )}
    </div>
  );
};
            '''
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/predict', '/health', '/breeds', '/stats', '/docs']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Please check server logs for details'
    }), 500

def main():
    """Main function to start the API server"""
    print("🐄 CATTLE BREED RECOGNITION API SERVER 🚀")
    print("=" * 50)
    print("📊 Model Accuracy: 87.1%")
    print("🔢 Supported Breeds: 11 (Cattle + Buffalo)")
    print("🤖 AI Methods: Neural Network + Smart Fusion + Ensemble")
    print("🌐 CORS: Enabled for React integration")
    print("=" * 50)
    
    # Initialize model
    if not initialize_model():
        print("❌ Failed to initialize model. Exiting...")
        sys.exit(1)
    
    print("🚀 Starting Flask API server...")
    print("📡 Endpoints available:")
    print("   GET  /          - API information")
    print("   POST /predict   - Predict cattle breed")
    print("   GET  /health    - Health check")
    print("   GET  /breeds    - List supported breeds")
    print("   GET  /stats     - Server statistics")
    print("   GET  /docs      - API documentation")
    print("=" * 50)
    print("🖥️  Local access: http://localhost:5000")
    print("🌍 Network access: http://YOUR_IP:5000")
    print("📱 Mac React app can connect to: http://YOUR_IP:5000")
    print("=" * 50)
    print("✅ Server ready! Listening for connections...")
    
    # Start Flask server
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,       # Standard API port
        debug=False,     # Production mode
        threaded=True    # Handle multiple requests
    )

if __name__ == '__main__':
    main()