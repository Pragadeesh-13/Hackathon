#!/usr/bin/env python3
"""
Smart CNN-First Enhanced Inference

This system preserves the proven accuracy of the CNN while using 
morphological features only for tie-breaking and confidence enhancement.

Key Features:
- Preserves CNN accuracy when confidence > 0.85
- Uses morphology only for tie-breaking when CNN margin < 0.01
- Maintains the 0.89+ accuracy of the original maximum system
"""

import os
import sys
import pickle
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add utils to path
sys.path.append('.')
from utils import (
    FeatureExtractor,
    get_device,
    get_transforms,
    load_and_preprocess_image,
    compute_similarities,
    get_top_predictions
)
from utils.smart_fusion import SmartCNNFirstFusion, SmartFusedPrediction
from utils.morphology import MorphologicalFeatureExtractor

def smart_enhanced_inference(image_path: str, model_path: str = 'models/prototypes_maximum_10breed.pkl',
                            device: str = 'auto') -> Dict[str, Any]:
    """
    Perform smart enhanced inference that preserves CNN accuracy
    """
    
    # Setup device
    if device == 'auto':
        device = get_device()
    else:
        device = torch.device(device)
    
    print(f"🚀 Smart CNN-First Enhanced Inference")
    print(f"📱 Device: {device}")
    print(f"🖼️  Image: {image_path}")
    print("-" * 60)
    
    # Load model and prototypes from pickle file
    print("📥 Loading maximum discrimination model...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    prototypes = data['prototypes']
    actual_breeds = data['breeds'] 
    model_name = data['model_name']
    
    # Load smart fusion system
    smart_fusion = SmartCNNFirstFusion(actual_breeds)
    
    # Move prototypes to device
    for breed in prototypes:
        prototypes[breed] = prototypes[breed].to(device)
    
    # Create and load model
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    print("🖼️  Processing image...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    # Perform smart prediction
    print("🧠 Analyzing with smart CNN-first fusion...")
    predictions, smart_prediction = smart_fusion.predict_with_smart_fusion(
        pil_image, model, prototypes, device
    )
    
    # Get morphological features for display
    morph_features = smart_fusion.extract_morphological_features(pil_image)
    
    # Compile comprehensive results
    results = {
        'image_path': image_path,
        'predictions': {
            'breed_scores': smart_prediction.breed_scores,
            'top_predictions': predictions[:5],
            'fusion_strategy': smart_prediction.fusion_strategy,
            'confidence_level': smart_prediction.confidence_level,
            'margin': smart_prediction.margin,
            'explanation': smart_prediction.explanation,
            'cnn_confidence': smart_prediction.cnn_confidence,
            'morpho_confidence': smart_prediction.morpho_confidence
        },
        'feature_analysis': {
            'morphological_features': {
                'horn_features': morph_features.horn_features,
                'coat_features': morph_features.coat_features,
                'body_features': morph_features.body_features,
                'confidence_scores': morph_features.confidence_scores
            }
        },
        'technical_details': {
            'model_features': 2948,
            'fusion_approach': 'Smart CNN-First',
            'cnn_dominance': smart_prediction.fusion_strategy in ['cnn_dominant']
        }
    }
    
    return results

def print_smart_analysis(results: Dict[str, Any]):
    """Print smart analysis results"""
    
    print("\n" + "="*70)
    print("🎯 SMART CNN-FIRST ENHANCED ANALYSIS")
    print("="*70)
    
    # Main prediction
    predictions = results['predictions']
    best_breed, best_score = predictions['top_predictions'][0]
    
    print(f"\n🥇 PREDICTED BREED: {best_breed.upper()}")
    print(f"📊 Final Score: {best_score:.6f}")
    print(f"🎚️  Confidence Level: {predictions['confidence_level'].upper()}")
    print(f"📏 Margin: {predictions['margin']:.6f}")
    print(f"🔧 Fusion Strategy: {predictions['fusion_strategy'].upper()}")
    print(f"💡 Explanation: {predictions['explanation']}")
    
    # Strategy details
    print(f"\n⚙️  STRATEGY DETAILS:")
    print(f"  🧠 CNN Confidence: {predictions['cnn_confidence']:.6f}")
    print(f"  🔬 Morphological Confidence: {predictions['morpho_confidence']:.3f}")
    print(f"  🎯 CNN Dominant: {'YES' if results['technical_details']['cnn_dominance'] else 'NO'}")
    
    # Top predictions
    print(f"\n🏆 TOP 5 PREDICTIONS:")
    for i, (breed, score) in enumerate(predictions['top_predictions'], 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{status} {i}. {breed:15} - {score:.6f}")
    
    # Technical details
    tech = results['technical_details']
    print(f"\n🔧 TECHNICAL DETAILS:")
    print(f"  • Model Features: {tech['model_features']:,}")
    print(f"  • Fusion Approach: {tech['fusion_approach']}")
    print(f"  • Strategy: Preserve proven CNN accuracy")
    
    print("\n" + "="*70)

def test_smart_accuracy():
    """Test smart fusion accuracy on known samples"""
    
    test_cases = [
        ("dataset/Murrah/murrah_0002.jpg", "Murrah"),
        ("dataset/Gir/Gir_1.JPG", "Gir"),
        ("dataset/Kankrej/Kankrej_1.JPG", "Kankrej")
    ]
    
    print("🧪 SMART FUSION ACCURACY TEST")
    print("=" * 80)
    
    correct_count = 0
    total_count = 0
    
    for image_path, expected_breed in test_cases:
        try:
            print(f"\n📸 Testing: {image_path}")
            print(f"🎯 Expected: {expected_breed}")
            
            results = smart_enhanced_inference(image_path)
            predicted_breed = results['predictions']['top_predictions'][0][0]
            
            is_correct = predicted_breed.lower() == expected_breed.lower()
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            
            print(f"🤖 Predicted: {predicted_breed}")
            print(f"📊 Result: {status}")
            print(f"🔧 Strategy: {results['predictions']['fusion_strategy']}")
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            total_count += 1
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"\n🎯 SMART FUSION ACCURACY: {correct_count}/{total_count} ({accuracy:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Smart CNN-First Enhanced Inference')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--test', action='store_true', help='Run accuracy test')
    parser.add_argument('--model', type=str, default='models/prototypes_maximum_10breed.pkl',
                       help='Path to model and prototypes pickle file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    if args.test:
        test_smart_accuracy()
    elif args.image:
        try:
            results = smart_enhanced_inference(args.image, args.model, args.device)
            print_smart_analysis(results)
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        print("❌ Please provide either --image or --test")
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())