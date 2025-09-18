#!/usr/bin/env python3
"""
Enhanced Multi-Feature Inference System

This script provides the ultimate breed prediction combining:
- Maximum discrimination CNN features (2,948 parameters)
- Morphological features (horn, coat, body analysis)
- Expert system insights
- Adaptive fusion for exact predictions

This system is designed to solve the 0.001 confidence difference problem
by providing multiple evidence sources for breed identification.
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
from utils.fusion import MultiFeatureFusion, BreedExpertSystem, FusedPrediction
from utils.morphology import MorphologicalFeatureExtractor

def setup_enhanced_inference():
    """Setup enhanced inference system with all components"""
    
    # Define breeds for our enhanced system (matching the prototypes)
    breeds = ['Bhadawari', 'Gir', 'Jaffarbadi', 'Kankrej', 'Mehsana', 'Murrah', 
             'Ongole', 'Sahiwal', 'Surti', 'Tharparkar']  # Exact order from prototypes
    
    # Initialize components
    fusion_system = MultiFeatureFusion(breeds)
    expert_system = BreedExpertSystem()
    
    return fusion_system, expert_system, breeds

def load_prototypes(prototypes_path: str, device: torch.device) -> Dict:
    """Load breed prototypes from pickle file"""
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract prototypes and move to device
    prototypes = data['prototypes']
    for breed in prototypes:
        prototypes[breed] = prototypes[breed].to(device)
    
    return prototypes

def enhanced_inference(image_path: str, model_path: str = 'models/prototypes_maximum_10breed.pkl',
                      device: str = 'auto') -> Dict[str, Any]:
    """
    Perform enhanced inference with multi-feature fusion
    
    Returns comprehensive analysis including:
    - Fused breed predictions
    - Feature-specific analysis
    - Expert insights
    - Confidence assessment
    """
    
    # Setup device
    if device == 'auto':
        device = get_device()
    else:
        device = torch.device(device)
    
    print(f"🔧 Enhanced Multi-Feature Inference System")
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
    
    # Load components - use actual breeds from prototypes
    fusion_system = MultiFeatureFusion(actual_breeds)
    expert_system = BreedExpertSystem()
    
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
    
    # Perform enhanced prediction
    print("🔍 Analyzing with multi-feature fusion...")
    predictions, fused_prediction = fusion_system.predict_with_fusion(
        pil_image, model, prototypes, device
    )
    
    # Get expert insights
    best_breed = predictions[0][0]
    expert_insights = expert_system.get_breed_insights(best_breed, fused_prediction)
    
    # Get feature visualization data
    morph_features = fusion_system.extract_morphological_features(pil_image)
    viz_data = fusion_system.get_feature_visualization_data(morph_features)
    
    # Compile comprehensive results
    results = {
        'image_path': image_path,
        'predictions': {
            'fused_scores': fused_prediction.breed_scores,
            'top_predictions': predictions[:5],  # Top 5 predictions
            'confidence_level': fused_prediction.confidence_level,
            'margin': fused_prediction.margin,
            'explanation': fused_prediction.explanation
        },
        'feature_analysis': {
            'morphological_features': {
                'horn_features': morph_features.horn_features,
                'coat_features': morph_features.coat_features,
                'body_features': morph_features.body_features,
                'confidence_scores': morph_features.confidence_scores
            },
            'feature_contributions': fused_prediction.feature_contributions,
            'visualization_data': viz_data
        },
        'expert_insights': expert_insights,
        'technical_details': {
            'fusion_weights': fusion_system.fusion_weights,
            'model_features': 2948,  # Maximum discrimination features
            'morphological_confidence': sum(morph_features.confidence_scores.values())
        }
    }
    
    return results

def print_detailed_analysis(results: Dict[str, Any]):
    """Print comprehensive analysis in a user-friendly format"""
    
    print("\n" + "="*70)
    print("🎯 ENHANCED BREED PREDICTION ANALYSIS")
    print("="*70)
    
    # Main prediction
    predictions = results['predictions']
    best_breed, best_score = predictions['top_predictions'][0]
    
    print(f"\n🥇 PREDICTED BREED: {best_breed.upper()}")
    print(f"📊 Confidence Score: {best_score:.6f}")
    print(f"🎚️  Confidence Level: {predictions['confidence_level'].upper()}")
    print(f"📏 Margin: {predictions['margin']:.6f}")
    print(f"💡 Explanation: {predictions['explanation']}")
    
    # Top predictions
    print(f"\n🏆 TOP 5 PREDICTIONS:")
    for i, (breed, score) in enumerate(predictions['top_predictions'], 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{status} {i}. {breed:15} - {score:.6f}")
    
    # Feature analysis
    print(f"\n🔬 MORPHOLOGICAL FEATURE ANALYSIS:")
    morph_features = results['feature_analysis']['morphological_features']
    
    # Horn analysis
    if morph_features['horn_features']:
        print(f"  🦬 Horn Features:")
        for feature, confidence in morph_features['horn_features'].items():
            print(f"     • {feature}: {confidence:.3f}")
        print(f"     Confidence: {morph_features['confidence_scores'].get('horn', 0):.3f}")
    
    # Coat analysis
    if morph_features['coat_features']:
        print(f"  🎨 Coat Features:")
        for feature, confidence in morph_features['coat_features'].items():
            print(f"     • {feature}: {confidence:.3f}")
        print(f"     Confidence: {morph_features['confidence_scores'].get('coat', 0):.3f}")
    
    # Body analysis
    if morph_features['body_features']:
        print(f"  📏 Body Features:")
        for feature, confidence in morph_features['body_features'].items():
            print(f"     • {feature}: {confidence:.3f}")
        print(f"     Confidence: {morph_features['confidence_scores'].get('body', 0):.3f}")
    
    # Feature contributions
    print(f"\n⚖️  FEATURE CONTRIBUTIONS FOR {best_breed.upper()}:")
    contributions = results['feature_analysis']['feature_contributions'][best_breed]
    print(f"  🌐 Global CNN: {contributions['cnn_contribution']:.6f} (raw: {contributions['cnn_raw_score']:.6f})")
    print(f"  🔍 Morphological: {contributions['morph_contribution']:.6f} (raw: {contributions['morph_raw_score']:.6f})")
    
    # Expert insights
    if results['expert_insights']:
        print(f"\n🧠 EXPERT INSIGHTS:")
        for key, insight in results['expert_insights'].items():
            emoji = "⚠️" if key == 'warning' else "📝"
            print(f"  {emoji} {insight}")
    
    # Technical details
    tech = results['technical_details']
    print(f"\n🔧 TECHNICAL DETAILS:")
    print(f"  • Model Features: {tech['model_features']:,}")
    print(f"  • Morphological Confidence: {tech['morphological_confidence']:.3f}")
    print(f"  • Fusion Strategy: Adaptive weighting")
    
    print("\n" + "="*70)

def analyze_multiple_images(image_dir: str, output_file: str = None):
    """Analyze multiple images and generate batch report"""
    
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    image_files = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"🔍 Analyzing {len(image_files)} images from {image_dir}")
    
    batch_results = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}...")
        
        try:
            results = enhanced_inference(str(image_file))
            batch_results.append(results)
            
            # Print summary for this image
            best_breed, best_score = results['predictions']['top_predictions'][0]
            conf_level = results['predictions']['confidence_level']
            margin = results['predictions']['margin']
            
            print(f"  ✅ {best_breed} ({best_score:.4f}) - {conf_level} confidence (margin: {margin:.4f})")
            
        except Exception as e:
            print(f"  ❌ Error processing {image_file.name}: {e}")
    
    # Save batch results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        print(f"\n📄 Batch results saved to {output_file}")
    
    # Print batch summary
    print(f"\n📊 BATCH ANALYSIS SUMMARY:")
    print(f"  • Total images: {len(image_files)}")
    print(f"  • Successfully processed: {len(batch_results)}")
    
    if batch_results:
        # Confidence level distribution
        conf_levels = [r['predictions']['confidence_level'] for r in batch_results]
        for level in ['exact', 'high', 'medium', 'low', 'uncertain']:
            count = conf_levels.count(level)
            if count > 0:
                print(f"  • {level.capitalize()} confidence: {count}")
        
        # Average margin
        margins = [r['predictions']['margin'] for r in batch_results]
        avg_margin = np.mean(margins)
        print(f"  • Average confidence margin: {avg_margin:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-Feature Breed Inference')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images for batch processing')
    parser.add_argument('--model', type=str, default='models/prototypes_maximum_10breed.pkl',
                       help='Path to model and prototypes pickle file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--output', type=str, help='Output JSON file for batch results')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed analysis (default for single image)')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image analysis
        try:
            results = enhanced_inference(args.image, args.model, args.device)
            print_detailed_analysis(results)
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results saved to {args.output}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    elif args.image_dir:
        # Batch analysis
        try:
            analyze_multiple_images(args.image_dir, args.output)
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    else:
        print("❌ Please provide either --image or --image_dir")
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())