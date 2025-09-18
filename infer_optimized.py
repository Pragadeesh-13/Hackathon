#!/usr/bin/env python3
"""
Optimized Inference Script for 5-Breed Recognition System
"""

import argparse
import pickle
import os
import torch

from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def get_breed_info(breed_name):
    """Get detailed information about each breed"""
    breed_details = {
        'Bhadawari': {
            'type': 'Buffalo',
            'origin': 'Uttar Pradesh, India',
            'characteristics': 'Medium-sized, hardy breed with good milk production',
            'milk_yield': '5-8 liters/day',
            'special_features': 'Adapted to waterlogged areas, good for rural farming'
        },
        'Jaffarbadi': {
            'type': 'Buffalo', 
            'origin': 'Gujarat, India',
            'characteristics': 'Large-sized, high milk producing breed',
            'milk_yield': '10-15 liters/day',
            'special_features': 'Excellent for commercial dairy farming'
        },
        'Mehsana': {
            'type': 'Buffalo',
            'origin': 'Gujarat, India', 
            'characteristics': 'Medium to large size, good milk producer',
            'milk_yield': '8-12 liters/day',
            'special_features': 'Heat tolerant, suitable for semi-arid regions'
        },
        'Murrah': {
            'type': 'Buffalo',
            'origin': 'Haryana, India',
            'characteristics': 'World-famous dairy breed, excellent milk producer',
            'milk_yield': '12-18 liters/day',
            'special_features': 'Most popular buffalo breed globally'
        },
        'Surti': {
            'type': 'Buffalo',
            'origin': 'Gujarat, India',
            'characteristics': 'Medium-sized, good milk quality',
            'milk_yield': '6-10 liters/day', 
            'special_features': 'High fat content in milk, drought resistant'
        }
    }
    return breed_details.get(breed_name, {})

def main():
    parser = argparse.ArgumentParser(description="Optimized 5-breed cattle/buffalo inference")
    parser.add_argument("image_path", help="Path to image for inference")
    parser.add_argument("--prototypes-path", 
                       default="models/prototypes_5breed_optimized.pkl",
                       help="Path to optimized prototypes file")
    parser.add_argument("--confidence-threshold",
                       type=float,
                       default=0.65,
                       help="Confidence threshold for positive identification")
    parser.add_argument("--show-details",
                       action="store_true",
                       help="Show detailed breed information")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIMIZED 5-BREED RECOGNITION - INFERENCE")
    print("=" * 70)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Load optimized prototypes
    if not os.path.exists(args.prototypes_path):
        print(f"Error: Optimized prototypes file not found at {args.prototypes_path}")
        print("Please run: python prototype_optimized.py --advanced-augment --multi-ensemble")
        return
        
    print(f"Loading optimized prototypes from '{args.prototypes_path}'...")
    with open(args.prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    config = data.get('config', {})
    class_weights = data.get('class_weights', {})
    
    print(f"Loaded optimized prototypes for {len(breed_names)} breeds using {model_name}")
    print(f"Breeds: {', '.join(breed_names)}")
    
    optimization_features = []
    if config.get('advanced_augment'):
        optimization_features.append("Advanced Augmentation")
    if config.get('multi_ensemble'):
        optimization_features.append("Multi-Ensemble Prototypes")
    if config.get('optimization_level') == 'maximum_accuracy':
        optimization_features.append("Maximum Accuracy Tuning")
    
    if optimization_features:
        print(f"Optimizations: {', '.join(optimization_features)}")
    
    # Load model
    device = get_device()
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load and process image
    print(f"\nProcessing image: {args.image_path}")
    try:
        image = load_and_preprocess_image(args.image_path)
        image = image.to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(image.unsqueeze(0)).squeeze(0)
            features = features / features.norm()  # L2 normalize
        
        print(f"Extracted features: {features.shape}")
        
        # Compute similarities using optimized prototypes
        print(f"\nComputing similarities with optimized prototypes...")
        similarities = compute_similarities(features, prototypes)
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop predictions:")
        for i, (breed, similarity) in enumerate(sorted_similarities):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            weight_info = f" (training weight: {class_weights.get(breed, 1.0):.2f})" if class_weights else ""
            print(f"  {rank_symbol} {breed}: {similarity:.3f}{weight_info}")
        
        # Determine result
        top_breed, top_confidence = sorted_similarities[0]
        second_breed, second_confidence = sorted_similarities[1] if len(sorted_similarities) > 1 else ("", 0)
        
        # Calculate confidence margin
        confidence_margin = top_confidence - second_confidence
        
        print(f"\n" + "=" * 70)
        if top_confidence >= args.confidence_threshold:
            print(f"✅ OPTIMIZED POSITIVE IDENTIFICATION: {top_breed}")
            print(f"   Confidence: {top_confidence:.3f} (>= {args.confidence_threshold:.3f} threshold)")
            print(f"   Confidence margin: {confidence_margin:.3f} (vs {second_breed})")
            
            # Confidence quality assessment
            if confidence_margin >= 0.05:
                quality = "🔥 EXCELLENT"
            elif confidence_margin >= 0.03:
                quality = "✨ VERY GOOD"
            elif confidence_margin >= 0.01:
                quality = "👍 GOOD"
            else:
                quality = "⚠️ MARGINAL"
            
            print(f"   Prediction quality: {quality}")
            
            # Show breed information
            if args.show_details:
                breed_info = get_breed_info(top_breed)
                if breed_info:
                    print(f"\n📋 Breed Information:")
                    print(f"   Type: {breed_info.get('type', 'Unknown')}")
                    print(f"   Origin: {breed_info.get('origin', 'Unknown')}")
                    print(f"   Characteristics: {breed_info.get('characteristics', 'N/A')}")
                    print(f"   Milk yield: {breed_info.get('milk_yield', 'N/A')}")
                    print(f"   Special features: {breed_info.get('special_features', 'N/A')}")
        else:
            print(f"❌ UNKNOWN / NOT RECOGNIZED")
            print(f"   Highest confidence: {top_confidence:.3f} (< {args.confidence_threshold:.3f} threshold)")
            print(f"   Most similar to: {top_breed}")
            print(f"   Confidence margin: {confidence_margin:.3f}")
            
            # Suggestion for threshold adjustment
            if top_confidence >= 0.5:
                print(f"   💡 Consider lowering threshold to ~{top_confidence-0.05:.2f} for this image type")
        
        print("=" * 70)
        
        # Advanced analysis
        print(f"\n📊 Advanced Analysis:")
        breed_confidences = [sim for _, sim in sorted_similarities]
        avg_confidence = sum(breed_confidences) / len(breed_confidences)
        confidence_spread = max(breed_confidences) - min(breed_confidences)
        
        print(f"   Average similarity: {avg_confidence:.3f}")
        print(f"   Confidence spread: {confidence_spread:.3f}")
        print(f"   Distinctiveness: {'HIGH' if confidence_spread > 0.15 else 'MEDIUM' if confidence_spread > 0.08 else 'LOW'}")
        
        # Show all similarities for analysis
        print(f"\n🔍 Detailed Similarities:")
        for breed, similarity in sorted_similarities:
            bar_length = int(similarity * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {breed:10} │{bar}│ {similarity:.3f}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

if __name__ == "__main__":
    main()