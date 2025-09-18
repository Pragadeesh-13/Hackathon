#!/usr/bin/env python3
"""
Ultra Inference Script for Maximum Differentiation
"""

import argparse
import pickle
import os
import torch

from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def main():
    parser = argparse.ArgumentParser(description="Ultra-optimized 5-breed inference")
    parser.add_argument("image_path", help="Path to image for inference")
    parser.add_argument("--prototypes-path", 
                       default="models/prototypes_5breed_ultra.pkl",
                       help="Path to ultra-optimized prototypes file")
    parser.add_argument("--confidence-threshold",
                       type=float,
                       default=0.7,  # Higher threshold for ultra model
                       help="Confidence threshold for positive identification")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ULTRA-OPTIMIZED 5-BREED RECOGNITION - INFERENCE")
    print("=" * 80)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Load ultra-optimized prototypes
    if not os.path.exists(args.prototypes_path):
        print(f"Error: Ultra-optimized prototypes file not found at {args.prototypes_path}")
        print("Please run: python prototype_ultra.py")
        return
        
    print(f"Loading ultra-optimized prototypes from '{args.prototypes_path}'...")
    with open(args.prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    config = data.get('config', {})
    class_weights = data.get('class_weights', {})
    
    print(f"Loaded ultra prototypes for {len(breed_names)} breeds using {model_name}")
    print(f"Breeds: {', '.join(breed_names)}")
    
    ultra_features = []
    if config.get('contrastive_refinement'):
        ultra_features.append("🔥 Contrastive Refinement")
    if config.get('feature_importance_weighting'):
        ultra_features.append("⚖️ Feature Importance Weighting")
    if config.get('ultra_augmentation'):
        ultra_features.append("🎯 Ultra Augmentation")
    
    if ultra_features:
        print(f"Ultra optimizations: {', '.join(ultra_features)}")
    
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
        
        # Compute similarities using ultra prototypes
        print(f"\nComputing similarities with ultra-optimized prototypes...")
        similarities = compute_similarities(features, prototypes)
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Ultra predictions:")
        for i, (breed, similarity) in enumerate(sorted_similarities):
            if i == 0:
                rank_symbol = "🥇"
                bar_color = "🔥"
            elif i == 1:
                rank_symbol = "🥈"
                bar_color = "✨"
            elif i == 2:
                rank_symbol = "🥉"
                bar_color = "⭐"
            else:
                rank_symbol = f"{i+1}."
                bar_color = "📊"
            
            # Ultra confidence bar
            bar_length = int(similarity * 25)
            bar = "█" * bar_length + "░" * (25 - bar_length)
            weight_info = f" (ultra-weight: {class_weights.get(breed, 1.0):.2f})" if class_weights else ""
            
            print(f"  {rank_symbol} {breed:10} │{bar_color}{bar}│ {similarity:.4f}{weight_info}")
        
        # Determine result with ultra analysis
        top_breed, top_confidence = sorted_similarities[0]
        second_breed, second_confidence = sorted_similarities[1] if len(sorted_similarities) > 1 else ("", 0)
        
        # Ultra confidence margin analysis
        confidence_margin = top_confidence - second_confidence
        
        print(f"\n" + "=" * 80)
        if top_confidence >= args.confidence_threshold:
            print(f"🎯 ULTRA POSITIVE IDENTIFICATION: {top_breed}")
            print(f"   Ultra confidence: {top_confidence:.4f} (>= {args.confidence_threshold:.3f} threshold)")
            print(f"   Ultra margin: {confidence_margin:.4f} (vs {second_breed})")
            
            # Ultra quality assessment
            if confidence_margin >= 0.08:
                quality = "🔥🔥🔥 ULTRA EXCELLENT"
            elif confidence_margin >= 0.05:
                quality = "🔥🔥 ULTRA STRONG"
            elif confidence_margin >= 0.03:
                quality = "🔥 ULTRA GOOD"
            elif confidence_margin >= 0.01:
                quality = "✨ ULTRA ACCEPTABLE"
            else:
                quality = "⚠️ ULTRA MARGINAL"
            
            print(f"   Ultra quality: {quality}")
            
            # Show differentiation metrics
            all_confidences = [sim for _, sim in sorted_similarities]
            confidence_std = torch.std(torch.tensor(all_confidences)).item()
            print(f"   Differentiation strength: {confidence_std:.4f}")
            
        else:
            print(f"❌ ULTRA REJECTION - NOT RECOGNIZED")
            print(f"   Highest confidence: {top_confidence:.4f} (< {args.confidence_threshold:.3f} threshold)")
            print(f"   Most similar to: {top_breed}")
            print(f"   Ultra margin: {confidence_margin:.4f}")
            
            # Ultra suggestion
            if top_confidence >= 0.6:
                suggested_threshold = max(0.5, top_confidence - 0.05)
                print(f"   💡 Ultra suggestion: Lower threshold to ~{suggested_threshold:.2f}")
        
        print("=" * 80)
        
        # Ultra advanced analysis
        print(f"\n🔬 Ultra Advanced Analysis:")
        breed_confidences = [sim for _, sim in sorted_similarities]
        avg_confidence = sum(breed_confidences) / len(breed_confidences)
        confidence_range = max(breed_confidences) - min(breed_confidences)
        
        print(f"   Ultra average similarity: {avg_confidence:.4f}")
        print(f"   Ultra confidence range: {confidence_range:.4f}")
        
        # Ultra distinctiveness rating
        if confidence_range > 0.20:
            distinctiveness = "🔥🔥🔥 ULTRA HIGH"
        elif confidence_range > 0.15:
            distinctiveness = "🔥🔥 ULTRA STRONG"
        elif confidence_range > 0.10:
            distinctiveness = "🔥 ULTRA GOOD"
        elif confidence_range > 0.05:
            distinctiveness = "✨ ULTRA MODERATE"
        else:
            distinctiveness = "⚠️ ULTRA LOW"
        
        print(f"   Ultra distinctiveness: {distinctiveness}")
        
        # Show relative performance vs breeds
        print(f"\n🎭 Ultra Breed Performance Analysis:")
        for breed, similarity in sorted_similarities:
            relative_strength = (similarity - min(breed_confidences)) / max(0.001, confidence_range)
            
            if relative_strength > 0.8:
                performance = "🔥 DOMINANT"
            elif relative_strength > 0.6:
                performance = "✨ STRONG"
            elif relative_strength > 0.4:
                performance = "📊 MODERATE"
            else:
                performance = "💤 WEAK"
            
            print(f"   {breed:10}: {performance} (relative: {relative_strength:.2f})")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

if __name__ == "__main__":
    main()