#!/usr/bin/env python3
"""
Enhanced Inference Script for Improved Jaffarbadi Recognition
"""

import argparse
import pickle
import os
import torch

from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def main():
    parser = argparse.ArgumentParser(description="Enhanced cattle breed inference")
    parser.add_argument("image_path", help="Path to image for inference")
    parser.add_argument("--prototypes-path", 
                       default="models/prototypes_enhanced.pkl",
                       help="Path to enhanced prototypes file")
    parser.add_argument("--confidence-threshold",
                       type=float,
                       default=0.6,
                       help="Confidence threshold for positive identification")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED CATTLE BREED RECOGNITION - INFERENCE")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Load enhanced prototypes
    if not os.path.exists(args.prototypes_path):
        print(f"Error: Enhanced prototypes file not found at {args.prototypes_path}")
        print("Please run: python prototype_enhanced.py --enhanced-augment --use-ensemble")
        return
        
    print(f"Loading enhanced prototypes from '{args.prototypes_path}'...")
    with open(args.prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    config = data.get('config', {})
    
    print(f"Loaded enhanced prototypes for {len(breed_names)} breeds using {model_name}")
    print(f"Breeds: {', '.join(breed_names)}")
    if config.get('enhanced_augment'):
        print("✓ Enhanced augmentation was used")
    if config.get('use_ensemble'):
        print("✓ Ensemble prototypes were used")
    
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
        
        # Compute similarities using enhanced prototypes
        print(f"\nComputing similarities with enhanced prototypes...")
        similarities = compute_similarities(features, prototypes)
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop predictions:")
        for breed, similarity in sorted_similarities:
            print(f"  {breed} ({similarity:.3f})")
        
        # Determine result
        top_breed, top_confidence = sorted_similarities[0]
        
        print(f"\n" + "=" * 60)
        if top_confidence >= args.confidence_threshold:
            print(f"[✓] ENHANCED POSITIVE IDENTIFICATION: {top_breed}")
            print(f"   Confidence: {top_confidence:.3f} (>= {args.confidence_threshold:.3f} threshold)")
            
            # Show breed information
            breed_info = {
                'Jaffarbadi': "Jaffarbadi buffalo - Known for high milk production and adaptability",
                'Murrah': "Murrah buffalo - Popular dairy breed with excellent milk quality", 
                'Surti': "Surti buffalo - Hardy breed from Gujarat, good for milk production"
            }
            
            if top_breed in breed_info:
                print(f"   Info: {breed_info[top_breed]}")
        else:
            print(f"[✗] UNKNOWN / NOT RECOGNIZED")
            print(f"   Highest confidence: {top_confidence:.3f} (< {args.confidence_threshold:.3f} threshold)")
            print(f"   Most similar to: {top_breed}")
        
        print("=" * 60)
        
        # Show confidence gap for Jaffarbadi specifically
        if 'Jaffarbadi' in similarities:
            jaffarbadi_conf = similarities['Jaffarbadi']
            other_confidences = [sim for breed, sim in similarities.items() if breed != 'Jaffarbadi']
            max_other = max(other_confidences) if other_confidences else 0
            confidence_gap = jaffarbadi_conf - max_other
            
            print(f"\nJaffarbadi Analysis:")
            print(f"  Jaffarbadi confidence: {jaffarbadi_conf:.3f}")
            print(f"  Confidence gap: {confidence_gap:.3f}")
            if confidence_gap > 0.02:
                print(f"  Status: Strong Jaffarbadi signal")
            elif confidence_gap > -0.02:
                print(f"  Status: Marginal Jaffarbadi signal")
            else:
                print(f"  Status: Weak Jaffarbadi signal")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

if __name__ == "__main__":
    main()