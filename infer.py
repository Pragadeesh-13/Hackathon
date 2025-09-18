#!/usr/bin/env python3
"""
Inference Script for Cattle Breed Recognition

This script loads saved prototypes and performs inference on a single image,
showing top-3 breed predictions with similarity scores.

Usage:
    python infer.py <image_path> [--prototypes-path path] [--top-k 3]
"""

import argparse
import pickle
import os
import torch

from utils import (
    FeatureExtractor,
    get_device,
    load_and_preprocess_image,
    extract_features,
    compute_similarities,
    get_top_predictions,
    print_predictions,
    get_breed_info
)

def load_prototypes(prototypes_path: str):
    """Load prototypes from pickle file."""
    if not os.path.exists(prototypes_path):
        raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
    
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Predict cattle breed from image")
    parser.add_argument("image_path", 
                       help="Path to image file")
    parser.add_argument("--prototypes-path",
                       default="models/prototypes.pkl", 
                       help="Path to prototypes file")
    parser.add_argument("--top-k",
                       type=int,
                       default=3,
                       help="Number of top predictions to show")
    parser.add_argument("--show-breed-info",
                       action="store_true",
                       help="Show detailed breed information")
    parser.add_argument("--confidence-threshold",
                       type=float,
                       default=0.6,
                       help="Minimum confidence threshold for positive identification (default: 0.6)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CATTLE BREED RECOGNITION - INFERENCE")
    print("=" * 60)
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Load prototypes
    print(f"Loading prototypes from '{args.prototypes_path}'...")
    try:
        prototype_data = load_prototypes(args.prototypes_path)
        prototypes = prototype_data['prototypes']
        model_name = prototype_data['model_name']
        feature_dim = prototype_data['feature_dim']
        breeds = prototype_data['breeds']
        
        print(f"Loaded prototypes for {len(breeds)} breeds using {model_name}")
        print(f"Breeds: {', '.join(breeds)}")
        
    except Exception as e:
        print(f"Error loading prototypes: {e}")
        return
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"\nLoading {model_name} model...")
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    
    # Extract features from input image
    print(f"\nProcessing image: {args.image_path}")
    try:
        # Load and preprocess image
        image_tensor = load_and_preprocess_image(args.image_path)
        image_batch = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Extract features
        model.eval()
        with torch.no_grad():
            features = model(image_batch)
            # Normalize the features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            query_embedding = features.squeeze(0)  # Remove batch dimension
        
        print(f"Extracted features: {query_embedding.shape}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    # Compute similarities
    print(f"\nComputing similarities...")
    similarities = compute_similarities(query_embedding, prototypes)
    
    # Get top predictions
    top_predictions = get_top_predictions(similarities, top_k=args.top_k)
    
    # Check if the best prediction meets the confidence threshold
    best_breed, best_score = top_predictions[0]
    
    print(f"\nTop predictions:")
    for breed, score in top_predictions:
        print(f"  {breed} ({score:.3f})")
    
    print(f"\n" + "=" * 60)
    if best_score >= args.confidence_threshold:
        print(f"[✓] POSITIVE IDENTIFICATION: {best_breed}")
        print(f"   Confidence: {best_score:.3f} (>= {args.confidence_threshold:.3f} threshold)")
        
        # Show breed information if requested
        if args.show_breed_info:
            breed_info = get_breed_info()
            if best_breed in breed_info:
                print(f"\n" + "=" * 40)
                print(f"BREED INFORMATION: {best_breed}")
                print("=" * 40)
                info = breed_info[best_breed]
                print(f"Type: {info['type']}")
                print(f"Region: {info['region']}")
                print(f"Average Milk Yield: {info['avg_milk_yield']}")
                print(f"Features: {info['features']}")
    else:
        print(f"[X] NO CONFIDENT MATCH")
        print(f"   Best match: {best_breed} ({best_score:.3f})")
        print(f"   This is below the confidence threshold of {args.confidence_threshold:.3f}")
        print(f"   The image likely does NOT show a {best_breed} or any recognized breed.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()