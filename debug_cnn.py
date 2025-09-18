#!/usr/bin/env python3
"""
Debug script to check CNN feature extraction
"""

import sys
import pickle
import torch
sys.path.append('.')
from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def debug_cnn_features(image_path):
    """Debug CNN feature extraction"""
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load prototypes
    with open('models/prototypes_maximum_10breed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    prototypes = data['prototypes']
    breeds = data['breeds']
    model_name = data['model_name']
    
    print(f"Breeds: {breeds}")
    print(f"Model: {model_name}")
    
    # Move prototypes to device
    for breed in prototypes:
        prototypes[breed] = prototypes[breed].to(device)
        print(f"{breed} prototype shape: {prototypes[breed].shape}")
    
    # Create model
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load and process image
    image_tensor = load_and_preprocess_image(image_path)
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Extract features
    with torch.no_grad():
        features = model(image_batch)
        print(f"Raw features shape: {features.shape}")
        print(f"Raw features sample: {features[0, :5]}")
        
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        query_embedding = features_normalized.squeeze(0)
        print(f"Normalized features shape: {query_embedding.shape}")
        print(f"Normalized features sample: {query_embedding[:5]}")
    
    # Compute similarities
    similarities = compute_similarities(query_embedding, prototypes)
    print(f"\nSimilarities:")
    for breed, sim in similarities.items():
        print(f"  {breed}: {sim:.6f}")

if __name__ == "__main__":
    debug_cnn_features("dataset/Gir/Gir_1.JPG")