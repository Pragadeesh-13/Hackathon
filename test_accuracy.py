#!/usr/bin/env python3
"""
CNN vs Fusion Accuracy Comparison
"""

import sys
import pickle
import torch
sys.path.append('.')
from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def compare_predictions(image_path):
    """Compare CNN-only vs fusion predictions"""
    
    device = get_device()
    print(f"🔍 Analyzing: {image_path}")
    print(f"📱 Device: {device}")
    print("=" * 60)
    
    # Load prototypes
    with open('models/prototypes_maximum_10breed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    prototypes = data['prototypes']
    breeds = data['breeds']
    model_name = data['model_name']
    
    # Move prototypes to device
    for breed in prototypes:
        prototypes[breed] = prototypes[breed].to(device)
    
    # Create model
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load and process image
    image_tensor = load_and_preprocess_image(image_path)
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    # Extract CNN features
    with torch.no_grad():
        features = model(image_batch)
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        query_embedding = features_normalized.squeeze(0)
    
    # Compute CNN similarities
    similarities = compute_similarities(query_embedding, prototypes)
    
    print("🧠 PURE CNN PREDICTIONS (Maximum System):")
    sorted_cnn = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for i, (breed, sim) in enumerate(sorted_cnn[:5], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"  {emoji} {breed:12} - {sim:.6f}")
    
    # Calculate CNN margin
    cnn_margin = sorted_cnn[0][1] - sorted_cnn[1][1]
    print(f"\n📏 CNN Margin: {cnn_margin:.6f}")
    print(f"🎯 CNN Top Prediction: {sorted_cnn[0][0]} ({sorted_cnn[0][1]:.6f})")
    
    return sorted_cnn, cnn_margin

def test_multiple_images():
    """Test multiple images to see accuracy patterns"""
    
    test_images = [
        "dataset/Murrah/murrah_0002.jpg",  # Should be Murrah
        "dataset/Gir/Gir_1.JPG",         # Should be Gir  
        "dataset/Kankrej/kankrej_0001.jpg",  # Should be Kankrej
        "dataset/Sahiwal/sahiwal_0001.jpg"   # Should be Sahiwal
    ]
    
    print("🧪 ACCURACY COMPARISON TEST")
    print("=" * 80)
    
    for image_path in test_images:
        try:
            expected_breed = image_path.split('/')[1]  # Extract breed from path
            print(f"\n📸 Expected: {expected_breed}")
            
            cnn_preds, margin = compare_predictions(image_path)
            cnn_top = cnn_preds[0][0]
            
            # Check if CNN prediction matches expected
            accuracy_status = "✅ CORRECT" if cnn_top.lower() == expected_breed.lower() else "❌ INCORRECT"
            print(f"🎯 CNN Result: {accuracy_status}")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error with {image_path}: {e}")
            print("-" * 60)

if __name__ == "__main__":
    test_multiple_images()