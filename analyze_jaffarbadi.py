#!/usr/bin/env python3
"""
Analyze Jaffarbadi Classification Issues

This script analyzes why Jaffarbadi has low accuracy by:
1. Testing all Jaffarbadi test images
2. Examining confidence scores and similarities
3. Identifying patterns in misclassifications
"""

import os
import pickle
import torch
from pathlib import Path
from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def main():
    print("=" * 60)
    print("JAFFARBADI CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    # Load prototypes
    prototypes_path = "models/prototypes.pkl"
    if not os.path.exists(prototypes_path):
        print(f"Error: Prototypes file not found at {prototypes_path}")
        return
        
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    
    print(f"Loaded prototypes for {len(breed_names)} breeds: {', '.join(breed_names)}")
    
    # Load model
    device = get_device()
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Get Jaffarbadi test images
    jaffarbadi_dir = Path("dataset/Jaffarbadi")
    if not jaffarbadi_dir.exists():
        print(f"Error: Jaffarbadi directory not found")
        return
        
    # Load dataset splits to get test images
    with open("models/prototypes.pkl", 'rb') as f:
        data = pickle.load(f)
        dataset_splits = data.get('dataset_splits', {})
    
    if 'Jaffarbadi' not in dataset_splits:
        print("Error: No dataset split found for Jaffarbadi")
        return
        
    test_images = dataset_splits['Jaffarbadi']['test']
    print(f"\nAnalyzing {len(test_images)} Jaffarbadi test images...")
    
    correct_predictions = 0
    total_predictions = len(test_images)
    results = []
    
    for i, image_path in enumerate(test_images):
        print(f"\nImage {i+1}/{total_predictions}: {os.path.basename(image_path)}")
        
        # Load and process image
        try:
            image = load_and_preprocess_image(image_path)
            image = image.to(device)
            
            # Extract features
            with torch.no_grad():
                features = model(image.unsqueeze(0)).squeeze(0)
                features = features / features.norm()  # L2 normalize
            
            # Compute similarities
            similarities = compute_similarities(features, prototypes)
            
            # Sort by similarity
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Check if correctly classified
            predicted_breed = sorted_similarities[0][0]
            confidence = sorted_similarities[0][1]
            is_correct = predicted_breed == 'Jaffarbadi'
            
            if is_correct:
                correct_predictions += 1
                status = "✓ CORRECT"
            else:
                status = "✗ INCORRECT"
            
            print(f"  {status}: Predicted {predicted_breed} (confidence: {confidence:.3f})")
            print(f"  Similarities: {', '.join([f'{breed}({sim:.3f})' for breed, sim in sorted_similarities])}")
            
            # Store detailed results
            results.append({
                'image': os.path.basename(image_path),
                'predicted': predicted_breed,
                'confidence': confidence,
                'similarities': similarities,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"  Error processing image: {e}")
    
    # Summary analysis
    accuracy = correct_predictions / total_predictions
    print(f"\n" + "=" * 60)
    print(f"JAFFARBADI ANALYSIS RESULTS")
    print(f"=" * 60)
    print(f"Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Analyze incorrect predictions
    incorrect_results = [r for r in results if not r['correct']]
    if incorrect_results:
        print(f"\nMisclassified as:")
        for result in incorrect_results:
            print(f"  {result['image']} → {result['predicted']} (conf: {result['confidence']:.3f})")
    
    # Analyze confidence scores
    jaffarbadi_confidences = [r['similarities']['Jaffarbadi'] for r in results]
    avg_jaffarbadi_conf = sum(jaffarbadi_confidences) / len(jaffarbadi_confidences)
    
    print(f"\nJaffarbadi similarity analysis:")
    print(f"  Average Jaffarbadi similarity: {avg_jaffarbadi_conf:.3f}")
    print(f"  Min Jaffarbadi similarity: {min(jaffarbadi_confidences):.3f}")
    print(f"  Max Jaffarbadi similarity: {max(jaffarbadi_confidences):.3f}")
    
    # Check confusion with other breeds
    other_breed_avg = {}
    for breed in breed_names:
        if breed != 'Jaffarbadi':
            breed_sims = [r['similarities'][breed] for r in results]
            other_breed_avg[breed] = sum(breed_sims) / len(breed_sims)
    
    print(f"\nConfusion analysis:")
    for breed, avg_sim in other_breed_avg.items():
        diff = avg_jaffarbadi_conf - avg_sim
        print(f"  Jaffarbadi vs {breed}: {diff:.3f} difference (avg: Jaffarbadi {avg_jaffarbadi_conf:.3f}, {breed} {avg_sim:.3f})")
    
    print(f"\nRecommendations:")
    if avg_jaffarbadi_conf < 0.9:
        print("  - Jaffarbadi prototype may need improvement")
        print("  - Consider more diverse training images")
    
    min_diff = min(other_breed_avg.values()) if other_breed_avg else 0
    if avg_jaffarbadi_conf - min_diff < 0.05:
        print("  - Low separation between breeds")
        print("  - Consider different augmentation strategies")
        print("  - May need more training data")

if __name__ == "__main__":
    main()