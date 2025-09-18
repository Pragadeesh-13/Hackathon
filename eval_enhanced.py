#!/usr/bin/env python3
"""
Enhanced Evaluation Script
"""

import os
import pickle
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def main():
    print("=" * 60)
    print("ENHANCED CATTLE BREED RECOGNITION - EVALUATION")
    print("=" * 60)
    
    # Load enhanced prototypes
    prototypes_path = "models/prototypes_enhanced.pkl"
    if not os.path.exists(prototypes_path):
        print(f"Error: Enhanced prototypes file not found at {prototypes_path}")
        return
        
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    dataset_splits = data['dataset_splits']
    config = data.get('config', {})
    
    print(f"Loaded enhanced prototypes for {len(breed_names)} breeds using {model_name}")
    if config.get('enhanced_augment'):
        print("✓ Enhanced augmentation was used")
    if config.get('use_ensemble'):
        print("✓ Ensemble prototypes were used")
    
    # Get test data
    test_data = []
    test_labels = []
    
    print(f"\nTest data summary:")
    total_test_images = 0
    for breed in breed_names:
        test_images = dataset_splits[breed]['test']
        print(f"  {breed}: {len(test_images)} images")
        total_test_images += len(test_images)
        
        for image_path in test_images:
            test_data.append(image_path)
            test_labels.append(breed)
    
    print(f"Total breeds: {len(breed_names)}")
    print(f"Total test images: {total_test_images}")
    
    # Load model
    device = get_device()
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    print(f"\nRunning enhanced evaluation...")
    
    # Evaluate each image
    predictions = []
    confidences = []
    
    for i, (image_path, true_label) in enumerate(zip(test_data, test_labels)):
        print(f"Evaluating {i+1}/{len(test_data)}: {os.path.basename(image_path)}")
        
        try:
            # Load and process image
            image = load_and_preprocess_image(image_path)
            image = image.to(device)
            
            # Extract features
            with torch.no_grad():
                features = model(image.unsqueeze(0)).squeeze(0)
                features = features / features.norm()
            
            # Compute similarities
            similarities = compute_similarities(features, prototypes)
            
            # Get prediction
            predicted_breed = max(similarities.items(), key=lambda x: x[1])[0]
            confidence = similarities[predicted_breed]
            
            predictions.append(predicted_breed)
            confidences.append(confidence)
            
        except Exception as e:
            print(f"  Error: {e}")
            predictions.append("Error")
            confidences.append(0.0)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct / len(test_labels)
    avg_confidence = np.mean(confidences)
    
    print(f"\n" + "=" * 60)
    print(f"ENHANCED EVALUATION RESULTS")
    print(f"=" * 60)
    print(f"Overall Accuracy: {accuracy:.3f} ({correct}/{len(test_labels)})")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Per-breed accuracy
    print(f"\nPer-breed Accuracy:")
    breed_accuracies = {}
    for breed in breed_names:
        breed_indices = [i for i, label in enumerate(test_labels) if label == breed]
        breed_predictions = [predictions[i] for i in breed_indices]
        breed_correct = sum(1 for pred in breed_predictions if pred == breed)
        breed_accuracy = breed_correct / len(breed_predictions) if breed_predictions else 0
        breed_accuracies[breed] = breed_accuracy
        print(f"  {breed}: {breed_accuracy:.3f} ({breed_correct}/{len(breed_predictions)})")
    
    # Classification report
    print(f"\n" + "-" * 60)
    print(f"DETAILED CLASSIFICATION REPORT")
    print(f"-" * 60)
    report = classification_report(test_labels, predictions, target_names=breed_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    print(f"Generating confusion matrix...")
    cm = confusion_matrix(test_labels, predictions, labels=breed_names)
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for breed in breed_names:
        print(f"{breed:>10}", end="")
    print()
    
    for i, breed in enumerate(breed_names):
        print(f"{breed:>12}", end="")
        for j in range(len(breed_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Detailed analysis for Jaffarbadi
    print(f"\n" + "=" * 60)
    print(f"JAFFARBADI IMPROVEMENT ANALYSIS")
    print(f"=" * 60)
    
    jaffarbadi_indices = [i for i, label in enumerate(test_labels) if label == 'Jaffarbadi']
    if jaffarbadi_indices:
        print(f"Jaffarbadi test results:")
        for idx in jaffarbadi_indices:
            image_name = os.path.basename(test_data[idx])
            pred = predictions[idx]
            conf = confidences[idx]
            status = "✓ CORRECT" if pred == 'Jaffarbadi' else "✗ INCORRECT"
            print(f"  {image_name}: {status} - Predicted {pred} (conf: {conf:.3f})")
        
        jaffarbadi_accuracy = breed_accuracies.get('Jaffarbadi', 0)
        print(f"\nJaffarbadi accuracy: {jaffarbadi_accuracy:.1%}")
        
        # Compare with original results (40% accuracy)
        original_accuracy = 0.40
        improvement = jaffarbadi_accuracy - original_accuracy
        print(f"Original accuracy: {original_accuracy:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        
        if improvement > 0:
            print(f"🎉 SUCCESS! Enhanced model improved Jaffarbadi recognition!")
        else:
            print(f"⚠️  Model needs further tuning for Jaffarbadi")
    
    print(f"\n" + "=" * 60)
    print(f"ENHANCED EVALUATION COMPLETE")
    print(f"=" * 60)

if __name__ == "__main__":
    main()