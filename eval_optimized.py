#!/usr/bin/env python3
"""
Comprehensive Evaluation for Optimized 5-Breed Recognition System
"""

import os
import pickle
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import FeatureExtractor, get_device, load_and_preprocess_image, compute_similarities

def main():
    print("=" * 70)
    print("COMPREHENSIVE 5-BREED OPTIMIZED EVALUATION")
    print("=" * 70)
    
    # Load optimized prototypes
    prototypes_path = "models/prototypes_5breed_optimized.pkl"
    if not os.path.exists(prototypes_path):
        print(f"Error: Optimized prototypes file not found at {prototypes_path}")
        return
        
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
        
    prototypes = data['prototypes']
    breed_names = list(prototypes.keys())
    model_name = data['model_name']
    dataset_splits = data['dataset_splits']
    config = data.get('config', {})
    class_weights = data.get('class_weights', {})
    
    print(f"Loaded optimized prototypes for {len(breed_names)} breeds using {model_name}")
    print(f"Breeds: {', '.join(breed_names)}")
    
    optimization_features = []
    if config.get('advanced_augment'):
        optimization_features.append("Advanced Augmentation")
    if config.get('multi_ensemble'):
        optimization_features.append("Multi-Ensemble Prototypes")
    
    print(f"Optimizations: {', '.join(optimization_features)}")
    
    # Get test data
    test_data = []
    test_labels = []
    
    print(f"\n📊 Test data summary:")
    total_test_images = 0
    for breed in breed_names:
        test_images = dataset_splits[breed]['test']
        print(f"  {breed}: {len(test_images)} test images (weight: {class_weights.get(breed, 1.0):.2f})")
        total_test_images += len(test_images)
        
        for image_path in test_images:
            test_data.append(image_path)
            test_labels.append(breed)
    
    print(f"Total test images: {total_test_images}")
    
    # Load model
    device = get_device()
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    print(f"\n🔄 Running comprehensive evaluation...")
    
    # Evaluate each image
    predictions = []
    confidences = []
    all_similarities = []
    
    for i, (image_path, true_label) in enumerate(zip(test_data, test_labels)):
        image_name = os.path.basename(image_path)
        print(f"Evaluating {i+1}/{len(test_data)}: {true_label}/{image_name}")
        
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
            all_similarities.append(similarities)
            
            # Get prediction
            predicted_breed = max(similarities.items(), key=lambda x: x[1])[0]
            confidence = similarities[predicted_breed]
            
            predictions.append(predicted_breed)
            confidences.append(confidence)
            
            # Show result
            status = "✅" if predicted_breed == true_label else "❌"
            print(f"    {status} Predicted: {predicted_breed} (conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            predictions.append("Error")
            confidences.append(0.0)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct / len(test_labels)
    avg_confidence = np.mean(confidences)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 COMPREHENSIVE EVALUATION RESULTS")
    print(f"=" * 70)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(test_labels)})")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Per-breed accuracy
    print(f"\n📈 Per-breed Performance:")
    breed_results = {}
    for breed in breed_names:
        breed_indices = [i for i, label in enumerate(test_labels) if label == breed]
        breed_predictions = [predictions[i] for i in breed_indices]
        breed_correct = sum(1 for pred in breed_predictions if pred == breed)
        breed_accuracy = breed_correct / len(breed_predictions) if breed_predictions else 0
        breed_confidence = np.mean([confidences[i] for i in breed_indices]) if breed_indices else 0
        
        breed_results[breed] = {
            'accuracy': breed_accuracy,
            'correct': breed_correct,
            'total': len(breed_predictions),
            'confidence': breed_confidence
        }
        
        status_emoji = "🟢" if breed_accuracy >= 0.8 else "🟡" if breed_accuracy >= 0.5 else "🔴"
        print(f"  {status_emoji} {breed}: {breed_accuracy:.1%} ({breed_correct}/{len(breed_predictions)}) - avg conf: {breed_confidence:.3f}")
    
    # Detailed classification report
    print(f"\n" + "-" * 60)
    print(f"📋 DETAILED CLASSIFICATION REPORT")
    print(f"-" * 60)
    report = classification_report(test_labels, predictions, target_names=breed_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    print(f"\n🔀 Confusion Matrix:")
    cm = confusion_matrix(test_labels, predictions, labels=breed_names)
    
    # Print text version
    print(f"{'':>12}", end="")
    for breed in breed_names:
        print(f"{breed[:8]:>10}", end="")
    print("\nActual ↓")
    
    for i, breed in enumerate(breed_names):
        print(f"{breed[:10]:>12}", end="")
        for j in range(len(breed_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Similarity analysis
    print(f"\n🔍 Similarity Analysis:")
    if all_similarities:
        breed_avg_similarities = {}
        for breed in breed_names:
            breed_indices = [i for i, label in enumerate(test_labels) if label == breed]
            if breed_indices:
                breed_sims = []
                for idx in breed_indices:
                    breed_sims.append(all_similarities[idx][breed])
                breed_avg_similarities[breed] = np.mean(breed_sims)
        
        print(f"  Average self-similarity per breed:")
        for breed, avg_sim in breed_avg_similarities.items():
            print(f"    {breed}: {avg_sim:.3f}")
    
    # Recommendations
    print(f"\n💡 Optimization Recommendations:")
    
    worst_breed = min(breed_results.items(), key=lambda x: x[1]['accuracy'])
    best_breed = max(breed_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"  🔴 Lowest accuracy: {worst_breed[0]} ({worst_breed[1]['accuracy']:.1%})")
    print(f"  🟢 Highest accuracy: {best_breed[0]} ({best_breed[1]['accuracy']:.1%})")
    
    if accuracy < 0.8:
        print(f"  📈 Overall accuracy can be improved:")
        print(f"     - Collect more images for {worst_breed[0]}")
        print(f"     - Increase augmentation rounds for low-performing breeds")
        print(f"     - Consider adjusting confidence thresholds")
    
    if avg_confidence < 0.85:
        print(f"  🎯 Low average confidence suggests:")
        print(f"     - Breeds may be too similar visually")
        print(f"     - Consider multi-scale feature extraction")
        print(f"     - Add more diverse training images")
    
    # Save detailed results
    results = {
        'overall_accuracy': accuracy,
        'average_confidence': avg_confidence,
        'breed_results': breed_results,
        'confusion_matrix': cm.tolist(),
        'breed_names': breed_names,
        'config': config,
        'class_weights': class_weights
    }
    
    results_path = "models/evaluation_results_5breed.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    print(f"\n" + "=" * 70)
    print(f"🏁 COMPREHENSIVE EVALUATION COMPLETE")
    print(f"=" * 70)
    
    # Final summary
    if accuracy >= 0.8:
        print(f"🎉 EXCELLENT: Model ready for production!")
    elif accuracy >= 0.6:
        print(f"✅ GOOD: Model suitable for most applications")
    else:
        print(f"⚠️ NEEDS IMPROVEMENT: Consider additional optimization")

if __name__ == "__main__":
    main()