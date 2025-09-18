#!/usr/bin/env python3
"""
Evaluation Script for Cattle Breed Recognition

This script evaluates the prototype-based model on the test set,
computing accuracy and showing a confusion matrix.

Usage:
    python eval.py [--prototypes-path path]
"""

import argparse
import pickle
import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    FeatureExtractor,
    get_device,
    extract_features,
    compute_similarities,
    get_top_predictions
)

def load_prototypes(prototypes_path: str):
    """Load prototypes from pickle file."""
    if not os.path.exists(prototypes_path):
        raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
    
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def evaluate_model(model, prototypes, test_data, device, batch_size=32):
    """
    Evaluate the model on test data.
    
    Args:
        model: Feature extraction model
        prototypes: Dictionary of breed prototypes
        test_data: Dictionary mapping breed names to test image paths
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    true_labels = []
    predicted_labels = []
    prediction_scores = []
    
    breed_names = sorted(prototypes.keys())
    
    print(f"Evaluating on {len(breed_names)} breeds...")
    
    for breed in breed_names:
        if breed not in test_data or not test_data[breed]:
            print(f"Warning: No test images for breed {breed}")
            continue
            
        test_images = test_data[breed]
        print(f"\nEvaluating {breed}: {len(test_images)} test images")
        
        try:
            # Extract features for test images
            features = extract_features(
                model=model,
                image_paths=test_images,
                device=device, 
                batch_size=batch_size,
                augment=False
            )
            
            # Make predictions for each test image
            for i, image_path in enumerate(test_images):
                try:
                    # Get embedding for this image
                    query_embedding = features[i]
                    
                    # Compute similarities
                    similarities = compute_similarities(query_embedding, prototypes)
                    
                    # Get top prediction
                    top_predictions = get_top_predictions(similarities, top_k=1)
                    predicted_breed = top_predictions[0][0]
                    prediction_score = top_predictions[0][1]
                    
                    # Store results
                    true_labels.append(breed)
                    predicted_labels.append(predicted_breed)
                    prediction_scores.append(prediction_score)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing breed {breed}: {e}")
            continue
    
    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'prediction_scores': prediction_scores,
        'breed_names': breed_names
    }

def plot_confusion_matrix(true_labels, predicted_labels, breed_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=breed_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=breed_names, yticklabels=breed_names)
    plt.title('Confusion Matrix - Cattle Breed Recognition')
    plt.xlabel('Predicted Breed')
    plt.ylabel('True Breed')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate cattle breed recognition model")
    parser.add_argument("--prototypes-path",
                       default="models/prototypes.pkl",
                       help="Path to prototypes file")
    parser.add_argument("--batch-size",
                       type=int,
                       default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--save-results",
                       action="store_true",
                       help="Save evaluation results and plots")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CATTLE BREED RECOGNITION - EVALUATION")
    print("=" * 60)
    
    # Load prototypes and dataset splits
    print(f"Loading prototypes from '{args.prototypes_path}'...")
    try:
        prototype_data = load_prototypes(args.prototypes_path)
        prototypes = prototype_data['prototypes']
        model_name = prototype_data['model_name']
        dataset_splits = prototype_data['dataset_splits']
        breeds = prototype_data['breeds']
        
        print(f"Loaded prototypes for {len(breeds)} breeds using {model_name}")
        
    except Exception as e:
        print(f"Error loading prototypes: {e}")
        return
    
    # Check if we have test data
    test_data = {}
    total_test_images = 0
    
    for breed in breeds:
        if breed in dataset_splits and 'test' in dataset_splits[breed]:
            test_images = dataset_splits[breed]['test']
            if test_images:
                test_data[breed] = test_images
                total_test_images += len(test_images)
    
    if not test_data:
        print("Error: No test data found. Make sure prototypes were built with test splits.")
        return
    
    if len(test_data) < 2:
        print("Warning: Evaluation requires at least 2 breeds with test data.")
        print("Current test data:")
        for breed, images in test_data.items():
            print(f"  {breed}: {len(images)} images")
        print("\nTo enable evaluation, add more breeds to your dataset and rebuild prototypes.")
        return
    
    print(f"\nTest data summary:")
    print(f"Total breeds: {len(test_data)}")
    print(f"Total test images: {total_test_images}")
    for breed, images in test_data.items():
        print(f"  {breed}: {len(images)} images")
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"\nLoading {model_name} model...")
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    
    # Evaluate model
    print(f"\nRunning evaluation...")
    results = evaluate_model(
        model=model,
        prototypes=prototypes,
        test_data=test_data,
        device=device,
        batch_size=args.batch_size
    )
    
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']
    prediction_scores = results['prediction_scores']
    breed_names = results['breed_names']
    
    if not true_labels:
        print("Error: No predictions generated. Check your test data and model.")
        return
    
    # Calculate metrics
    print(f"\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Overall accuracy
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct_predictions / len(true_labels)
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{len(true_labels)})")
    
    # Average confidence score
    avg_confidence = np.mean(prediction_scores)
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Per-breed accuracy
    print(f"\nPer-breed Accuracy:")
    breed_correct = defaultdict(int)
    breed_total = defaultdict(int)
    
    for true, pred in zip(true_labels, predicted_labels):
        breed_total[true] += 1
        if true == pred:
            breed_correct[true] += 1
    
    for breed in sorted(breed_total.keys()):
        breed_acc = breed_correct[breed] / breed_total[breed]
        print(f"  {breed}: {breed_acc:.3f} ({breed_correct[breed]}/{breed_total[breed]})")
    
    # Classification report
    print(f"\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    # Confusion matrix
    try:
        print(f"\nGenerating confusion matrix...")
        save_path = "models/confusion_matrix.png" if args.save_results else None
        plot_confusion_matrix(true_labels, predicted_labels, sorted(set(true_labels)), save_path)
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        # Print text-based confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix (text):")
        print(cm)
    
    # Save detailed results
    if args.save_results:
        results_file = "models/evaluation_results.txt"
        try:
            with open(results_file, 'w') as f:
                f.write("CATTLE BREED RECOGNITION - EVALUATION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Total breeds: {len(test_data)}\n")
                f.write(f"Total test images: {len(true_labels)}\n")
                f.write(f"Overall accuracy: {accuracy:.3f}\n")
                f.write(f"Average confidence: {avg_confidence:.3f}\n\n")
                
                f.write("Per-breed results:\n")
                for breed in sorted(breed_total.keys()):
                    breed_acc = breed_correct[breed] / breed_total[breed]
                    f.write(f"  {breed}: {breed_acc:.3f} ({breed_correct[breed]}/{breed_total[breed]})\n")
                
                f.write(f"\nClassification Report:\n")
                f.write(classification_report(true_labels, predicted_labels, zero_division=0))
            
            print(f"Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    print(f"\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()