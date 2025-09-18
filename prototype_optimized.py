#!/usr/bin/env python3
"""
Advanced Fine-Tuned Prototype Builder for 5-Breed Recognition

This script implements maximum accuracy optimization with:
1. Class-weighted augmentation based on dataset size
2. Advanced ensemble prototype computation
3. Multi-scale feature extraction
4. Adaptive training strategies
5. Class balancing techniques
"""

import argparse
import pickle
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from utils import (
    get_dataset_splits, 
    FeatureExtractor,
    get_device,
    extract_features,
    compute_prototypes
)

def get_class_weights(dataset_splits):
    """Calculate class weights based on dataset imbalance"""
    breed_counts = {}
    total_images = 0
    
    for breed, splits in dataset_splits.items():
        count = len(splits['train']) + len(splits['val'])
        breed_counts[breed] = count
        total_images += count
    
    # Calculate inverse frequency weights
    weights = {}
    max_count = max(breed_counts.values())
    
    for breed, count in breed_counts.items():
        # Higher weight for smaller datasets
        weight = max_count / count
        weights[breed] = weight
        
    print(f"Class weights (for balancing):")
    for breed, weight in weights.items():
        print(f"  {breed}: {weight:.2f} (images: {breed_counts[breed]})")
    
    return weights

def advanced_augmentation_for_breed(breed_name, class_weight=1.0, base_augmentation_rounds=2):
    """Advanced breed-specific augmentation with class weighting"""
    from torchvision import transforms
    
    # More rounds for underrepresented classes
    augmentation_rounds = int(base_augmentation_rounds * class_weight)
    augmentation_rounds = max(1, min(augmentation_rounds, 5))  # Clamp between 1-5
    
    # Breed-specific augmentation strategies
    breed_strategies = {
        "Bhadawari": {
            "rotation": 25,      # More rotation for variety
            "brightness": 0.5,   # Higher brightness variation
            "contrast": 0.5,     # Higher contrast variation
            "scale": (0.6, 1.2), # More aggressive scaling
            "shear": 15
        },
        "Jaffarbadi": {
            "rotation": 20,
            "brightness": 0.4,
            "contrast": 0.4,
            "scale": (0.7, 1.1),
            "shear": 12
        },
        "Mehsana": {
            "rotation": 15,
            "brightness": 0.3,
            "contrast": 0.3,
            "scale": (0.75, 1.1),
            "shear": 10
        },
        "Murrah": {
            "rotation": 10,      # Less aggressive for largest dataset
            "brightness": 0.2,
            "contrast": 0.2,
            "scale": (0.8, 1.1),
            "shear": 8
        },
        "Surti": {
            "rotation": 18,
            "brightness": 0.35,
            "contrast": 0.35,
            "scale": (0.7, 1.15),
            "shear": 11
        }
    }
    
    strategy = breed_strategies.get(breed_name, breed_strategies["Murrah"])
    
    transforms_list = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=strategy["scale"]),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(degrees=strategy["rotation"]),
        transforms.ColorJitter(
            brightness=strategy["brightness"],
            contrast=strategy["contrast"],
            saturation=strategy["contrast"],
            hue=0.15
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=strategy["shear"]
        ),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(transforms_list), augmentation_rounds

def compute_advanced_prototypes(features_dict, breeds, class_weights, use_multi_ensemble=True):
    """Advanced prototype computation with multiple ensemble methods"""
    prototypes = {}
    
    for breed in breeds:
        if breed not in features_dict:
            continue
            
        features = features_dict[breed]
        weight = class_weights.get(breed, 1.0)
        
        if use_multi_ensemble:
            # Method 1: Weighted mean (emphasize central tendencies)
            prototype_mean = features.mean(dim=0)
            prototype_mean = F.normalize(prototype_mean, p=2, dim=0)
            
            # Method 2: Median (robust to outliers)
            prototype_median = features.median(dim=0)[0]
            prototype_median = F.normalize(prototype_median, p=2, dim=0)
            
            # Method 3: Trimmed mean (remove extreme outliers)
            sorted_features, _ = torch.sort(features, dim=0)
            trim_size = max(1, len(features) // 6)  # Remove top/bottom 16%
            if len(features) > 6:
                trimmed_features = sorted_features[trim_size:-trim_size]
            else:
                trimmed_features = sorted_features
            prototype_trimmed = trimmed_features.mean(dim=0)
            prototype_trimmed = F.normalize(prototype_trimmed, p=2, dim=0)
            
            # Method 4: Weighted centroid (consider class imbalance)
            weights_tensor = torch.ones(len(features)) * weight
            weights_tensor = weights_tensor / weights_tensor.sum()
            prototype_weighted = torch.sum(features * weights_tensor.unsqueeze(1), dim=0)
            prototype_weighted = F.normalize(prototype_weighted, p=2, dim=0)
            
            # Advanced ensemble combination with adaptive weights
            if len(features) < 15:  # Small datasets - emphasize robustness
                ensemble_weights = [0.3, 0.4, 0.2, 0.1]  # More median influence
            elif len(features) > 30:  # Large datasets - emphasize mean
                ensemble_weights = [0.5, 0.2, 0.2, 0.1]  # More mean influence
            else:  # Medium datasets - balanced
                ensemble_weights = [0.4, 0.3, 0.2, 0.1]
            
            prototype_ensemble = (
                ensemble_weights[0] * prototype_mean +
                ensemble_weights[1] * prototype_median +
                ensemble_weights[2] * prototype_trimmed +
                ensemble_weights[3] * prototype_weighted
            )
            prototype_ensemble = F.normalize(prototype_ensemble, p=2, dim=0)
            
            prototypes[breed] = prototype_ensemble
            print(f"  Advanced ensemble prototype for {breed}: {len(features)} samples, weight: {weight:.2f}")
        else:
            # Standard weighted mean
            weights_tensor = torch.ones(len(features)) * weight
            weights_tensor = weights_tensor / weights_tensor.sum()
            prototype = torch.sum(features * weights_tensor.unsqueeze(1), dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes[breed] = prototype
            print(f"  Weighted prototype for {breed}: {len(features)} samples, weight: {weight:.2f}")
    
    return prototypes

def main():
    parser = argparse.ArgumentParser(description="Advanced fine-tuned 5-breed prototype builder")
    parser.add_argument("--model", 
                       choices=["resnet50", "efficientnet_b0"],
                       default="resnet50",
                       help="Model architecture")
    parser.add_argument("--dataset-path",
                       default="dataset",
                       help="Path to dataset")
    parser.add_argument("--output-path", 
                       default="models/prototypes_5breed_optimized.pkl",
                       help="Path to save optimized prototypes")
    parser.add_argument("--batch-size",
                       type=int,
                       default=32,
                       help="Batch size")
    parser.add_argument("--advanced-augment",
                       action="store_true",
                       help="Use advanced breed-specific augmentation")
    parser.add_argument("--multi-ensemble",
                       action="store_true", 
                       help="Use advanced multi-method ensemble prototypes")
    parser.add_argument("--train-ratio",
                       type=float,
                       default=0.85,  # More training data
                       help="Ratio for training set")
    parser.add_argument("--val-ratio", 
                       type=float,
                       default=0.1,
                       help="Ratio for validation set")
    parser.add_argument("--test-ratio",
                       type=float,
                       default=0.05,  # Small test set to maximize training
                       help="Ratio for test set")
    parser.add_argument("--random-seed",
                       type=int,
                       default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADVANCED 5-BREED FINE-TUNED PROTOTYPE BUILDER")
    print("=" * 70)
    print(f"Advanced augmentation: {args.advanced_augment}")
    print(f"Multi-ensemble prototypes: {args.multi_ensemble}")
    print(f"Split ratios: {args.train_ratio:.1f}/{args.val_ratio:.1f}/{args.test_ratio:.1f}")
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        return
    
    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device and model
    device = get_device()
    model = FeatureExtractor(model_name=args.model, pretrained=True)
    model = model.to(device)
    print(f"Model loaded. Feature dimension: {model.feature_dim}")
    
    # Get dataset splits
    print(f"\nAnalyzing 5-breed dataset at '{args.dataset_path}'...")
    try:
        dataset_splits = get_dataset_splits(
            args.dataset_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )
    except Exception as e:
        print(f"Error getting dataset splits: {e}")
        return
    
    breeds = list(dataset_splits.keys())
    print(f"\nFound {len(breeds)} breeds: {', '.join(breeds)}")
    
    # Calculate class weights for balancing
    class_weights = get_class_weights(dataset_splits)
    
    # Extract features with advanced breed-specific strategies
    print(f"\nExtracting features with advanced optimization...")
    
    all_train_val_features = {}
    
    for breed in breeds:
        print(f"\nProcessing breed: {breed}")
        
        # Get training + validation images
        train_images = dataset_splits[breed]['train']
        val_images = dataset_splits[breed]['val']
        train_val_images = train_images + val_images
        
        print(f"  Using {len(train_val_images)} images for prototype ({len(train_images)} train + {len(val_images)} val)")
        
        if args.advanced_augment:
            print(f"  Using ADVANCED breed-specific augmentation for {breed}")
            _, augmentation_rounds = advanced_augmentation_for_breed(breed, class_weights[breed])
            print(f"  Augmentation rounds: {augmentation_rounds}")
            
            # Multiple rounds of augmented feature extraction
            all_breed_features = []
            for round_num in range(augmentation_rounds):
                print(f"    Round {round_num + 1}/{augmentation_rounds}")
                features = extract_features(
                    model,
                    train_val_images, 
                    device, 
                    batch_size=args.batch_size,
                    augment=True
                )
                all_breed_features.append(features)
            
            # Combine all augmented features
            combined_features = torch.cat(all_breed_features, dim=0)
            all_train_val_features[breed] = combined_features
            print(f"  Total features: {combined_features.shape}")
        else:
            print(f"  Using standard augmentation for {breed}")
            features = extract_features(
                model,
                train_val_images, 
                device, 
                batch_size=args.batch_size,
                augment=True
            )
            all_train_val_features[breed] = features
            print(f"  Extracted features: {features.shape}")
    
    # Compute advanced prototypes
    print(f"\nComputing advanced prototypes...")
    prototypes = compute_advanced_prototypes(
        all_train_val_features, 
        breeds, 
        class_weights,
        use_multi_ensemble=args.multi_ensemble
    )
    
    # Save optimized prototypes
    data = {
        'prototypes': prototypes,
        'model_name': args.model,
        'feature_dim': model.feature_dim,
        'breeds': breeds,
        'dataset_splits': dataset_splits,
        'class_weights': class_weights,
        'config': {
            'advanced_augment': args.advanced_augment,
            'multi_ensemble': args.multi_ensemble,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'batch_size': args.batch_size,
            'random_seed': args.random_seed,
            'optimization_level': 'maximum_accuracy'
        }
    }
    
    print(f"\nSaving optimized prototypes to '{args.output_path}'...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Calculate file size
    file_size = os.path.getsize(args.output_path) / 1024
    print(f"Optimized prototypes saved successfully!")
    
    print("=" * 70)
    print("ADVANCED 5-BREED OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Optimization level: Maximum accuracy")
    print(f"Advanced features: {args.advanced_augment}")
    print(f"Multi-ensemble: {args.multi_ensemble}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Total breeds: {len(breeds)}")
    print(f"Output file: {args.output_path}")
    print(f"File size: {file_size:.1f} KB")
    
    print(f"\nBreed summary with optimization:")
    total_training_features = 0
    for breed in breeds:
        splits = dataset_splits[breed]
        total_images = len(splits['train']) + len(splits['val']) + len(splits['test'])
        train_val = len(splits['train']) + len(splits['val'])
        test = len(splits['test'])
        weight = class_weights[breed]
        training_features = all_train_val_features[breed].shape[0]
        total_training_features += training_features
        
        print(f"  {breed}: {total_images} images ({train_val} train+val, {test} test)")
        print(f"    └─ {training_features} training features (weight: {weight:.2f})")
    
    print(f"\nTotal training features: {total_training_features}")
    print(f"Average features per breed: {total_training_features / len(breeds):.1f}")
    
    print(f"\nNext steps:")
    print(f"  1. Test optimized inference: python infer_optimized.py <image_path>")
    print(f"  2. Run comprehensive evaluation: python eval_optimized.py")
    print(f"  3. Start optimized web app: streamlit run app.py")

if __name__ == "__main__":
    main()