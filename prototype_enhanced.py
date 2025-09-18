#!/usr/bin/env python3
"""
Enhanced Prototype Builder with Class-Specific Optimization

This script improves Jaffarbadi recognition by:
1. Using more aggressive augmentation for Jaffarbadi
2. Adjusting training ratios to give Jaffarbadi more training data
3. Multiple prototype computation to improve representation
4. Feature ensemble for better Jaffarbadi prototypes
"""

import argparse
import pickle
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from utils import (
    get_dataset_splits, 
    FeatureExtractor,
    get_device,
    extract_features,
    compute_prototypes
)

def enhanced_augmentation_for_breed(breed_name, enable_enhanced=True):
    """Return more aggressive augmentation for specific breeds"""
    from torchvision import transforms
    
    if breed_name == "Jaffarbadi" and enable_enhanced:
        # More aggressive augmentation for Jaffarbadi
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive crop
            transforms.RandomHorizontalFlip(p=0.7),  # Higher flip probability  
            transforms.RandomRotation(degrees=20),    # More rotation
            transforms.ColorJitter(
                brightness=0.4,    # More brightness variation
                contrast=0.4,      # More contrast variation
                saturation=0.4,    # More saturation variation
                hue=0.2           # More hue variation
            ),
            transforms.RandomGrayscale(p=0.2),        # Sometimes convert to grayscale
            transforms.RandomAffine(
                degrees=15,
                translate=(0.15, 0.15),  # More translation
                scale=(0.8, 1.2),        # More scaling
                shear=10                 # Add shear transformation
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard augmentation for other breeds
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def compute_enhanced_prototypes(features_dict, breeds, use_ensemble=True):
    """Compute enhanced prototypes with optional ensemble method"""
    prototypes = {}
    
    for breed in breeds:
        if breed not in features_dict:
            continue
            
        features = features_dict[breed]
        
        if use_ensemble and breed == "Jaffarbadi":
            # For Jaffarbadi, use ensemble of multiple prototype computation methods
            
            # Method 1: Standard mean
            prototype_mean = features.mean(dim=0)
            prototype_mean = F.normalize(prototype_mean, p=2, dim=0)
            
            # Method 2: Median-based (more robust to outliers)
            prototype_median = features.median(dim=0)[0]
            prototype_median = F.normalize(prototype_median, p=2, dim=0)
            
            # Method 3: Trimmed mean (remove outliers)
            sorted_features, _ = torch.sort(features, dim=0)
            trim_size = max(1, len(features) // 5)  # Remove 20% outliers
            trimmed_features = sorted_features[trim_size:-trim_size] if trim_size > 0 else sorted_features
            prototype_trimmed = trimmed_features.mean(dim=0)
            prototype_trimmed = F.normalize(prototype_trimmed, p=2, dim=0)
            
            # Ensemble: weighted combination
            prototype_ensemble = 0.5 * prototype_mean + 0.3 * prototype_median + 0.2 * prototype_trimmed
            prototype_ensemble = F.normalize(prototype_ensemble, p=2, dim=0)
            
            prototypes[breed] = prototype_ensemble
            print(f"  Enhanced prototype for {breed}: ensemble of 3 methods")
        else:
            # Standard prototype computation for other breeds
            prototype = features.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes[breed] = prototype
            print(f"  Standard prototype for {breed}: mean aggregation")
    
    return prototypes

def main():
    parser = argparse.ArgumentParser(description="Enhanced breed prototype builder")
    parser.add_argument("--model", 
                       choices=["resnet50", "efficientnet_b0"],
                       default="resnet50",
                       help="Model architecture")
    parser.add_argument("--dataset-path",
                       default="dataset",
                       help="Path to dataset")
    parser.add_argument("--output-path", 
                       default="models/prototypes_enhanced.pkl",
                       help="Path to save enhanced prototypes")
    parser.add_argument("--batch-size",
                       type=int,
                       default=32,
                       help="Batch size")
    parser.add_argument("--enhanced-augment",
                       action="store_true",
                       help="Use enhanced augmentation for Jaffarbadi")
    parser.add_argument("--use-ensemble",
                       action="store_true", 
                       help="Use ensemble prototype computation for Jaffarbadi")
    parser.add_argument("--train-ratio",
                       type=float,
                       default=0.8,  # Increase training ratio for more data
                       help="Ratio for training set")
    parser.add_argument("--val-ratio", 
                       type=float,
                       default=0.1,   # Decrease validation for more training
                       help="Ratio for validation set")
    parser.add_argument("--test-ratio",
                       type=float,
                       default=0.1,   # Decrease test for more training
                       help="Ratio for test set")
    parser.add_argument("--random-seed",
                       type=int,
                       default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED CATTLE BREED PROTOTYPE BUILDER")
    print("=" * 60)
    print(f"Enhanced augmentation: {args.enhanced_augment}")
    print(f"Ensemble prototypes: {args.use_ensemble}")
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
    
    # Get dataset splits with new ratios
    print(f"\nAnalyzing dataset at '{args.dataset_path}'...")
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
    
    # Extract features with breed-specific augmentation
    print(f"\nExtracting features with breed-specific optimization...")
    
    all_features = {}
    all_train_val_features = {}
    
    for breed in breeds:
        print(f"\nProcessing breed: {breed}")
        
        # Get training + validation images for prototype building
        train_images = dataset_splits[breed]['train']
        val_images = dataset_splits[breed]['val']
        train_val_images = train_images + val_images
        
        print(f"  Using {len(train_val_images)} images for prototype ({len(train_images)} train + {len(val_images)} val)")
        
        # Use enhanced augmentation for Jaffarbadi if requested
        if args.enhanced_augment and breed == "Jaffarbadi":
            print(f"  Using ENHANCED augmentation for {breed}")
            # Extract features multiple times with different augmentations for Jaffarbadi
            all_breed_features = []
            for round_num in range(3):  # 3 rounds of augmented feature extraction
                print(f"    Augmentation round {round_num + 1}/3")
                features = extract_features(
                    model,
                    train_val_images, 
                    device, 
                    batch_size=args.batch_size,
                    augment=True  # Use built-in augmentation for now
                )
                all_breed_features.append(features)
            
            # Combine all augmented features
            combined_features = torch.cat(all_breed_features, dim=0)
            all_train_val_features[breed] = combined_features
            print(f"  Combined features: {combined_features.shape}")
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
    
    # Compute enhanced prototypes
    print(f"\nComputing enhanced prototypes...")
    prototypes = compute_enhanced_prototypes(
        all_train_val_features, 
        breeds, 
        use_ensemble=args.use_ensemble
    )
    
    # Save enhanced prototypes
    data = {
        'prototypes': prototypes,
        'model_name': args.model,
        'feature_dim': model.feature_dim,
        'breeds': breeds,
        'dataset_splits': dataset_splits,
        'config': {
            'enhanced_augment': args.enhanced_augment,
            'use_ensemble': args.use_ensemble,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'batch_size': args.batch_size,
            'random_seed': args.random_seed
        }
    }
    
    print(f"\nSaving enhanced prototypes to '{args.output_path}'...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Calculate file size
    file_size = os.path.getsize(args.output_path) / 1024
    print(f"Enhanced prototypes saved successfully!")
    
    print("=" * 60)
    print("ENHANCED PROTOTYPE BUILDING COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Enhanced features: {args.enhanced_augment}")
    print(f"Ensemble prototypes: {args.use_ensemble}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Total breeds: {len(breeds)}")
    print(f"Output file: {args.output_path}")
    print(f"File size: {file_size:.1f} KB")
    
    print(f"\nBreed summary:")
    for breed in breeds:
        splits = dataset_splits[breed]
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        train_val = len(splits['train']) + len(splits['val'])
        test = len(splits['test'])
        print(f"  {breed}: {total} images ({train_val} train+val, {test} test)")
    
    print(f"\nNext steps:")
    print(f"  1. Test enhanced inference: python infer_enhanced.py <image_path>")
    print(f"  2. Run enhanced evaluation: python eval_enhanced.py")
    print(f"  3. Compare with original model performance")

if __name__ == "__main__":
    main()