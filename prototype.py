#!/usr/bin/env python3
"""
Prototype Builder for Cattle Breed Recognition

This script builds breed prototypes from the dataset by:
1. Discovering all breeds in the dataset
2. Splitting images into train/val/test sets 
3. Extracting features using a pretrained model
4. Computing prototype vectors for each breed
5. Saving prototypes to a pickle file

Usage:
    python prototype.py [--model resnet50|efficientnet_b0] [--dataset-path path] [--output-path path]
"""

import argparse
import pickle
import os
from pathlib import Path
import torch

from utils import (
    get_dataset_splits, 
    FeatureExtractor,
    get_device,
    extract_features,
    compute_prototypes
)

def main():
    parser = argparse.ArgumentParser(description="Build breed prototypes from dataset")
    parser.add_argument("--model", 
                       choices=["resnet50", "efficientnet_b0"],
                       default="resnet50",
                       help="Model architecture to use for feature extraction")
    parser.add_argument("--dataset-path",
                       default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output-path", 
                       default="models/prototypes.pkl",
                       help="Path to save prototypes")
    parser.add_argument("--batch-size",
                       type=int,
                       default=32,
                       help="Batch size for feature extraction")
    parser.add_argument("--augment",
                       action="store_true",
                       help="Use data augmentation during feature extraction")
    parser.add_argument("--train-ratio",
                       type=float,
                       default=0.7,
                       help="Ratio for training set")
    parser.add_argument("--val-ratio", 
                       type=float,
                       default=0.15,
                       help="Ratio for validation set")
    parser.add_argument("--test-ratio",
                       type=float,
                       default=0.15, 
                       help="Ratio for test set")
    parser.add_argument("--random-seed",
                       type=int,
                       default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CATTLE BREED PROTOTYPE BUILDER")
    print("=" * 60)
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"\nLoading {args.model} model...")
    model = FeatureExtractor(model_name=args.model, pretrained=True)
    model = model.to(device)
    print(f"Model loaded. Feature dimension: {model.feature_dim}")
    
    # Get dataset splits
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
        print(f"Error analyzing dataset: {e}")
        return
    
    if not dataset_splits:
        print("No valid breeds found in dataset")
        return
    
    print(f"\nFound {len(dataset_splits)} breeds:")
    for breed in dataset_splits.keys():
        print(f"  - {breed}")
    
    # Extract features for each breed (using training + validation images for prototypes)
    print(f"\nExtracting features using {args.model}...")
    features_by_breed = {}
    
    for breed, splits in dataset_splits.items():
        print(f"\nProcessing breed: {breed}")
        
        # Combine train and validation images for prototype computation
        # This gives us more data to build robust prototypes
        prototype_images = splits['train'] + splits['val']
        
        if not prototype_images:
            print(f"  Warning: No images found for {breed}, skipping")
            continue
        
        print(f"  Extracting features from {len(prototype_images)} images...")
        try:
            features = extract_features(
                model=model,
                image_paths=prototype_images,
                device=device,
                batch_size=args.batch_size,
                augment=args.augment
            )
            features_by_breed[breed] = features
            print(f"  Extracted features: {features.shape}")
            
        except Exception as e:
            print(f"  Error extracting features for {breed}: {e}")
            continue
    
    if not features_by_breed:
        print("Error: No features extracted for any breed")
        return
    
    # Compute prototypes
    print(f"\nComputing prototypes...")
    prototypes = compute_prototypes(features_by_breed)
    
    # Prepare data to save
    save_data = {
        'prototypes': prototypes,
        'model_name': args.model,
        'feature_dim': model.feature_dim,
        'breeds': list(prototypes.keys()),
        'dataset_splits': dataset_splits,
        'config': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'random_seed': args.random_seed,
            'augment': args.augment,
            'batch_size': args.batch_size
        }
    }
    
    # Save prototypes
    print(f"\nSaving prototypes to '{args.output_path}'...")
    try:
        with open(args.output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print("Prototypes saved successfully!")
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("PROTOTYPE BUILDING COMPLETE")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Feature dimension: {model.feature_dim}")
        print(f"Total breeds: {len(prototypes)}")
        print(f"Output file: {args.output_path}")
        print(f"File size: {os.path.getsize(args.output_path) / 1024:.1f} KB")
        
        print(f"\nBreed summary:")
        for breed in prototypes.keys():
            train_count = len(dataset_splits[breed]['train'])
            val_count = len(dataset_splits[breed]['val']) 
            test_count = len(dataset_splits[breed]['test'])
            total_count = train_count + val_count + test_count
            print(f"  {breed}: {total_count} images ({train_count} train, {val_count} val, {test_count} test)")
        
        print(f"\nNext steps:")
        print(f"  1. Test inference: python infer.py <image_path>")
        print(f"  2. Run evaluation: python eval.py")
        print(f"  3. Start web app: streamlit run app.py")
        
    except Exception as e:
        print(f"Error saving prototypes: {e}")
        return

if __name__ == "__main__":
    main()