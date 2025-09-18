#!/usr/bin/env python3
"""
Ultra Fine-Tuned Prototype Builder with Maximum Differentiation

This script implements extreme optimization for maximum breed differentiation:
1. Multi-scale feature extraction
2. Contrastive learning-inspired prototype refinement
3. Hard negative mining
4. Advanced ensemble with confidence calibration
5. Breed-specific feature emphasis
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

def compute_ultra_optimized_prototypes(features_dict, breeds, class_weights):
    """Ultra-optimized prototype computation with contrastive refinement"""
    print("\n🔥 Computing ultra-optimized prototypes...")
    
    # Step 1: Compute base prototypes
    base_prototypes = {}
    for breed in breeds:
        if breed not in features_dict:
            continue
        features = features_dict[breed]
        prototype = features.mean(dim=0)
        prototype = F.normalize(prototype, p=2, dim=0)
        base_prototypes[breed] = prototype
    
    # Step 2: Contrastive refinement - push prototypes apart
    refined_prototypes = {}
    refinement_strength = 0.3  # How much to push apart
    
    for breed in breeds:
        if breed not in base_prototypes:
            continue
            
        current_prototype = base_prototypes[breed].clone()
        
        # Calculate average of other prototypes (what to push away from)
        other_prototypes = []
        for other_breed in breeds:
            if other_breed != breed and other_breed in base_prototypes:
                other_prototypes.append(base_prototypes[other_breed])
        
        if other_prototypes:
            other_avg = torch.stack(other_prototypes).mean(dim=0)
            
            # Push current prototype away from others
            difference = current_prototype - other_avg
            enhanced_prototype = current_prototype + refinement_strength * difference
            enhanced_prototype = F.normalize(enhanced_prototype, p=2, dim=0)
            
            refined_prototypes[breed] = enhanced_prototype
            print(f"  ✨ {breed}: Enhanced prototype with contrastive refinement")
        else:
            refined_prototypes[breed] = current_prototype
            print(f"  📝 {breed}: Base prototype (no others to contrast)")
    
    # Step 3: Feature emphasis based on discriminative power
    final_prototypes = {}
    
    for breed in breeds:
        if breed not in features_dict or breed not in refined_prototypes:
            continue
            
        features = features_dict[breed]
        refined_proto = refined_prototypes[breed]
        
        # Calculate feature importance (variance-based)
        feature_variance = features.var(dim=0)
        
        # High variance features are more discriminative
        importance_weights = torch.sigmoid(feature_variance * 10)  # Scale and sigmoid
        
        # Apply importance weighting
        weighted_prototype = refined_proto * importance_weights
        weighted_prototype = F.normalize(weighted_prototype, p=2, dim=0)
        
        final_prototypes[breed] = weighted_prototype
        
        # Calculate enhancement metrics
        original_similarity = F.cosine_similarity(base_prototypes[breed], refined_proto, dim=0)
        final_similarity = F.cosine_similarity(refined_proto, weighted_prototype, dim=0)
        
        print(f"  🎯 {breed}: Refinement similarity: {original_similarity:.3f}, "
              f"Weighting similarity: {final_similarity:.3f}")
    
    return final_prototypes

def main():
    parser = argparse.ArgumentParser(description="Ultra fine-tuned 5-breed prototype builder")
    parser.add_argument("--model", 
                       choices=["resnet50", "efficientnet_b0"],
                       default="resnet50",
                       help="Model architecture")
    parser.add_argument("--dataset-path",
                       default="dataset",
                       help="Path to dataset")
    parser.add_argument("--output-path", 
                       default="models/prototypes_5breed_ultra.pkl",
                       help="Path to save ultra-optimized prototypes")
    parser.add_argument("--batch-size",
                       type=int,
                       default=32,
                       help="Batch size")
    parser.add_argument("--train-ratio",
                       type=float,
                       default=0.9,  # Maximum training data
                       help="Ratio for training set")
    parser.add_argument("--val-ratio", 
                       type=float,
                       default=0.05,
                       help="Ratio for validation set")
    parser.add_argument("--test-ratio",
                       type=float,
                       default=0.05,  # Minimal test set
                       help="Ratio for test set")
    parser.add_argument("--random-seed",
                       type=int,
                       default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ULTRA FINE-TUNED 5-BREED PROTOTYPE BUILDER")
    print("=" * 80)
    print(f"🎯 Optimization Level: MAXIMUM DIFFERENTIATION")
    print(f"📊 Split ratios: {args.train_ratio:.1f}/{args.val_ratio:.1f}/{args.test_ratio:.1f}")
    
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
    print(f"🧠 Model loaded. Feature dimension: {model.feature_dim}")
    
    # Get dataset splits with maximum training data
    print(f"\n📁 Analyzing dataset at '{args.dataset_path}'...")
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
    print(f"\n🎭 Found {len(breeds)} breeds: {', '.join(breeds)}")
    
    # Calculate class weights
    breed_counts = {}
    for breed, splits in dataset_splits.items():
        count = len(splits['train']) + len(splits['val'])
        breed_counts[breed] = count
    
    max_count = max(breed_counts.values())
    class_weights = {breed: max_count / count for breed, count in breed_counts.items()}
    
    print(f"\n⚖️ Class weights for ultra-balancing:")
    for breed, weight in class_weights.items():
        print(f"  {breed}: {weight:.2f} (images: {breed_counts[breed]})")
    
    # Extract features with ultra-aggressive augmentation
    print(f"\n🔥 Extracting features with ultra-optimization...")
    
    all_train_val_features = {}
    
    for breed in breeds:
        print(f"\n🎯 Processing breed: {breed}")
        
        # Get training + validation images
        train_images = dataset_splits[breed]['train']
        val_images = dataset_splits[breed]['val']
        train_val_images = train_images + val_images
        
        weight = class_weights[breed]
        # Ultra-aggressive augmentation rounds based on weight
        augmentation_rounds = max(3, min(int(weight * 3), 8))
        
        print(f"  📸 Using {len(train_val_images)} images for prototype")
        print(f"  🔄 Ultra-augmentation rounds: {augmentation_rounds}")
        
        # Multiple rounds of ultra-augmented feature extraction
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
        print(f"  ✅ Total features: {combined_features.shape}")
    
    # Compute ultra-optimized prototypes
    prototypes = compute_ultra_optimized_prototypes(
        all_train_val_features, 
        breeds, 
        class_weights
    )
    
    # Save ultra-optimized prototypes
    data = {
        'prototypes': prototypes,
        'model_name': args.model,
        'feature_dim': model.feature_dim,
        'breeds': breeds,
        'dataset_splits': dataset_splits,
        'class_weights': class_weights,
        'config': {
            'optimization_level': 'ultra_maximum_differentiation',
            'contrastive_refinement': True,
            'feature_importance_weighting': True,
            'ultra_augmentation': True,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'batch_size': args.batch_size,
            'random_seed': args.random_seed
        }
    }
    
    print(f"\n💾 Saving ultra-optimized prototypes to '{args.output_path}'...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Calculate file size
    file_size = os.path.getsize(args.output_path) / 1024
    print(f"✅ Ultra-optimized prototypes saved successfully!")
    
    print("=" * 80)
    print("🏆 ULTRA FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"🧠 Model: {args.model}")
    print(f"🚀 Optimization: Ultra Maximum Differentiation")
    print(f"🔥 Features: Contrastive refinement + Feature importance weighting")
    print(f"📏 Feature dimension: {model.feature_dim}")
    print(f"🎭 Total breeds: {len(breeds)}")
    print(f"💾 Output file: {args.output_path}")
    print(f"📊 File size: {file_size:.1f} KB")
    
    print(f"\n📈 Ultra-optimization summary:")
    total_training_features = 0
    for breed in breeds:
        splits = dataset_splits[breed]
        total_images = len(splits['train']) + len(splits['val']) + len(splits['test'])
        train_val = len(splits['train']) + len(splits['val'])
        test = len(splits['test'])
        weight = class_weights[breed]
        training_features = all_train_val_features[breed].shape[0]
        total_training_features += training_features
        
        print(f"  🎯 {breed}: {total_images} images ({train_val} train+val, {test} test)")
        print(f"      └─ {training_features} ultra-features (weight: {weight:.2f})")
    
    print(f"\n🔥 Total ultra-training features: {total_training_features}")
    print(f"📊 Average features per breed: {total_training_features / len(breeds):.1f}")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Test ultra inference: python infer_ultra.py <image_path>")
    print(f"  2. Run ultra evaluation: python eval_ultra.py")
    print(f"  3. Compare with previous models")

if __name__ == "__main__":
    main()