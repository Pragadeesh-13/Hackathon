#!/usr/bin/env python3
"""
Hackathon Dataset Analyzer and Training Pipeline

This script analyzes the current dataset including new breeds
and creates an optimized training pipeline for hackathon performance.
"""

import os
import sys
from pathlib import Path
import json
from collections import defaultdict

def analyze_hackathon_dataset(dataset_dir="dataset"):
    """Analyze dataset for hackathon training"""
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory {dataset_dir} not found")
        return {}
    
    breeds_analysis = {}
    total_images = 0
    
    print("🏆 HACKATHON DATASET ANALYSIS")
    print("=" * 60)
    
    for breed_dir in sorted(dataset_path.iterdir()):
        if breed_dir.is_dir():
            # Count different image formats
            image_files = (
                list(breed_dir.glob("*.jpg")) + 
                list(breed_dir.glob("*.jpeg")) + 
                list(breed_dir.glob("*.JPG")) +
                list(breed_dir.glob("*.png")) +
                list(breed_dir.glob("*.webp"))
            )
            
            image_count = len(image_files)
            if image_count > 0:
                breeds_analysis[breed_dir.name] = {
                    'count': image_count,
                    'files': [str(f) for f in image_files[:3]],  # Sample files
                    'status': 'excellent' if image_count >= 30 else 'good' if image_count >= 15 else 'needs_augmentation'
                }
                total_images += image_count
    
    # Print analysis
    for breed, info in breeds_analysis.items():
        count = info['count']
        status = info['status']
        
        if status == 'excellent':
            status_icon = "🥇"
            status_text = "EXCELLENT"
        elif status == 'good':
            status_icon = "🥈"
            status_text = "GOOD"
        else:
            status_icon = "🔧"
            status_text = "NEEDS AUGMENTATION"
        
        print(f"{status_icon} {breed:15} | {count:3d} images | {status_text}")
    
    print("=" * 60)
    print(f"📊 TOTAL: {len(breeds_analysis)} breeds, {total_images} images")
    
    # Recommendations
    print(f"\n🎯 HACKATHON RECOMMENDATIONS:")
    excellent_breeds = [b for b, info in breeds_analysis.items() if info['status'] == 'excellent']
    good_breeds = [b for b, info in breeds_analysis.items() if info['status'] == 'good']
    needs_aug_breeds = [b for b, info in breeds_analysis.items() if info['status'] == 'needs_augmentation']
    
    print(f"✅ Excellent breeds ({len(excellent_breeds)}): {', '.join(excellent_breeds)}")
    print(f"👍 Good breeds ({len(good_breeds)}): {', '.join(good_breeds)}")
    print(f"🔧 Need augmentation ({len(needs_aug_breeds)}): {', '.join(needs_aug_breeds)}")
    
    return breeds_analysis

def create_hackathon_config(breeds_analysis):
    """Create optimized config for hackathon training"""
    
    config = {
        'model_config': {
            'architecture': 'resnet50',
            'pretrained': True,
            'dropout': 0.3,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 100,
            'early_stopping_patience': 15
        },
        'augmentation_config': {
            'base_augmentations': [
                'horizontal_flip',
                'rotation_15',
                'brightness_0.2',
                'contrast_0.2',
                'color_jitter'
            ],
            'advanced_augmentations': [
                'random_perspective',
                'gaussian_blur',
                'random_erasing',
                'cutmix',
                'mixup'
            ]
        },
        'training_strategy': {
            'feature_extraction_rounds': 5,  # Maximum feature extraction
            'contrastive_learning': True,
            'hard_negative_mining': True,
            'cross_breed_discrimination': True,
            'morphological_features': True
        },
        'hackathon_optimizations': {
            'target_accuracy': 0.95,  # 95% minimum accuracy
            'margin_threshold': 0.1,  # Minimum confidence margin
            'ensemble_models': 3,     # Use 3 models for voting
            'test_time_augmentation': True
        }
    }
    
    # Adaptive augmentation based on dataset size
    total_breeds = len(breeds_analysis)
    for breed, info in breeds_analysis.items():
        if info['count'] < 15:
            config['augmentation_multiplier'] = config.get('augmentation_multiplier', {})
            config['augmentation_multiplier'][breed] = 8  # Heavy augmentation
        elif info['count'] < 25:
            config['augmentation_multiplier'] = config.get('augmentation_multiplier', {})
            config['augmentation_multiplier'][breed] = 4  # Medium augmentation
    
    return config

def main():
    print("🚀 HACKATHON PREPARATION - DATASET ANALYSIS")
    print("📅 Optimizing for maximum competition performance")
    print()
    
    # Analyze current dataset
    breeds_analysis = analyze_hackathon_dataset()
    
    if not breeds_analysis:
        print("❌ No breeds found. Please check your dataset directory.")
        return
    
    # Create hackathon config
    config = create_hackathon_config(breeds_analysis)
    
    # Save analysis results
    with open('hackathon_analysis.json', 'w') as f:
        json.dump({
            'breeds_analysis': breeds_analysis,
            'hackathon_config': config
        }, f, indent=2)
    
    print(f"\n💾 Analysis saved to: hackathon_analysis.json")
    print(f"🎯 Ready for hackathon training with {len(breeds_analysis)} breeds!")
    print(f"🏆 Target accuracy: {config['hackathon_optimizations']['target_accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()