#!/usr/bin/env python3
"""
HACKATHON MAXIMUM ACCURACY TRAINING PIPELINE

Specialized training system designed for competition-level performance
with 11 breeds including Sahiwal, targeting 95%+ accuracy.

Features:
- Maximum discrimination training
- Ensemble model approach
- Advanced augmentation for competition edge
- Real-time accuracy monitoring
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Add utils to path
sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device
from utils.dataset import discover_breeds, get_breed_images, split_dataset

class HackathonTrainingPipeline:
    """Competition-grade training pipeline for maximum accuracy"""
    
    def __init__(self, config_path='hackathon_analysis.json'):
        self.device = get_device()
        self.load_config(config_path)
        self.setup_logging()
        
        print(f"🏆 HACKATHON TRAINING PIPELINE INITIALIZED")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Target Accuracy: {self.config['hackathon_optimizations']['target_accuracy']*100:.1f}%")
        print(f"🔢 Ensemble Models: {self.config['hackathon_optimizations']['ensemble_models']}")
    
    def load_config(self, config_path):
        """Load hackathon configuration"""
        with open(config_path, 'r') as f:
            data = json.load(f)
            self.config = data['hackathon_config']
            self.breeds_analysis = data['breeds_analysis']
        
        self.breeds = list(self.breeds_analysis.keys())
        self.num_classes = len(self.breeds)
        print(f"🧬 Training {self.num_classes} breeds: {', '.join(self.breeds)}")
    
    def setup_logging(self):
        """Setup training progress logging"""
        self.training_history = {
            'accuracy': [],
            'loss': [],
            'validation_accuracy': [],
            'validation_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
    
    def get_competition_transforms(self, augment=True):
        """Get transforms optimized for hackathon competition"""
        
        if augment:
            # Advanced competition-grade augmentation
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Competition-specific augmentations
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_ensemble_models(self):
        """Create ensemble of models for maximum accuracy"""
        models = []
        
        for i in range(self.config['hackathon_optimizations']['ensemble_models']):
            model = FeatureExtractor(
                model_name=self.config['model_config']['architecture'], 
                pretrained=True
            )
            
            # Replace classifier for our number of breeds
            if hasattr(model.model, 'fc'):
                model.model.fc = nn.Sequential(
                    nn.Dropout(self.config['model_config']['dropout']),
                    nn.Linear(model.model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, self.num_classes)
                )
            
            model.to(self.device)
            models.append(model)
            
            print(f"✅ Created ensemble model {i+1}/{self.config['hackathon_optimizations']['ensemble_models']}")
        
        return models
    
    def prepare_competition_dataset(self):
        """Prepare dataset with competition-level augmentation"""
        
        print("📊 Preparing competition dataset...")
        
        # Discover and split dataset
        breeds = discover_breeds("dataset")
        if set(breeds) != set(self.breeds):
            print(f"⚠️  Warning: Found breeds {breeds} vs expected {self.breeds}")
        
        # Create balanced dataset with smart augmentation
        all_images = []
        all_labels = []
        
        for breed_idx, breed in enumerate(self.breeds):
            breed_images = get_breed_images("dataset", breed)
            breed_count = len(breed_images)
            
            # Apply breed-specific augmentation multiplier
            aug_multiplier = self.config.get('augmentation_multiplier', {}).get(breed, 1)
            
            print(f"🔄 {breed}: {breed_count} base images × {aug_multiplier} augmentation = {breed_count * aug_multiplier} total")
            
            # Add base images
            all_images.extend(breed_images)
            all_labels.extend([breed_idx] * breed_count)
            
            # Add augmented images for smaller datasets
            if aug_multiplier > 1:
                for _ in range(aug_multiplier - 1):
                    all_images.extend(breed_images)
                    all_labels.extend([breed_idx] * breed_count)
        
        print(f"📈 Total training samples: {len(all_images)}")
        
        # Split into train/validation using sklearn
        from sklearn.model_selection import train_test_split
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        return (train_images, train_labels), (val_images, val_labels)
    
    def train_ensemble_model(self, model_idx, train_data, val_data):
        """Train individual model in ensemble"""
        
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        
        print(f"\n🚀 Training Ensemble Model {model_idx + 1}")
        print("=" * 50)
        
        # Create model
        model = FeatureExtractor(self.config['model_config']['architecture'], pretrained=True)
        
        # Replace classifier
        if hasattr(model.model, 'fc'):
            model.model.fc = nn.Sequential(
                nn.Dropout(self.config['model_config']['dropout']),
                nn.Linear(model.model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes)
            )
        
        model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['model_config']['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        best_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['model_config']['epochs']):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            with tqdm(range(len(train_images)), desc=f"Epoch {epoch+1}") as pbar:
                for i in pbar:
                    # Load and transform image
                    try:
                        from PIL import Image
                        image = Image.open(train_images[i]).convert('RGB')
                        image = self.get_competition_transforms(augment=True)(image).unsqueeze(0).to(self.device)
                        label = torch.tensor([train_labels[i]], dtype=torch.long).to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(image)
                        loss = criterion(outputs, label)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        # Statistics
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == label).sum().item()
                        
                        # Update progress bar
                        current_accuracy = train_correct / (i + 1) * 100
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Acc': f'{current_accuracy:.2f}%'
                        })
                        
                    except Exception as e:
                        print(f"⚠️  Error processing {train_images[i]}: {e}")
                        continue
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for i in range(len(val_images)):
                    try:
                        image = Image.open(val_images[i]).convert('RGB')
                        image = self.get_competition_transforms(augment=False)(image).unsqueeze(0).to(self.device)
                        label = torch.tensor([val_labels[i]], dtype=torch.long).to(self.device)
                        
                        outputs = model(image)
                        loss = criterion(outputs, label)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += 1
                        val_correct += (predicted == label).sum().item()
                        
                    except Exception as e:
                        continue
            
            # Calculate metrics
            epoch_time = time.time() - start_time
            train_accuracy = train_correct / len(train_images) * 100
            val_accuracy = val_correct / val_total * 100 if val_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{self.config['model_config']['epochs']}")
            print(f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Time: {epoch_time:.1f}s")
            
            # Learning rate scheduling
            scheduler.step(val_accuracy)
            
            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'models/hackathon_ensemble_{model_idx}_best.pth')
                print(f"🎯 New best accuracy: {best_accuracy:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['model_config']['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Check if target accuracy reached
            if val_accuracy >= self.config['hackathon_optimizations']['target_accuracy'] * 100:
                print(f"🏆 TARGET ACCURACY REACHED: {val_accuracy:.2f}%")
                break
        
        print(f"✅ Model {model_idx + 1} training complete. Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy
    
    def train_hackathon_models(self):
        """Train ensemble of models for hackathon"""
        
        print("🏆 STARTING HACKATHON TRAINING PIPELINE")
        print("=" * 60)
        
        # Prepare dataset
        train_data, val_data = self.prepare_competition_dataset()
        
        # Train ensemble models
        ensemble_accuracies = []
        
        for model_idx in range(self.config['hackathon_optimizations']['ensemble_models']):
            accuracy = self.train_ensemble_model(model_idx, train_data, val_data)
            ensemble_accuracies.append(accuracy)
        
        # Save ensemble information
        ensemble_info = {
            'num_models': len(ensemble_accuracies),
            'individual_accuracies': ensemble_accuracies,
            'average_accuracy': np.mean(ensemble_accuracies),
            'best_accuracy': max(ensemble_accuracies),
            'breeds': self.breeds,
            'num_classes': self.num_classes,
            'total_images': len(train_data[0]) + len(val_data[0])
        }
        
        with open('models/hackathon_ensemble_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print("\n🎯 HACKATHON TRAINING RESULTS")
        print("=" * 60)
        print(f"✅ Ensemble Size: {len(ensemble_accuracies)} models")
        print(f"🎯 Individual Accuracies: {[f'{acc:.2f}%' for acc in ensemble_accuracies]}")
        print(f"📊 Average Accuracy: {np.mean(ensemble_accuracies):.2f}%")
        print(f"🏆 Best Individual: {max(ensemble_accuracies):.2f}%")
        print(f"🧬 Breeds Trained: {self.num_classes}")
        print(f"📈 Total Training Samples: {ensemble_info['total_images']}")
        
        if max(ensemble_accuracies) >= self.config['hackathon_optimizations']['target_accuracy'] * 100:
            print(f"🏆 HACKATHON READY! Target accuracy achieved!")
        else:
            print(f"⚠️  Consider additional training or data augmentation")
        
        return ensemble_info

def main():
    """Main training function"""
    
    # Check if analysis exists
    if not os.path.exists('hackathon_analysis.json'):
        print("❌ Please run analyze_hackathon_dataset.py first")
        return
    
    # Initialize and run training pipeline
    pipeline = HackathonTrainingPipeline()
    results = pipeline.train_hackathon_models()
    
    print(f"\n💾 Models saved to: models/hackathon_ensemble_*.pth")
    print(f"📊 Results saved to: models/hackathon_ensemble_info.json")
    print(f"🏆 Ready for hackathon deployment!")

if __name__ == "__main__":
    main()