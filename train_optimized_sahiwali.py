#!/usr/bin/env python3
"""
OPTIMIZED SINGLE MODEL TRAINING FOR SAHIWALI

High-performance training specifically optimized for CPU training
with maximum accuracy for all 11 breeds including Sahiwali.

Features:
- Optimized for CPU training speed
- Smart learning rate scheduling
- Advanced data augmentation
- Early stopping with patience
- Validation-based model selection
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device
from utils.dataset import discover_breeds, get_breed_images

class CattleDataset(Dataset):
    """Optimized dataset class for cattle breed recognition"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx]

class OptimizedTrainer:
    """Optimized trainer for maximum accuracy with CPU efficiency"""
    
    def __init__(self, config_path='hackathon_analysis.json'):
        self.device = get_device()
        self.load_config(config_path)
        self.setup_model()
        
        print(f"🎯 OPTIMIZED TRAINING INITIALIZED")
        print(f"📱 Device: {self.device}")
        print(f"🧬 Breeds: {len(self.breeds)} ({', '.join(self.breeds)})")
        print(f"🎯 Target Accuracy: 95%+")
    
    def load_config(self, config_path):
        """Load configuration from analysis"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        self.breeds_analysis = data['breeds_analysis']
        self.breeds = list(self.breeds_analysis.keys())
        self.num_classes = len(self.breeds)
        
        # Create breed mappings
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for idx, breed in enumerate(self.breeds)}
    
    def setup_model(self):
        """Setup the model with optimized architecture"""
        self.model = FeatureExtractor('resnet50', pretrained=True, num_classes=self.num_classes)
        self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=0.01
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
    
    def get_transforms(self, augment=True):
        """Get optimized transforms for training"""
        
        if augment:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_dataset(self):
        """Prepare dataset with smart augmentation"""
        
        print("📊 Preparing optimized dataset...")
        
        all_images = []
        all_labels = []
        
        for breed_idx, breed in enumerate(self.breeds):
            breed_images = get_breed_images("dataset", breed)
            breed_count = len(breed_images)
            
            print(f"📥 {breed}: {breed_count} images")
            
            all_images.extend(breed_images)
            all_labels.extend([breed_idx] * breed_count)
        
        print(f"📈 Total samples: {len(all_images)}")
        
        # Split dataset
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Create datasets
        train_dataset = CattleDataset(train_images, train_labels, self.get_transforms(augment=True))
        val_dataset = CattleDataset(val_images, val_labels, self.get_transforms(augment=False))
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
        return len(train_dataset), len(val_dataset)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_accuracy = correct_predictions / total_samples * 100
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples * 100
        
        return epoch_loss, epoch_accuracy
    
    def validate(self):
        """Validate the model"""
        
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Per-breed accuracy tracking
        breed_correct = defaultdict(int)
        breed_total = defaultdict(int)
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Per-breed statistics
                for i in range(labels.size(0)):
                    true_breed = self.idx_to_breed[labels[i].item()]
                    breed_total[true_breed] += 1
                    if predicted[i] == labels[i]:
                        breed_correct[true_breed] += 1
        
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = correct_predictions / total_samples * 100
        
        # Calculate per-breed accuracies
        breed_accuracies = {}
        for breed in self.breeds:
            if breed_total[breed] > 0:
                breed_accuracies[breed] = breed_correct[breed] / breed_total[breed] * 100
            else:
                breed_accuracies[breed] = 0
        
        return val_loss, val_accuracy, breed_accuracies
    
    def train(self, epochs=50, patience=10):
        """Train the model with early stopping"""
        
        print(f"\n🚀 Starting Optimized Training")
        print("=" * 60)
        
        # Prepare dataset
        train_size, val_size = self.prepare_dataset()
        
        # Training loop
        best_accuracy = 0.0
        patience_counter = 0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, breed_accuracies = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Show per-breed accuracy
            print("Per-breed accuracy:")
            for breed, acc in breed_accuracies.items():
                status = "✅" if acc >= 90 else "⚠️" if acc >= 70 else "❌"
                print(f"  {status} {breed}: {acc:.1f}%")
            
            # Early stopping and model saving
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'models/optimized_sahiwali_best.pth')
                
                # Save breed accuracies
                with open('models/optimized_results.json', 'w') as f:
                    json.dump({
                        'best_accuracy': best_accuracy,
                        'breed_accuracies': breed_accuracies,
                        'epoch': epoch,
                        'breeds': self.breeds
                    }, f, indent=2)
                
                print(f"🎯 New best accuracy: {best_accuracy:.2f}%")
                
                # Check if target reached
                if val_acc >= 95.0:
                    print(f"🏆 TARGET ACCURACY REACHED: {val_acc:.2f}%")
                    break
                    
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Final results
        print(f"\n🎯 TRAINING COMPLETE")
        print("=" * 60)
        print(f"✅ Best Accuracy: {best_accuracy:.2f}%")
        print(f"📊 Total Epochs: {epoch}")
        print(f"🧬 Breeds Trained: {len(self.breeds)}")
        print(f"💾 Model saved: models/optimized_sahiwali_best.pth")
        
        return best_accuracy, training_history

def main():
    """Main training function"""
    
    # Check requirements
    if not os.path.exists('hackathon_analysis.json'):
        print("❌ Please run analyze_hackathon_dataset.py first")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize and train
    trainer = OptimizedTrainer()
    best_accuracy, history = trainer.train()
    
    print(f"\n🚀 Training complete! Best accuracy: {best_accuracy:.2f}%")
    
    if best_accuracy >= 95.0:
        print("🏆 HACKATHON READY - Target accuracy achieved!")
    else:
        print(f"⚠️  Consider additional training or data augmentation")

if __name__ == "__main__":
    main()