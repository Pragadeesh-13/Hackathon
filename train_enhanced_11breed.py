#!/usr/bin/env python3
"""
ENHANCED 11-BREED TRAINING WITH NAGPURI INTEGRATION

Fine-tuned training for maximum accuracy with all 11 breeds including Nagpuri.
Integrates with existing smart fusion inference system for best predictions.

Features:
- All 11 breeds including Nagpuri
- Fine-tuned for maximum accuracy
- Compatible with infer_smart_fusion.py
- Optimized for test_accuracy.py validation
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
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device
from utils.dataset import discover_breeds, get_breed_images

class Enhanced11BreedDataset(Dataset):
    """Dataset class optimized for 11 breeds with smart augmentation"""
    
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
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx]

class Enhanced11BreedTrainer:
    """Enhanced trainer for 11 breeds with maximum accuracy fine-tuning"""
    
    def __init__(self):
        self.device = get_device()
        self.dataset_path = "dataset"
        self.discover_breeds()
        self.setup_model()
        
        print(f"🎯 ENHANCED 11-BREED TRAINER INITIALIZED")
        print(f"📱 Device: {self.device}")
        print(f"🧬 Breeds: {self.num_classes} ({', '.join(self.breeds)})")
        print(f"📊 Total Images: {self.total_images}")
    
    def discover_breeds(self):
        """Discover all breeds in dataset"""
        
        self.breeds = discover_breeds(self.dataset_path)
        self.num_classes = len(self.breeds)
        
        # Create mappings
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for idx, breed in enumerate(self.breeds)}
        
        # Count images
        self.breed_counts = {}
        self.total_images = 0
        
        for breed in self.breeds:
            images = get_breed_images(self.dataset_path, breed)
            count = len(images)
            self.breed_counts[breed] = count
            self.total_images += count
            
            quality = "🏆" if count >= 50 else "✅" if count >= 30 else "⚠️"
            print(f"{quality} {breed}: {count} images")
        
        print(f"📈 Total dataset: {self.total_images} images across {self.num_classes} breeds")
    
    def setup_model(self):
        """Setup model architecture"""
        
        self.model = FeatureExtractor('resnet50', pretrained=True, num_classes=self.num_classes)
        self.model.to(self.device)
        
        # Setup optimizers with different learning rates for different parts
        self.criterion = nn.CrossEntropyLoss()
        
        # Separate parameters for fine-tuning
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Different learning rates for backbone and classifier
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 0.0001},  # Lower LR for pretrained backbone
            {'params': classifier_params, 'lr': 0.001}   # Higher LR for new classifier
        ], weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
    
    def get_transforms(self, augment=True):
        """Get enhanced transforms for maximum accuracy"""
        
        if augment:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_datasets(self):
        """Prepare datasets with smart stratification"""
        
        print("📊 Preparing enhanced datasets...")
        
        all_images = []
        all_labels = []
        
        # Collect all images with balanced sampling for smaller breeds
        for breed_idx, breed in enumerate(self.breeds):
            images = get_breed_images(self.dataset_path, breed)
            
            # Smart augmentation for smaller datasets
            if len(images) < 40:
                # Duplicate smaller breeds to balance dataset
                images = images * 2
                print(f"🔄 {breed}: Augmented to {len(images)} samples")
            
            all_images.extend(images)
            all_labels.extend([breed_idx] * len(images))
        
        print(f"📈 Total training samples: {len(all_images)}")
        
        # Stratified split
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Create datasets
        train_dataset = Enhanced11BreedDataset(
            train_images, train_labels, self.get_transforms(augment=True)
        )
        val_dataset = Enhanced11BreedDataset(
            val_images, val_labels, self.get_transforms(augment=False)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"✅ Training: {len(train_dataset)} samples")
        print(f"✅ Validation: {len(val_dataset)} samples")
        
        return len(train_dataset), len(val_dataset)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model with detailed per-breed analysis"""
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-breed tracking
        breed_correct = defaultdict(int)
        breed_total = defaultdict(int)
        breed_predictions = defaultdict(list)
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-breed analysis
                for i in range(labels.size(0)):
                    true_breed = self.idx_to_breed[labels[i].item()]
                    pred_breed = self.idx_to_breed[predicted[i].item()]
                    
                    breed_total[true_breed] += 1
                    breed_predictions[true_breed].append(pred_breed)
                    
                    if predicted[i] == labels[i]:
                        breed_correct[true_breed] += 1
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        # Calculate per-breed accuracies
        breed_accuracies = {}
        for breed in self.breeds:
            if breed_total[breed] > 0:
                breed_accuracies[breed] = breed_correct[breed] / breed_total[breed] * 100
            else:
                breed_accuracies[breed] = 0
        
        return val_loss, val_acc, breed_accuracies
    
    def train(self, epochs=80, patience=15):
        """Enhanced training with fine-tuning"""
        
        print(f"\n🚀 STARTING ENHANCED 11-BREED TRAINING")
        print("=" * 70)
        
        # Prepare datasets
        train_size, val_size = self.prepare_datasets()
        
        best_accuracy = 0.0
        patience_counter = 0
        training_history = []
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, breed_accuracies = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Time tracking
            epoch_time = time.time() - start_time
            
            # Print results
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Show breed performance (focus on new additions like Nagpuri)
            key_breeds = ['Nagpuri', 'Sahiwal', 'Bhadawari']  # Show important breeds
            print("Key breed accuracies:")
            for breed in key_breeds:
                if breed in breed_accuracies:
                    acc = breed_accuracies[breed]
                    status = "🎯" if acc >= 90 else "✅" if acc >= 80 else "⚠️"
                    print(f"  {status} {breed}: {acc:.1f}%")
            
            # Save training history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'breed_accuracies': breed_accuracies
            })
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                
                # Save model for compatibility with existing inference
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/enhanced_11breed_model.pth')
                
                # Save breed information for inference compatibility
                model_info = {
                    'breeds': self.breeds,
                    'num_classes': self.num_classes,
                    'best_accuracy': best_accuracy,
                    'breed_accuracies': breed_accuracies,
                    'breed_to_idx': self.breed_to_idx,
                    'idx_to_breed': self.idx_to_breed,
                    'total_images': self.total_images,
                    'epoch': epoch
                }
                
                with open('models/enhanced_11breed_info.json', 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                print(f"🎯 New best accuracy: {best_accuracy:.2f}%")
                
                # Check target achievement
                if val_acc >= 95.0:
                    print(f"🏆 TARGET ACCURACY ACHIEVED: {val_acc:.2f}%")
                    break
                    
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save complete training history
        with open('models/enhanced_11breed_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Final results
        print(f"\n🎯 TRAINING COMPLETE")
        print("=" * 70)
        print(f"✅ Best Accuracy: {best_accuracy:.2f}%")
        print(f"📊 Epochs Trained: {epoch}")
        print(f"🧬 Breeds: {self.num_classes} (including Nagpuri)")
        print(f"💾 Model: models/enhanced_11breed_model.pth")
        print(f"📋 Info: models/enhanced_11breed_info.json")
        
        return best_accuracy
    
    def create_compatible_inference(self):
        """Create inference compatible with existing smart fusion system"""
        
        inference_code = '''#!/usr/bin/env python3
"""
ENHANCED 11-BREED INFERENCE

Compatible with existing smart fusion system.
Usage: python infer_enhanced_11breed.py <image_path>
"""

import sys
import torch
import json
from pathlib import Path

sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device
from infer_smart_fusion import SmartFusionInference

class Enhanced11BreedInference(SmartFusionInference):
    def __init__(self, model_path='models/enhanced_11breed_model.pth',
                 info_path='models/enhanced_11breed_info.json'):
        
        # Load model info
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.device = get_device()
        self.breeds = self.model_info['breeds']
        self.num_classes = self.model_info['num_classes']
        
        # Load model
        self.model = FeatureExtractor('resnet50', pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🎯 Enhanced 11-Breed Model Loaded")
        print(f"🧬 Breeds: {len(self.breeds)} including Nagpuri")
        print(f"📊 Training Accuracy: {self.model_info['best_accuracy']:.2f}%")

def main():
    if len(sys.argv) != 2:
        print("Usage: python infer_enhanced_11breed.py <image_path>")
        return
    
    image_path = sys.argv[1]
    inferencer = Enhanced11BreedInference()
    
    # Use smart fusion prediction if available, otherwise standard prediction
    try:
        result = inferencer.predict_with_smart_fusion(image_path)
    except:
        result = inferencer.predict_standard(image_path)
    
    print(f"\\n🎯 PREDICTION: {result['predicted_breed']}")
    print(f"🎪 Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()
'''
        
        with open('infer_enhanced_11breed.py', 'w') as f:
            f.write(inference_code)
        
        print("✅ Compatible inference created: infer_enhanced_11breed.py")

def main():
    """Main training function"""
    
    print("🎯 ENHANCED 11-BREED TRAINING WITH NAGPURI")
    print("=" * 80)
    print("📋 Integrating Nagpuri with existing smart fusion system")
    print("🎪 Optimized for test_accuracy.py and infer_smart_fusion.py")
    print()
    
    # Initialize and train
    trainer = Enhanced11BreedTrainer()
    best_accuracy = trainer.train()
    
    # Create compatible inference
    trainer.create_compatible_inference()
    
    print(f"\n🏆 INTEGRATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Model ready with Nagpuri integration")
    print(f"🎯 Best Accuracy: {best_accuracy:.2f}%")
    print(f"🧪 Test with: python test_accuracy.py")
    print(f"🔮 Infer with: python infer_smart_fusion.py <image>")
    print(f"🎪 Enhanced: python infer_enhanced_11breed.py <image>")
    print("=" * 80)

if __name__ == "__main__":
    main()