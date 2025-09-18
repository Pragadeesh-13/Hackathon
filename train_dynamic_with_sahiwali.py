#!/usr/bin/env python3
"""
DYNAMIC MULTI-BREED TRAINER WITH SAHIWALI

Automatically detects all available breeds in the dataset and trains
a high-accuracy model with advanced fine-tuning for optimal predictions.

Features:
- Automatic breed detection
- Progressive fine-tuning
- Smart data augmentation
- Real-time accuracy monitoring
- Best model selection
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
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device
from utils.dataset import discover_breeds, get_breed_images

class AdvancedCattleDataset(Dataset):
    """Advanced dataset class with smart augmentation"""
    
    def __init__(self, image_paths, labels, breed_names, transform=None, augment_factor=1):
        self.image_paths = image_paths
        self.labels = labels
        self.breed_names = breed_names
        self.transform = transform
        self.augment_factor = augment_factor
        
        # Create augmented dataset
        if augment_factor > 1:
            self.image_paths = self.image_paths * augment_factor
            self.labels = self.labels * augment_factor
    
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
            # Return a dummy image
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx]

class DynamicBreedTrainer:
    """Dynamic trainer that automatically adapts to any number of breeds"""
    
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.device = get_device()
        self.discover_and_analyze_breeds()
        self.setup_training_config()
        
        print(f"🎯 DYNAMIC MULTI-BREED TRAINER INITIALIZED")
        print(f"📱 Device: {self.device}")
        print(f"🧬 Breeds Detected: {self.num_classes}")
        print(f"📊 Total Images: {self.total_images}")
        self.print_breed_summary()
    
    def discover_and_analyze_breeds(self):
        """Automatically discover and analyze all breeds in dataset"""
        
        print("🔍 Discovering breeds in dataset...")
        
        self.breeds = discover_breeds(self.dataset_path)
        self.num_classes = len(self.breeds)
        
        # Create breed mappings
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for idx, breed in enumerate(self.breeds)}
        
        # Analyze each breed
        self.breed_stats = {}
        self.total_images = 0
        
        for breed in self.breeds:
            images = get_breed_images(self.dataset_path, breed)
            count = len(images)
            self.breed_stats[breed] = {
                'count': count,
                'images': images,
                'quality': 'excellent' if count >= 40 else 'good' if count >= 25 else 'fair'
            }
            self.total_images += count
        
        # Save analysis
        analysis = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'breeds': self.breeds,
            'num_classes': self.num_classes,
            'total_images': self.total_images,
            'breed_stats': {k: {'count': v['count'], 'quality': v['quality']} 
                          for k, v in self.breed_stats.items()}
        }
        
        with open('dynamic_breed_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Analysis saved to: dynamic_breed_analysis.json")
    
    def print_breed_summary(self):
        """Print summary of discovered breeds"""
        
        print(f"\n📋 BREED SUMMARY")
        print("=" * 60)
        
        for breed, stats in self.breed_stats.items():
            status_icon = "🏆" if stats['quality'] == 'excellent' else "✅" if stats['quality'] == 'good' else "⚠️"
            print(f"{status_icon} {breed}: {stats['count']} images ({stats['quality']})")
        
        print("=" * 60)
    
    def setup_training_config(self):
        """Setup training configuration based on dataset analysis"""
        
        # Base configuration
        self.config = {
            'epochs': 100,
            'batch_size': 16 if self.total_images < 500 else 32,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'patience': 15,
            'target_accuracy': 95.0,
            'min_improvement': 0.5
        }
        
        # Adjust based on dataset size
        if self.total_images < 300:
            self.config['epochs'] = 150
            self.config['patience'] = 20
            print("📊 Small dataset detected - extended training config")
        elif self.total_images > 1000:
            self.config['batch_size'] = 64
            self.config['learning_rate'] = 0.0005
            print("📊 Large dataset detected - optimized training config")
    
    def get_progressive_transforms(self, stage='base', augment=True):
        """Get progressive transforms that increase in complexity"""
        
        if not augment:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if stage == 'base':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif stage == 'enhanced':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif stage == 'advanced':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_datasets(self, augment_stage='enhanced'):
        """Prepare training and validation datasets"""
        
        print(f"📊 Preparing datasets with {augment_stage} augmentation...")
        
        all_images = []
        all_labels = []
        
        # Collect all images and labels
        for breed_idx, breed in enumerate(self.breeds):
            images = self.breed_stats[breed]['images']
            all_images.extend(images)
            all_labels.extend([breed_idx] * len(images))
        
        # Stratified split to ensure balanced validation
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=all_labels
        )
        
        # Create datasets with smart augmentation
        train_dataset = AdvancedCattleDataset(
            train_images, train_labels, self.breeds,
            transform=self.get_progressive_transforms(augment_stage, augment=True),
            augment_factor=2 if self.total_images < 400 else 1
        )
        
        val_dataset = AdvancedCattleDataset(
            val_images, val_labels, self.breeds,
            transform=self.get_progressive_transforms(augment=False)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
        return len(train_dataset), len(val_dataset)
    
    def create_model(self, pretrained=True):
        """Create and configure the model"""
        
        model = FeatureExtractor('resnet50', pretrained=pretrained, num_classes=self.num_classes)
        model.to(self.device)
        
        return model
    
    def train_epoch(self, model, optimizer, criterion, epoch, stage="training"):
        """Train for one epoch"""
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} ({stage})")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
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
    
    def validate(self, model, criterion):
        """Validate the model"""
        
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Per-breed accuracy tracking
        breed_correct = defaultdict(int)
        breed_total = defaultdict(int)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Per-breed statistics
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    
                    true_breed = self.idx_to_breed[true_label]
                    breed_total[true_breed] += 1
                    
                    if pred_label == true_label:
                        breed_correct[true_breed] += 1
                    
                    confusion_matrix[true_label][pred_label] += 1
        
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = correct_predictions / total_samples * 100
        
        # Calculate per-breed accuracies
        breed_accuracies = {}
        for breed in self.breeds:
            if breed_total[breed] > 0:
                breed_accuracies[breed] = breed_correct[breed] / breed_total[breed] * 100
            else:
                breed_accuracies[breed] = 0
        
        return val_loss, val_accuracy, breed_accuracies, confusion_matrix
    
    def progressive_training(self):
        """Progressive training with multiple stages"""
        
        print(f"\n🚀 STARTING PROGRESSIVE TRAINING")
        print("=" * 80)
        
        # Stage 1: Base training
        print(f"🎯 Stage 1: Base Training")
        print("-" * 40)
        
        self.prepare_datasets('base')
        model = self.create_model(pretrained=True)
        
        best_accuracy, model, history = self.train_stage(
            model, stage_name="Base", epochs=30
        )
        
        if best_accuracy < 70:
            print("⚠️  Base accuracy too low, extending training...")
            best_accuracy, model, _ = self.train_stage(
                model, stage_name="Extended Base", epochs=20
            )
        
        # Stage 2: Enhanced training with more augmentation
        print(f"\n🎯 Stage 2: Enhanced Training")
        print("-" * 40)
        
        self.prepare_datasets('enhanced')
        
        # Reduce learning rate for fine-tuning
        for param_group in model.parameters():
            param_group.requires_grad = True
        
        best_accuracy, model, _ = self.train_stage(
            model, stage_name="Enhanced", epochs=30, lr=0.0005
        )
        
        # Stage 3: Advanced fine-tuning (if needed)
        if best_accuracy < self.config['target_accuracy']:
            print(f"\n🎯 Stage 3: Advanced Fine-tuning")
            print("-" * 40)
            
            self.prepare_datasets('advanced')
            
            best_accuracy, model, _ = self.train_stage(
                model, stage_name="Advanced", epochs=40, lr=0.0001
            )
        
        return best_accuracy, model, history
    
    def train_stage(self, model, stage_name, epochs, lr=None):
        """Train a single stage"""
        
        if lr is None:
            lr = self.config['learning_rate']
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_accuracy = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(model, optimizer, criterion, epoch, stage_name)
            
            # Validation
            val_loss, val_acc, breed_accuracies, confusion = self.validate(model, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print results
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Show per-breed accuracy (top 5 worst performing)
            worst_breeds = sorted(breed_accuracies.items(), key=lambda x: x[1])[:5]
            print("Worst performing breeds:")
            for breed, acc in worst_breeds:
                status = "✅" if acc >= 80 else "⚠️" if acc >= 60 else "❌"
                print(f"  {status} {breed}: {acc:.1f}%")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                
                # Save model
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), f'models/dynamic_breed_model_best.pth')
                
                # Save detailed results
                results = {
                    'best_accuracy': best_accuracy,
                    'breed_accuracies': breed_accuracies,
                    'epoch': epoch,
                    'stage': stage_name,
                    'breeds': self.breeds,
                    'num_classes': self.num_classes,
                    'total_images': self.total_images,
                    'confusion_matrix': confusion.tolist()
                }
                
                with open('models/dynamic_breed_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"🎯 New best accuracy: {best_accuracy:.2f}%")
                
                # Check if target reached
                if val_acc >= self.config['target_accuracy']:
                    print(f"🏆 TARGET ACCURACY REACHED: {val_acc:.2f}%")
                    break
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        return best_accuracy, model, history
    
    def create_inference_system(self):
        """Create inference system for the trained model"""
        
        inference_code = f'''#!/usr/bin/env python3
"""
DYNAMIC BREED INFERENCE SYSTEM

Inference system for the dynamically trained multi-breed model.
Automatically handles {self.num_classes} breeds including Sahiwali.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import sys
import os

sys.path.append('.')
from utils.feature_extractor import FeatureExtractor, get_device

class DynamicBreedInference:
    def __init__(self, model_path='models/dynamic_breed_model_best.pth',
                 results_path='models/dynamic_breed_results.json'):
        
        self.device = get_device()
        self.load_model_info(results_path)
        self.load_model(model_path)
        self.setup_transforms()
        
        print(f"🎯 Dynamic Breed Inference Loaded")
        print(f"📱 Device: {{self.device}}")
        print(f"🧬 Breeds: {{len(self.breeds)}} ({{', '.join(self.breeds)}})")
        print(f"🎯 Training Accuracy: {{self.best_accuracy:.2f}}%")
    
    def load_model_info(self, results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        self.breeds = results['breeds']
        self.num_classes = results['num_classes']
        self.best_accuracy = results['best_accuracy']
        self.breed_accuracies = results['breed_accuracies']
        
        self.idx_to_breed = {{i: breed for i, breed in enumerate(self.breeds)}}
    
    def load_model(self, model_path):
        self.model = FeatureExtractor('resnet50', pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for i in range(top_k):
                breed = self.idx_to_breed[top_indices[i].item()]
                confidence = top_probs[i].item()
                training_acc = self.breed_accuracies.get(breed, 0)
                
                predictions.append({{
                    'breed': breed,
                    'confidence': confidence,
                    'training_accuracy': training_acc
                }})
            
            return {{
                'image_path': image_path,
                'predictions': predictions,
                'model_accuracy': self.best_accuracy
            }}
            
        except Exception as e:
            return {{'error': str(e), 'image_path': image_path}}

def main():
    if len(sys.argv) != 2:
        print("Usage: python infer_dynamic_breeds.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {{image_path}}")
        return
    
    # Load inference system
    inferencer = DynamicBreedInference()
    
    # Make prediction
    result = inferencer.predict(image_path)
    
    if 'error' in result:
        print(f"Error: {{result['error']}}")
        return
    
    # Display results
    print(f"\\n🎯 BREED PREDICTION RESULTS")
    print("=" * 50)
    print(f"📸 Image: {{os.path.basename(image_path)}}")
    print(f"🤖 Model Accuracy: {{result['model_accuracy']:.2f}}%")
    print()
    
    for i, pred in enumerate(result['predictions']):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        confidence_bar = "█" * int(pred['confidence'] * 20)
        print(f"{{rank}} {{pred['breed']}}")
        print(f"   Confidence: {{pred['confidence']:.3f}} {{confidence_bar}}")
        print(f"   Training Acc: {{pred['training_accuracy']:.1f}}%")
        print()

if __name__ == "__main__":
    main()
'''
        
        with open('infer_dynamic_breeds.py', 'w') as f:
            f.write(inference_code)
        
        print("✅ Inference system created: infer_dynamic_breeds.py")

def main():
    """Main training function"""
    
    print("🎯 DYNAMIC MULTI-BREED TRAINER WITH SAHIWALI")
    print("=" * 80)
    
    # Initialize trainer
    trainer = DynamicBreedTrainer()
    
    # Train the model
    best_accuracy, model, history = trainer.progressive_training()
    
    # Create inference system
    trainer.create_inference_system()
    
    # Final summary
    print(f"\n🏆 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ Best Accuracy: {best_accuracy:.2f}%")
    print(f"🧬 Breeds Trained: {trainer.num_classes}")
    print(f"📊 Total Images: {trainer.total_images}")
    print(f"💾 Model saved: models/dynamic_breed_model_best.pth")
    print(f"🚀 Inference: python infer_dynamic_breeds.py <image_path>")
    
    if best_accuracy >= trainer.config['target_accuracy']:
        print("🏆 TARGET ACCURACY ACHIEVED - Model ready for deployment!")
    else:
        print(f"⚠️  Consider adding more data or extended training")
    
    print("=" * 80)

if __name__ == "__main__":
    main()