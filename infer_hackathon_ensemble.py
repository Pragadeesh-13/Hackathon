#!/usr/bin/env python3
"""
HACKATHON ENSEMBLE INFERENCE SYSTEM

Competition-grade inference using ensemble models for maximum accuracy
with confidence optimization and real-time performance metrics.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from PIL import Image
import cv2
from collections import defaultdict
import time

# Add utils to path
sys.path.append('.')
from utils import FeatureExtractor, get_device

class HackathonEnsembleInference:
    """Competition-grade ensemble inference system"""
    
    def __init__(self, ensemble_info_path='models/hackathon_ensemble_info.json'):
        self.device = get_device()
        self.load_ensemble_info(ensemble_info_path)
        self.load_ensemble_models()
        self.setup_transforms()
        
        print(f"🏆 HACKATHON ENSEMBLE INFERENCE LOADED")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Ensemble Size: {self.num_models} models")
        print(f"🧬 Breeds: {len(self.breeds)} ({', '.join(self.breeds)})")
        print(f"🎯 Best Training Accuracy: {self.best_accuracy:.2f}%")
    
    def load_ensemble_info(self, info_path):
        """Load ensemble configuration"""
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Ensemble info not found: {info_path}")
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        self.num_models = info['num_models']
        self.breeds = info['breeds']
        self.num_classes = info['num_classes']
        self.individual_accuracies = info['individual_accuracies']
        self.average_accuracy = info['average_accuracy']
        self.best_accuracy = info['best_accuracy']
        
        # Create breed to index mapping
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for idx, breed in enumerate(self.breeds)}
    
    def load_ensemble_models(self):
        """Load all ensemble models"""
        self.models = []
        
        for model_idx in range(self.num_models):
            model_path = f'models/hackathon_ensemble_{model_idx}_best.pth'
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model {model_idx} not found: {model_path}")
                continue
            
            # Create model architecture
            model = FeatureExtractor('resnet50', pretrained=False)
            
            # Replace classifier to match training
            if hasattr(model.model, 'fc'):
                model.model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(model.model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, self.num_classes)
                )
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            print(f"✅ Loaded model {model_idx + 1}: Accuracy {self.individual_accuracies[model_idx]:.2f}%")
        
        if not self.models:
            raise RuntimeError("No ensemble models found!")
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_input):
        """Preprocess image for inference"""
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert from numpy array (OpenCV format)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Convert BGR to RGB
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # Already PIL Image
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_single_model(self, model, image_tensor):
        """Get prediction from single model"""
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
    
    def ensemble_predict(self, image_input, voting_method='weighted_average'):
        """Predict using ensemble of models"""
        start_time = time.time()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_input)
        
        # Get predictions from all models
        all_predictions = []
        model_weights = []
        
        for model_idx, model in enumerate(self.models):
            probabilities = self.predict_single_model(model, image_tensor)
            all_predictions.append(probabilities)
            
            # Weight by individual model accuracy
            weight = self.individual_accuracies[model_idx] / 100.0
            model_weights.append(weight)
        
        all_predictions = np.array(all_predictions)
        model_weights = np.array(model_weights)
        
        # Ensemble voting
        if voting_method == 'simple_average':
            ensemble_probs = np.mean(all_predictions, axis=0)
        elif voting_method == 'weighted_average':
            # Weight by model accuracy
            weighted_preds = all_predictions * model_weights.reshape(-1, 1)
            ensemble_probs = np.sum(weighted_preds, axis=0) / np.sum(model_weights)
        elif voting_method == 'max_confidence':
            # Use prediction from most confident model
            max_confidences = np.max(all_predictions, axis=1)
            best_model_idx = np.argmax(max_confidences)
            ensemble_probs = all_predictions[best_model_idx]
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        # Get top predictions
        top_indices = np.argsort(ensemble_probs)[::-1]
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Prepare results
        results = {
            'predicted_breed': self.idx_to_breed[top_indices[0]],
            'confidence': float(ensemble_probs[top_indices[0]]),
            'top_3_predictions': [
                {
                    'breed': self.idx_to_breed[idx],
                    'confidence': float(ensemble_probs[idx])
                }
                for idx in top_indices[:3]
            ],
            'all_predictions': {
                self.idx_to_breed[i]: float(prob) 
                for i, prob in enumerate(ensemble_probs)
            },
            'ensemble_details': {
                'voting_method': voting_method,
                'num_models': len(self.models),
                'model_predictions': [
                    {
                        'model_idx': i,
                        'accuracy': self.individual_accuracies[i],
                        'prediction': self.idx_to_breed[np.argmax(pred)],
                        'confidence': float(np.max(pred))
                    }
                    for i, pred in enumerate(all_predictions)
                ],
                'inference_time_ms': inference_time * 1000
            }
        }
        
        return results
    
    def batch_predict(self, image_paths, voting_method='weighted_average'):
        """Predict multiple images"""
        results = []
        
        print(f"🔄 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.ensemble_predict(image_path, voting_method)
                result['image_path'] = image_path
                results.append(result)
                
                print(f"✅ {i+1}/{len(image_paths)}: {result['predicted_breed']} ({result['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_accuracy(self, test_images, ground_truth, voting_method='weighted_average'):
        """Evaluate ensemble accuracy on test set"""
        
        print(f"🧪 Evaluating ensemble accuracy on {len(test_images)} test images...")
        
        correct_predictions = 0
        total_predictions = 0
        breed_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        confidence_scores = []
        
        for i, (image_path, true_breed) in enumerate(zip(test_images, ground_truth)):
            try:
                result = self.ensemble_predict(image_path, voting_method)
                predicted_breed = result['predicted_breed']
                confidence = result['confidence']
                
                # Update statistics
                total_predictions += 1
                confidence_scores.append(confidence)
                breed_accuracy[true_breed]['total'] += 1
                
                if predicted_breed == true_breed:
                    correct_predictions += 1
                    breed_accuracy[true_breed]['correct'] += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    current_acc = correct_predictions / total_predictions * 100
                    print(f"📊 Progress: {i+1}/{len(test_images)} | Current Accuracy: {current_acc:.2f}%")
                
            except Exception as e:
                print(f"❌ Error evaluating {image_path}: {e}")
                continue
        
        # Calculate final metrics
        overall_accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        average_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Per-breed accuracy
        breed_accuracies = {}
        for breed, stats in breed_accuracy.items():
            if stats['total'] > 0:
                breed_accuracies[breed] = stats['correct'] / stats['total'] * 100
            else:
                breed_accuracies[breed] = 0
        
        # Evaluation summary
        evaluation_results = {
            'overall_accuracy': overall_accuracy,
            'average_confidence': average_confidence,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'breed_accuracies': breed_accuracies,
            'voting_method': voting_method,
            'ensemble_size': len(self.models)
        }
        
        print(f"\n🎯 ENSEMBLE EVALUATION RESULTS")
        print("=" * 50)
        print(f"✅ Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"🎯 Average Confidence: {average_confidence:.3f}")
        print(f"🤖 Ensemble Size: {len(self.models)} models")
        print(f"📊 Voting Method: {voting_method}")
        print(f"\n📈 Per-Breed Accuracy:")
        for breed, acc in sorted(breed_accuracies.items()):
            print(f"  {breed}: {acc:.2f}%")
        
        return evaluation_results

def main():
    """Test hackathon ensemble inference"""
    
    try:
        # Initialize ensemble inference
        ensemble = HackathonEnsembleInference()
        
        # Test with sample images
        print("\n🧪 Testing ensemble inference...")
        
        # Find some test images
        test_images = []
        for breed in ensemble.breeds[:3]:  # Test first 3 breeds
            breed_dir = Path(breed)
            if breed_dir.exists():
                breed_images = list(breed_dir.glob('*.jpg'))[:2]  # 2 images per breed
                test_images.extend([(str(img), breed) for img in breed_images])
        
        if test_images:
            # Test different voting methods
            voting_methods = ['weighted_average', 'simple_average', 'max_confidence']
            
            for method in voting_methods:
                print(f"\n🗳️  Testing voting method: {method}")
                print("-" * 40)
                
                for image_path, true_breed in test_images[:3]:
                    result = ensemble.ensemble_predict(image_path, method)
                    predicted = result['predicted_breed']
                    confidence = result['confidence']
                    correct = "✅" if predicted == true_breed else "❌"
                    
                    print(f"{correct} True: {true_breed} | Predicted: {predicted} | Confidence: {confidence:.3f}")
        
        print(f"\n🏆 Hackathon ensemble system ready!")
        print(f"📁 Models loaded from: models/hackathon_ensemble_*.pth")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please run train_hackathon_pipeline.py first")

if __name__ == "__main__":
    main()