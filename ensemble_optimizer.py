#!/usr/bin/env python3
"""
Advanced Ensemble Accuracy Optimizer

This script creates an advanced ensemble system that combines both 
trained model and smart fusion for near-perfect accuracy.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict

# Add utils to path
sys.path.append('.')
from utils import get_device, get_transforms
from utils.feature_extractor import FeatureExtractor
from utils.smart_fusion import SmartCNNFirstFusion
from smart_fusion_fixed import EnhancedSmartFusion

class AdvancedEnsembleSystem:
    """Advanced ensemble system combining trained model and smart fusion"""
    
    def __init__(self):
        self.device = get_device()
        self.transform = get_transforms(augment=False)
        
        # Load model info and trained model
        self.load_trained_model()
        self.load_smart_fusion_system()
        
        # Ensemble weights (will be optimized)
        self.ensemble_weights = {
            'trained_model_base': 0.6,
            'smart_fusion_base': 0.4,
            'confidence_boost': 0.2,
            'agreement_bonus': 0.3
        }
    
    def load_trained_model(self):
        """Load the enhanced 11-breed trained model"""
        # Load model info
        with open('models/enhanced_11breed_info.json', 'r') as f:
            self.model_info = json.load(f)
        
        # Load trained model
        self.trained_model = FeatureExtractor(model_name='resnet50', num_classes=11)
        self.trained_model.load_state_dict(
            torch.load('models/enhanced_11breed_model.pth', map_location=self.device)
        )
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()
        
        print(f"✅ Loaded trained model with {self.model_info['best_accuracy']:.1f}% accuracy")
    
    def load_smart_fusion_system(self):
        """Load enhanced smart fusion system with prototypes"""
        # Load prototypes
        with open('models/prototypes_maximum_11breed.pkl', 'rb') as f:
            self.prototype_data = pickle.load(f)
        
        # Load prototypes to device
        self.prototypes = {}
        for breed in self.model_info['breeds']:
            if breed in self.prototype_data['prototypes']:
                self.prototypes[breed] = self.prototype_data['prototypes'][breed].to(self.device)
        
        # Initialize enhanced smart fusion system
        self.smart_fusion = EnhancedSmartFusion(
            self.model_info['breeds'], 
            self.trained_model, 
            self.prototypes, 
            self.device
        )
        
        print("✅ Loaded enhanced smart fusion system with prototypes")
    
    def predict_trained_model(self, image):
        """Get predictions from trained model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.trained_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences = probabilities[0].cpu().numpy()
        
        # Create breed predictions
        predictions = {}
        for i, breed in enumerate(self.model_info['breeds']):
            predictions[breed] = float(confidences[i])
        
        # Get top prediction
        top_breed = max(predictions.items(), key=lambda x: x[1])
        confidence = top_breed[1]
        
        return {
            'predictions': predictions,
            'top_breed': top_breed[0],
            'confidence': confidence,
            'method': 'trained_model'
        }
    
    def predict_smart_fusion(self, image):
        """Get predictions from enhanced smart fusion"""
        try:
            # Use enhanced smart fusion system
            predictions, smart_prediction = self.smart_fusion.predict_with_enhanced_fusion(image)
            
            if smart_prediction:
                top_breed = max(predictions.items(), key=lambda x: x[1])
                
                return {
                    'predictions': predictions,
                    'top_breed': top_breed[0],
                    'confidence': smart_prediction.cnn_confidence,
                    'fusion_strategy': smart_prediction.fusion_strategy,
                    'margin': smart_prediction.margin,
                    'method': 'enhanced_smart_fusion'
                }
            else:
                # Fallback to trained model if smart fusion fails
                return self.predict_trained_model(image)
                
        except Exception as e:
            print(f"Smart fusion error: {e}")
            # Fallback to trained model
            return self.predict_trained_model(image)
    
    def advanced_ensemble_predict(self, image, true_breed=None):
        """Advanced ensemble prediction combining both methods"""
        
        # Get predictions from both methods
        trained_pred = self.predict_trained_model(image)
        fusion_pred = self.predict_smart_fusion(image)
        
        if not fusion_pred:
            return trained_pred
        
        # Extract key information
        trained_top = trained_pred['top_breed']
        trained_conf = trained_pred['confidence']
        fusion_top = fusion_pred['top_breed']
        fusion_conf = fusion_pred['confidence']
        
        # Check agreement
        agreement = (trained_top == fusion_top)
        
        # Ensemble scoring
        ensemble_scores = {}
        
        for breed in self.model_info['breeds']:
            # Base scores
            trained_score = trained_pred['predictions'].get(breed, 0.0)
            fusion_score = fusion_pred['predictions'].get(breed, 0.0)
            
            # Weighted combination
            base_score = (self.ensemble_weights['trained_model_base'] * trained_score + 
                         self.ensemble_weights['smart_fusion_base'] * fusion_score)
            
            # Confidence boost for high-confidence predictions
            if breed == trained_top and trained_conf > 0.8:
                base_score += self.ensemble_weights['confidence_boost'] * trained_conf
            
            if breed == fusion_top and fusion_conf > 0.8:
                base_score += self.ensemble_weights['confidence_boost'] * fusion_conf
            
            # Agreement bonus
            if agreement and breed == trained_top:
                base_score += self.ensemble_weights['agreement_bonus']
            
            ensemble_scores[breed] = base_score
        
        # Normalize scores
        total_score = sum(ensemble_scores.values())
        if total_score > 0:
            ensemble_scores = {k: v/total_score for k, v in ensemble_scores.items()}
        
        # Get final prediction
        final_breed = max(ensemble_scores.items(), key=lambda x: x[1])
        final_confidence = final_breed[1]
        
        # Determine confidence level
        if agreement and min(trained_conf, fusion_conf) > 0.85:
            confidence_level = "VERY_HIGH"
        elif agreement and min(trained_conf, fusion_conf) > 0.7:
            confidence_level = "HIGH"
        elif final_confidence > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        result = {
            'final_prediction': final_breed[0],
            'final_confidence': final_confidence,
            'confidence_level': confidence_level,
            'agreement': agreement,
            'trained_model': trained_pred,
            'smart_fusion': fusion_pred,
            'ensemble_scores': ensemble_scores,
            'method': 'advanced_ensemble'
        }
        
        # Add accuracy info if true breed provided
        if true_breed:
            result['correct'] = (final_breed[0] == true_breed)
            result['trained_correct'] = (trained_top == true_breed)
            result['fusion_correct'] = (fusion_top == true_breed)
        
        return result
    
    def analyze_dataset_accuracy(self, max_samples_per_breed=10):
        """Analyze accuracy across the dataset"""
        print("🔍 ANALYZING DATASET ACCURACY")
        print("=" * 50)
        
        results = []
        breed_accuracies = defaultdict(lambda: {'total': 0, 'trained_correct': 0, 
                                               'fusion_correct': 0, 'ensemble_correct': 0})
        
        dataset_path = Path('dataset')
        
        for breed_dir in dataset_path.iterdir():
            if not breed_dir.is_dir() or breed_dir.name not in self.model_info['breeds']:
                continue
            
            breed = breed_dir.name
            print(f"\n📊 Testing {breed}...")
            
            # Get image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(breed_dir.glob(ext)))
            
            # Limit samples for faster testing
            image_files = image_files[:max_samples_per_breed]
            
            for img_path in tqdm(image_files, desc=f"Analyzing {breed}"):
                try:
                    # Load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get ensemble prediction
                    result = self.advanced_ensemble_predict(image_rgb, true_breed=breed)
                    
                    # Store results
                    result['true_breed'] = breed
                    result['image_path'] = str(img_path)
                    results.append(result)
                    
                    # Update breed accuracies
                    breed_accuracies[breed]['total'] += 1
                    if result.get('trained_correct', False):
                        breed_accuracies[breed]['trained_correct'] += 1
                    if result.get('fusion_correct', False):
                        breed_accuracies[breed]['fusion_correct'] += 1
                    if result.get('correct', False):
                        breed_accuracies[breed]['ensemble_correct'] += 1
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return results, breed_accuracies
    
    def optimize_ensemble_weights(self, results):
        """Optimize ensemble weights based on results"""
        print("\n🔧 OPTIMIZING ENSEMBLE WEIGHTS")
        print("=" * 40)
        
        best_accuracy = 0
        best_weights = self.ensemble_weights.copy()
        
        # Grid search for optimal weights
        trained_weights = [0.4, 0.5, 0.6, 0.7, 0.8]
        confidence_boosts = [0.1, 0.15, 0.2, 0.25, 0.3]
        agreement_bonuses = [0.2, 0.25, 0.3, 0.35, 0.4]
        
        for tw in trained_weights:
            for cb in confidence_boosts:
                for ab in agreement_bonuses:
                    test_weights = {
                        'trained_model_base': tw,
                        'smart_fusion_base': 1 - tw,
                        'confidence_boost': cb,
                        'agreement_bonus': ab
                    }
                    
                    # Test these weights
                    correct = 0
                    total = 0
                    
                    for result in results:
                        # Recalculate ensemble score with new weights
                        trained_pred = result['trained_model']
                        fusion_pred = result['smart_fusion']
                        true_breed = result['true_breed']
                        
                        if not fusion_pred:
                            continue
                        
                        ensemble_scores = {}
                        for breed in self.model_info['breeds']:
                            trained_score = trained_pred['predictions'].get(breed, 0.0)
                            fusion_score = fusion_pred['predictions'].get(breed, 0.0)
                            
                            base_score = (test_weights['trained_model_base'] * trained_score + 
                                         test_weights['smart_fusion_base'] * fusion_score)
                            
                            # Add bonuses
                            if breed == trained_pred['top_breed'] and trained_pred['confidence'] > 0.8:
                                base_score += test_weights['confidence_boost'] * trained_pred['confidence']
                            
                            if breed == fusion_pred['top_breed'] and fusion_pred['confidence'] > 0.8:
                                base_score += test_weights['confidence_boost'] * fusion_pred['confidence']
                            
                            if (trained_pred['top_breed'] == fusion_pred['top_breed'] and 
                                breed == trained_pred['top_breed']):
                                base_score += test_weights['agreement_bonus']
                            
                            ensemble_scores[breed] = base_score
                        
                        # Normalize and get prediction
                        total_score = sum(ensemble_scores.values())
                        if total_score > 0:
                            ensemble_scores = {k: v/total_score for k, v in ensemble_scores.items()}
                        
                        predicted_breed = max(ensemble_scores.items(), key=lambda x: x[1])[0]
                        
                        if predicted_breed == true_breed:
                            correct += 1
                        total += 1
                    
                    accuracy = correct / total if total > 0 else 0
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = test_weights.copy()
        
        self.ensemble_weights = best_weights
        print(f"✅ Optimized weights - Best accuracy: {best_accuracy:.3f}")
        print(f"📊 Optimal weights: {best_weights}")
        
        return best_accuracy, best_weights
    
    def generate_accuracy_report(self, results, breed_accuracies):
        """Generate comprehensive accuracy report"""
        print("\n📈 COMPREHENSIVE ACCURACY REPORT")
        print("=" * 60)
        
        # Overall statistics
        total_samples = len(results)
        trained_correct = sum(1 for r in results if r.get('trained_correct', False))
        fusion_correct = sum(1 for r in results if r.get('fusion_correct', False))
        ensemble_correct = sum(1 for r in results if r.get('correct', False))
        agreement_cases = sum(1 for r in results if r.get('agreement', False))
        
        print(f"\n🎯 OVERALL PERFORMANCE (Total samples: {total_samples})")
        print(f"Trained Model Accuracy: {trained_correct/total_samples:.1%}")
        print(f"Smart Fusion Accuracy: {fusion_correct/total_samples:.1%}")
        print(f"Ensemble Accuracy: {ensemble_correct/total_samples:.1%}")
        print(f"Method Agreement: {agreement_cases/total_samples:.1%}")
        
        # Per-breed analysis
        print(f"\n🐄 PER-BREED ACCURACY")
        print("-" * 60)
        print(f"{'Breed':<12} {'Trained':<10} {'Fusion':<10} {'Ensemble':<10} {'Samples':<8}")
        print("-" * 60)
        
        for breed in sorted(breed_accuracies.keys()):
            stats = breed_accuracies[breed]
            if stats['total'] > 0:
                trained_acc = stats['trained_correct'] / stats['total']
                fusion_acc = stats['fusion_correct'] / stats['total']
                ensemble_acc = stats['ensemble_correct'] / stats['total']
                
                print(f"{breed:<12} {trained_acc:<10.1%} {fusion_acc:<10.1%} "
                      f"{ensemble_acc:<10.1%} {stats['total']:<8}")
        
        # Confidence level analysis
        confidence_analysis = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in results:
            conf_level = result.get('confidence_level', 'UNKNOWN')
            confidence_analysis[conf_level]['total'] += 1
            if result.get('correct', False):
                confidence_analysis[conf_level]['correct'] += 1
        
        print(f"\n🎚️ CONFIDENCE LEVEL ANALYSIS")
        print("-" * 40)
        for level in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']:
            if level in confidence_analysis:
                stats = confidence_analysis[level]
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total']
                    print(f"{level}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
        
        # Agreement analysis
        agreement_correct = sum(1 for r in results if r.get('agreement', False) and r.get('correct', False))
        disagreement_cases = [r for r in results if not r.get('agreement', False)]
        
        print(f"\n🤝 AGREEMENT ANALYSIS")
        print(f"When methods agree: {agreement_correct}/{agreement_cases} correct ({agreement_correct/agreement_cases:.1%})")
        print(f"Disagreement cases: {len(disagreement_cases)}")
        
        return {
            'overall_accuracy': ensemble_correct / total_samples,
            'trained_accuracy': trained_correct / total_samples,
            'fusion_accuracy': fusion_correct / total_samples,
            'agreement_rate': agreement_cases / total_samples,
            'breed_accuracies': dict(breed_accuracies),
            'confidence_analysis': dict(confidence_analysis)
        }

def main():
    """Main function to run accuracy analysis and optimization"""
    print("🚀 ADVANCED ENSEMBLE ACCURACY OPTIMIZATION")
    print("=" * 60)
    
    # Initialize system
    ensemble_system = AdvancedEnsembleSystem()
    
    # Analyze dataset accuracy
    results, breed_accuracies = ensemble_system.analyze_dataset_accuracy(max_samples_per_breed=12)
    
    # Optimize ensemble weights
    best_accuracy, best_weights = ensemble_system.optimize_ensemble_weights(results)
    
    # Generate comprehensive report
    report = ensemble_system.generate_accuracy_report(results, breed_accuracies)
    
    # Save results
    output_data = {
        'ensemble_weights': best_weights,
        'accuracy_report': report,
        'detailed_results': results[:50]  # Save sample results
    }
    
    with open('models/ensemble_optimization_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: models/ensemble_optimization_results.json")
    print(f"🎯 Final Ensemble Accuracy: {report['overall_accuracy']:.1%}")
    print(f"🔧 Optimized Weights: {best_weights}")

if __name__ == "__main__":
    main()