#!/usr/bin/env python3
"""
Advanced Accuracy Analysis & Ensemble System

This script analyzes both trained model and smart fusion performance,
identifies accuracy patterns, and creates an optimized ensemble approach.
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
import pandas as pd
from collections import defaultdict

# Add utils to path
sys.path.append('.')
from utils import get_device, get_transforms
from utils.feature_extractor import FeatureExtractor
from utils.smart_fusion import SmartCNNFirstFusion

class AdvancedEnsembleSystem:
    """Advanced ensemble system combining trained model and smart fusion"""
    
    def __init__(self):
        self.device = get_device()
        self.transform = get_transforms(augment=False)
        
        # Load model info and trained model
        self.load_trained_model()
        self.load_smart_fusion_system()
        
        # Analysis results
        self.accuracy_analysis = {
            'trained_model': defaultdict(list),
            'smart_fusion': defaultdict(list),
            'agreement_cases': [],
            'disagreement_cases': []
        }
        
        # Ensemble weights (will be optimized)
        self.ensemble_weights = {
            'trained_model_base': 0.6,
            'smart_fusion_base': 0.4,
            'confidence_boost': 0.2,
            'agreement_bonus': 0.3
        }
        
        # Load prototypes to device
        self.prototypes = {}
        for breed in self.model_info['breeds']:
            if breed in self.prototype_data['prototypes']:
                self.prototypes[breed] = self.prototype_data['prototypes'][breed].to(self.device)
        
        # Initialize smart fusion system
        self.fusion_system = SmartCNNFirstFusion(self.model_info['breeds'])
        
        # Create feature model for smart fusion
        from utils import FeatureExtractor as BasicFeatureExtractor
        self.feature_model = BasicFeatureExtractor(model_name='resnet50', pretrained=True)
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()
        
        print("✅ Models loaded successfully")
    
    def load_test_samples(self):
        """Load test samples from each breed"""
        print("📂 Loading test samples...")
        
        self.test_samples = {}
        dataset_path = 'dataset'
        
        for breed in self.model_info['breeds']:
            breed_path = os.path.join(dataset_path, breed)
            if os.path.exists(breed_path):
                # Get all images for this breed
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images.extend(list(Path(breed_path).glob(ext)))
                
                # Take up to 5 samples per breed for analysis
                self.test_samples[breed] = [str(img) for img in images[:5]]
        
        total_samples = sum(len(samples) for samples in self.test_samples.values())
        print(f"✅ Loaded {total_samples} test samples from {len(self.test_samples)} breeds")
    
    def predict_with_trained_model(self, image_path):
        """Get prediction from trained model"""
        try:
            transform = get_transforms(augment=False)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.trained_model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences = probabilities[0].cpu().numpy()
            
            # Get top prediction
            top_idx = np.argmax(confidences)
            top_breed = self.model_info['breeds'][top_idx]
            top_confidence = float(confidences[top_idx])
            
            return {
                'breed': top_breed,
                'confidence': top_confidence,
                'all_scores': {self.model_info['breeds'][i]: float(confidences[i]) for i in range(len(confidences))}
            }
        except Exception as e:
            print(f"❌ Trained model error for {image_path}: {e}")
            return None
    
    def predict_with_smart_fusion(self, image_path):
        """Get prediction from smart fusion"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            predictions, smart_prediction = self.fusion_system.predict_with_smart_fusion(
                image, self.feature_model, self.prototypes, self.device
            )
            
            if smart_prediction and smart_prediction.breed_scores:
                top_breed = max(smart_prediction.breed_scores.items(), key=lambda x: x[1])
                
                return {
                    'breed': top_breed[0],
                    'confidence': top_breed[1],
                    'strategy': smart_prediction.fusion_strategy,
                    'margin': smart_prediction.margin,
                    'all_scores': smart_prediction.breed_scores
                }
            else:
                return None
        except Exception as e:
            print(f"❌ Smart fusion error for {image_path}: {e}")
            return None
    
    def analyze_sample(self, image_path, true_breed):
        """Analyze a single sample with both methods"""
        trained_result = self.predict_with_trained_model(image_path)
        fusion_result = self.predict_with_smart_fusion(image_path)
        
        analysis = {
            'image_path': image_path,
            'true_breed': true_breed,
            'trained_model': trained_result,
            'smart_fusion': fusion_result,
            'agreement': None,
            'confidence_gap': None,
            'recommendation': 'unknown'
        }
        
        if trained_result and fusion_result:
            # Check if methods agree
            analysis['agreement'] = trained_result['breed'] == fusion_result['breed']
            
            # Calculate confidence gap
            analysis['confidence_gap'] = abs(trained_result['confidence'] - fusion_result['confidence'])
            
            # Determine recommendation
            trained_correct = trained_result['breed'] == true_breed
            fusion_correct = fusion_result['breed'] == true_breed
            
            if trained_correct and fusion_correct:
                analysis['recommendation'] = 'both_correct'
            elif trained_correct and not fusion_correct:
                analysis['recommendation'] = 'trained_better'
            elif fusion_correct and not trained_correct:
                analysis['recommendation'] = 'fusion_better'
            else:
                analysis['recommendation'] = 'both_wrong'
        
        return analysis
    
    def run_comprehensive_analysis(self):
        """Run analysis on all test samples"""
        print("🔬 Running comprehensive accuracy analysis...")
        
        all_results = []
        breed_stats = {}
        
        for breed, samples in self.test_samples.items():
            print(f"\n📊 Analyzing {breed} ({len(samples)} samples)...")
            breed_results = []
            
            for sample_path in tqdm(samples, desc=f"Processing {breed}"):
                result = self.analyze_sample(sample_path, breed)
                all_results.append(result)
                breed_results.append(result)
            
            # Calculate breed-specific stats
            breed_stats[breed] = self.calculate_breed_stats(breed_results)
        
        # Calculate overall stats
        overall_stats = self.calculate_overall_stats(all_results)
        
        return {
            'all_results': all_results,
            'breed_stats': breed_stats,
            'overall_stats': overall_stats
        }
    
    def calculate_breed_stats(self, breed_results):
        """Calculate statistics for a specific breed"""
        total = len(breed_results)
        if total == 0:
            return {}
        
        trained_correct = sum(1 for r in breed_results if r['trained_model'] and r['trained_model']['breed'] == r['true_breed'])
        fusion_correct = sum(1 for r in breed_results if r['smart_fusion'] and r['smart_fusion']['breed'] == r['true_breed'])
        agreement = sum(1 for r in breed_results if r['agreement'])
        both_correct = sum(1 for r in breed_results if r['recommendation'] == 'both_correct')
        
        return {
            'total_samples': total,
            'trained_accuracy': trained_correct / total * 100,
            'fusion_accuracy': fusion_correct / total * 100,
            'agreement_rate': agreement / total * 100,
            'both_correct_rate': both_correct / total * 100
        }
    
    def calculate_overall_stats(self, all_results):
        """Calculate overall statistics"""
        total = len(all_results)
        if total == 0:
            return {}
        
        trained_correct = sum(1 for r in all_results if r['trained_model'] and r['trained_model']['breed'] == r['true_breed'])
        fusion_correct = sum(1 for r in all_results if r['smart_fusion'] and r['smart_fusion']['breed'] == r['true_breed'])
        agreement = sum(1 for r in all_results if r['agreement'])
        both_correct = sum(1 for r in all_results if r['recommendation'] == 'both_correct')
        
        # Find best strategy per scenario
        recommendations = {}
        for r in all_results:
            rec = r['recommendation']
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        return {
            'total_samples': total,
            'trained_accuracy': trained_correct / total * 100,
            'fusion_accuracy': fusion_correct / total * 100,
            'agreement_rate': agreement / total * 100,
            'both_correct_rate': both_correct / total * 100,
            'recommendations': recommendations
        }
    
    def print_analysis_report(self, analysis_results):
        """Print comprehensive analysis report"""
        print("\n🎯 COMPREHENSIVE ACCURACY ANALYSIS REPORT")
        print("=" * 80)
        
        overall = analysis_results['overall_stats']
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"  • Total Samples: {overall['total_samples']}")
        print(f"  • Trained Model Accuracy: {overall['trained_accuracy']:.1f}%")
        print(f"  • Smart Fusion Accuracy: {overall['fusion_accuracy']:.1f}%")
        print(f"  • Agreement Rate: {overall['agreement_rate']:.1f}%")
        print(f"  • Both Correct Rate: {overall['both_correct_rate']:.1f}%")
        
        print(f"\n📈 RECOMMENDATION BREAKDOWN:")
        for rec, count in overall['recommendations'].items():
            percentage = count / overall['total_samples'] * 100
            print(f"  • {rec.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n🐄 BREED-SPECIFIC PERFORMANCE:")
        for breed, stats in analysis_results['breed_stats'].items():
            print(f"\n  {breed}:")
            print(f"    Trained: {stats['trained_accuracy']:.1f}% | Fusion: {stats['fusion_accuracy']:.1f}% | Agreement: {stats['agreement_rate']:.1f}%")
        
        # Identify problematic breeds
        problematic_breeds = []
        excellent_breeds = []
        
        for breed, stats in analysis_results['breed_stats'].items():
            max_accuracy = max(stats['trained_accuracy'], stats['fusion_accuracy'])
            if max_accuracy < 70:
                problematic_breeds.append((breed, max_accuracy))
            elif max_accuracy > 90:
                excellent_breeds.append((breed, max_accuracy))
        
        if problematic_breeds:
            print(f"\n⚠️  BREEDS NEEDING IMPROVEMENT:")
            for breed, acc in sorted(problematic_breeds, key=lambda x: x[1]):
                print(f"    • {breed}: {acc:.1f}% best accuracy")
        
        if excellent_breeds:
            print(f"\n✅ EXCELLENT PERFORMING BREEDS:")
            for breed, acc in sorted(excellent_breeds, key=lambda x: x[1], reverse=True):
                print(f"    • {breed}: {acc:.1f}% best accuracy")

def main():
    analyzer = AccuracyAnalyzer()
    analysis_results = analyzer.run_comprehensive_analysis()
    analyzer.print_analysis_report(analysis_results)
    
    # Save results for ensemble development
    with open('models/accuracy_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(analysis_results), f, indent=2)
    
    print(f"\n💾 Analysis results saved to: models/accuracy_analysis_results.json")

if __name__ == "__main__":
    main()