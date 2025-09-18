#!/usr/bin/env python3
"""
Fixed Smart Fusion Implementation

This implementation properly combines the trained CNN model features 
with morphological analysis for enhanced breed recognition accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

from utils import FeatureExtractor, load_and_preprocess_image, get_transforms
from utils.morphology import MorphologicalFeatureExtractor, MorphologicalFeatures

@dataclass
class EnhancedFusedPrediction:
    """Container for enhanced fused prediction results"""
    breed_scores: Dict[str, float]
    fusion_strategy: str
    confidence_level: str
    explanation: str
    margin: float
    cnn_confidence: float
    morpho_confidence: float

class EnhancedSmartFusion:
    """
    Enhanced smart fusion system using the actual trained model
    """
    
    def __init__(self, breeds: List[str], trained_model, prototypes, device):
        self.breeds = breeds
        self.trained_model = trained_model
        self.prototypes = prototypes
        self.device = device
        self.morphology_extractor = MorphologicalFeatureExtractor()
        
        # Fusion thresholds
        self.high_confidence_threshold = 0.85
        self.clear_margin_threshold = 0.05
        self.tie_break_threshold = 0.01
        
    def extract_cnn_features(self, image):
        """Extract features using the trained model backbone"""
        try:
            # Preprocess image
            if isinstance(image, str):
                image_tensor = load_and_preprocess_image(image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL then to tensor
                image_pil = Image.fromarray(image)
                transform = get_transforms(augment=False)
                image_tensor = transform(image_pil)
            else:
                # Assume PIL image
                transform = get_transforms(augment=False)
                image_tensor = transform(image)
            
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            # Extract features from trained model backbone
            with torch.no_grad():
                # For ResNet, extract features before final FC layer
                x = image_batch
                for name, module in self.trained_model.model.named_children():
                    if name == 'fc':  # Stop before final classifier
                        break
                    x = module(x)
                
                # Global average pooling
                features = F.adaptive_avg_pool2d(x, (1, 1))
                features = torch.flatten(features, 1)
                
                # Normalize features
                features_normalized = F.normalize(features, p=2, dim=1)
                query_embedding = features_normalized.squeeze(0)
            
            return query_embedding
            
        except Exception as e:
            print(f"CNN feature extraction error: {e}")
            return None
    
    def compute_cnn_similarities(self, query_embedding):
        """Compute similarities with breed prototypes"""
        from utils import compute_similarities
        return compute_similarities(query_embedding, self.prototypes)
    
    def extract_morphological_features(self, image):
        """Extract morphological features with error handling"""
        try:
            # Convert to proper format for morphology extractor
            if isinstance(image, str):
                image_cv = cv2.imread(image)
                if image_cv is None:
                    return None
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = image
                else:
                    return None
            else:
                # Convert PIL to numpy
                image_rgb = np.array(image)
            
            return self.morphology_extractor.extract_all_features(image_rgb)
            
        except Exception as e:
            print(f"Morphology extraction error: {e}")
            return None
    
    def smart_fuse_predictions(self, cnn_similarities, morph_features, image):
        """Smart fusion combining CNN and morphological features"""
        
        # Normalize CNN similarities
        cnn_scores = self.normalize_scores(cnn_similarities)
        sorted_cnn = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_cnn:
            return None
        
        best_cnn_breed, best_cnn_score = sorted_cnn[0]
        second_cnn_score = sorted_cnn[1][1] if len(sorted_cnn) > 1 else 0.0
        cnn_margin = best_cnn_score - second_cnn_score
        
        # Compute morphological confidence
        total_morph_confidence = 0.0
        if morph_features and morph_features.confidence_scores:
            total_morph_confidence = sum(morph_features.confidence_scores.values())
        
        print(f"🧠 CNN Top: {best_cnn_breed} ({best_cnn_score:.6f})")
        print(f"📏 CNN Margin: {cnn_margin:.6f}")
        print(f"🔬 Morph Confidence: {total_morph_confidence:.3f}")
        
        # STRATEGY 1: High CNN confidence - Use CNN prediction
        if best_cnn_score >= self.high_confidence_threshold:
            print("✅ Strategy: CNN_DOMINANT (High confidence)")
            fusion_strategy = "cnn_dominant"
            final_scores = dict(sorted_cnn)
            explanation = f"High CNN confidence ({best_cnn_score:.4f}) - Using CNN prediction"
        
        # STRATEGY 2: Clear CNN margin - Use CNN prediction
        elif cnn_margin >= self.clear_margin_threshold:
            print("✅ Strategy: CNN_DOMINANT (Clear margin)")
            fusion_strategy = "cnn_dominant"
            final_scores = dict(sorted_cnn)
            explanation = f"Clear CNN margin ({cnn_margin:.4f}) - Using CNN decision"
        
        # STRATEGY 3: Small CNN margin and good morphology - Use morphology for tiebreak
        elif cnn_margin <= self.tie_break_threshold and total_morph_confidence > 1.0 and morph_features:
            print("🔧 Strategy: MORPHO_TIEBREAK (Breaking CNN tie)")
            fusion_strategy = "morpho_tiebreak"
            
            # Get morphological scores
            morph_scores = self.compute_morphological_scores(morph_features)
            morph_scores_normalized = self.normalize_scores(morph_scores)
            
            # Apply morphology boost to top CNN candidates
            top_cnn_candidates = sorted_cnn[:3]
            enhanced_scores = {}
            
            for breed, cnn_score in top_cnn_candidates:
                breed_lower = breed.lower()
                morph_boost = morph_scores_normalized.get(breed_lower, 0.0) * 0.1
                enhanced_scores[breed] = cnn_score + morph_boost
            
            # Add remaining breeds
            for breed, cnn_score in sorted_cnn[3:]:
                enhanced_scores[breed] = cnn_score
            
            final_scores = dict(sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True))
            explanation = f"CNN tie ({cnn_margin:.4f}) - Morphology tiebreaker applied"
        
        # STRATEGY 4: Medium confidence - Light fusion
        else:
            print("⚖️ Strategy: LIGHT_FUSION (Balanced approach)")
            fusion_strategy = "light_fusion"
            
            if morph_features:
                morph_scores = self.compute_morphological_scores(morph_features)
                morph_scores_normalized = self.normalize_scores(morph_scores)
                
                # Light fusion: 80% CNN, 20% morphology
                fused_scores = {}
                for breed in self.breeds:
                    breed_lower = breed.lower()
                    cnn_score = cnn_scores.get(breed, 0.0)
                    morph_score = morph_scores_normalized.get(breed_lower, 0.0)
                    fused_score = 0.8 * cnn_score + 0.2 * morph_score
                    fused_scores[breed] = fused_score
                
                final_scores = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
            else:
                final_scores = dict(sorted_cnn)
            
            explanation = "Light fusion (80% CNN, 20% morphology) for balanced accuracy"
        
        # Calculate final metrics
        final_sorted = list(final_scores.items())
        best_breed, best_score = final_sorted[0]
        second_score = final_sorted[1][1] if len(final_sorted) > 1 else 0.0
        final_margin = best_score - second_score
        
        # Determine confidence level
        if fusion_strategy == "cnn_dominant" and best_cnn_score >= 0.90:
            confidence_level = 'exact'
        elif fusion_strategy == "cnn_dominant" and best_cnn_score >= 0.85:
            confidence_level = 'high'
        elif best_score >= 0.75 or best_cnn_score >= 0.80:
            confidence_level = 'medium'
        elif best_score >= 0.65 or best_cnn_score >= 0.70:
            confidence_level = 'low'
        else:
            confidence_level = 'uncertain'
        
        return EnhancedFusedPrediction(
            breed_scores=final_scores,
            fusion_strategy=fusion_strategy,
            confidence_level=confidence_level,
            explanation=explanation,
            margin=final_margin,
            cnn_confidence=best_cnn_score,
            morpho_confidence=total_morph_confidence
        )
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to sum to 1"""
        if not scores:
            return {}
        
        total = sum(scores.values())
        if total <= 0:
            return {k: 1.0/len(scores) for k in scores.keys()}
        
        return {k: v/total for k, v in scores.items()}
    
    def compute_morphological_scores(self, morph_features: MorphologicalFeatures) -> Dict[str, float]:
        """Compute morphological similarity scores"""
        if not morph_features or not morph_features.confidence_scores:
            return {}
        
        # Map morphology confidence to breed scores
        morpho_scores = {}
        for breed_lower, confidence in morph_features.confidence_scores.items():
            # Find matching breed in our list
            for breed in self.breeds:
                if breed.lower() == breed_lower:
                    morpho_scores[breed_lower] = confidence
                    break
        
        return morpho_scores
    
    def predict_with_enhanced_fusion(self, image) -> Tuple[Dict[str, float], EnhancedFusedPrediction]:
        """
        Main prediction function with enhanced smart fusion
        """
        
        # Step 1: Extract CNN features
        query_embedding = self.extract_cnn_features(image)
        if query_embedding is None:
            return {}, None
        
        # Step 2: Compute CNN similarities
        cnn_similarities = self.compute_cnn_similarities(query_embedding)
        
        # Step 3: Extract morphological features
        morph_features = self.extract_morphological_features(image)
        
        # Step 4: Smart fusion
        smart_prediction = self.smart_fuse_predictions(cnn_similarities, morph_features, image)
        
        if smart_prediction:
            return smart_prediction.breed_scores, smart_prediction
        else:
            return {}, None


def test_enhanced_fusion():
    """Test the enhanced fusion system"""
    print("🧪 Testing Enhanced Smart Fusion System")
    print("=" * 50)
    
    # Load model and prototypes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model info
    import json
    with open('models/enhanced_11breed_info.json', 'r') as f:
        model_info = json.load(f)
    
    # Load trained model
    from utils.feature_extractor import FeatureExtractor as EnhancedFeatureExtractor
    trained_model = EnhancedFeatureExtractor(model_name='resnet50', num_classes=11)
    trained_model.load_state_dict(
        torch.load('models/enhanced_11breed_model.pth', map_location=device)
    )
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    # Load prototypes
    with open('models/prototypes_maximum_11breed.pkl', 'rb') as f:
        prototype_data = pickle.load(f)
    
    prototypes = {}
    for breed in model_info['breeds']:
        if breed in prototype_data['prototypes']:
            prototypes[breed] = prototype_data['prototypes'][breed].to(device)
    
    # Create fusion system
    fusion_system = EnhancedSmartFusion(model_info['breeds'], trained_model, prototypes, device)
    
    # Test with a sample image
    test_image = "dataset/Nagpuri/Nagpuri_1.jpg"
    
    print(f"\n🧪 Testing with: {test_image}")
    predictions, smart_pred = fusion_system.predict_with_enhanced_fusion(test_image)
    
    if smart_pred:
        print(f"\n✅ Fusion Strategy: {smart_pred.fusion_strategy}")
        print(f"📊 Confidence Level: {smart_pred.confidence_level}")
        print(f"💡 Explanation: {smart_pred.explanation}")
        print("\n🏆 Top Predictions:")
        for breed, score in list(predictions.items())[:3]:
            print(f"  {breed}: {score:.4f}")
    else:
        print("❌ Fusion failed")

if __name__ == "__main__":
    test_enhanced_fusion()