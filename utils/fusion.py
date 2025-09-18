#!/usr/bin/env python3
"""
Multi-Feature Fusion System

This module combines morphological features (horn, coat, body) with global CNN features
to produce highly discriminative and accurate breed predictions.

The fusion system addresses the 0.001 confidence difference problem by leveraging
breed-specific morphological characteristics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .morphology import MorphologicalFeatureExtractor, MorphologicalFeatures

@dataclass
class FusedPrediction:
    """Container for fused prediction results"""
    breed_scores: Dict[str, float]
    feature_contributions: Dict[str, Dict[str, float]]
    confidence_level: str
    explanation: str
    margin: float

class MultiFeatureFusion:
    """
    Fusion system that combines CNN features with morphological features
    for exact breed prediction
    """
    
    def __init__(self, breeds: List[str]):
        self.breeds = breeds  # Keep original capitalization for CNN matching
        self.breeds_lower = [breed.lower() for breed in breeds]  # For morphological matching
        self.morphology_extractor = MorphologicalFeatureExtractor()
        
        # Adaptive fusion weights (can be learned from validation data)
        self.fusion_weights = {
            'global_cnn': 0.4,      # Global CNN features (current system)
            'morphological': 0.6    # Morphological features (new system)
        }
        
        # Confidence thresholds for different prediction levels
        self.confidence_thresholds = {
            'exact': 0.85,       # Very high confidence
            'high': 0.75,        # High confidence  
            'medium': 0.65,      # Medium confidence
            'low': 0.55,         # Low confidence
            'uncertain': 0.0     # Below low threshold
        }
        
        # Minimum margin for confident prediction
        self.min_margin = 0.05
        
    def extract_morphological_features(self, image) -> MorphologicalFeatures:
        """Extract morphological features from image"""
        return self.morphology_extractor.extract_all_features(image)
    
    def compute_morphological_scores(self, morph_features: MorphologicalFeatures) -> Dict[str, float]:
        """Compute breed scores from morphological features"""
        return self.morphology_extractor.compute_breed_scores(morph_features)
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to sum to 1.0"""
        total = sum(scores.values())
        if total > 0:
            return {breed: score / total for breed, score in scores.items()}
        else:
            # If no valid scores, return uniform distribution
            return {breed: 1.0 / len(scores) for breed in scores.keys()}
    
    def fuse_predictions(self, cnn_similarities: Dict[str, float], 
                        morph_features: MorphologicalFeatures,
                        image) -> FusedPrediction:
        """
        Fuse CNN predictions with morphological features for enhanced accuracy
        """
        
        # Normalize CNN similarities (they're already computed)
        cnn_scores = self.normalize_scores(cnn_similarities)
        
        # Compute morphological scores
        morph_scores = self.compute_morphological_scores(morph_features)
        morph_scores_normalized = self.normalize_scores(morph_scores)
        
        # Adaptive fusion based on feature confidence
        total_morph_confidence = sum(morph_features.confidence_scores.values())
        
        # Adjust fusion weights based on morphological feature quality
        if total_morph_confidence > 1.5:  # High confidence in morphological features
            adaptive_weights = {
                'global_cnn': 0.3,
                'morphological': 0.7
            }
        elif total_morph_confidence > 0.8:  # Medium confidence
            adaptive_weights = {
                'global_cnn': 0.4,
                'morphological': 0.6
            }
        else:  # Low confidence in morphological features
            adaptive_weights = {
                'global_cnn': 0.6,
                'morphological': 0.4
            }
        
        # Perform weighted fusion
        fused_scores = {}
        feature_contributions = {}
        
        for i, breed in enumerate(self.breeds):
            breed_lower = self.breeds_lower[i]
            
            # Get scores (default to 0 if breed not found)
            # CNN similarities use original capitalized breed names
            cnn_score = cnn_scores.get(breed, 0.0)
            # Morphological scores use lowercase breed names
            morph_score = morph_scores_normalized.get(breed_lower, 0.0)
            
            # Weighted combination
            fused_score = (adaptive_weights['global_cnn'] * cnn_score + 
                          adaptive_weights['morphological'] * morph_score)
            
            fused_scores[breed] = fused_score
            
            # Track feature contributions for explanation
            feature_contributions[breed] = {
                'cnn_contribution': cnn_score * adaptive_weights['global_cnn'],
                'morph_contribution': morph_score * adaptive_weights['morphological'],
                'cnn_raw_score': cnn_score,
                'morph_raw_score': morph_score
            }
        
        # Sort by fused score
        sorted_breeds = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence and margin
        best_breed, best_score = sorted_breeds[0]
        second_score = sorted_breeds[1][1] if len(sorted_breeds) > 1 else 0.0
        margin = best_score - second_score
        
        # Determine confidence level
        confidence_level = 'uncertain'
        for level, threshold in sorted(self.confidence_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if best_score >= threshold and margin >= self.min_margin:
                confidence_level = level
                break
        
        # Generate explanation
        explanation = self._generate_explanation(
            best_breed, morph_features, feature_contributions[best_breed],
            confidence_level, margin
        )
        
        return FusedPrediction(
            breed_scores=dict(sorted_breeds),
            feature_contributions=feature_contributions,
            confidence_level=confidence_level,
            explanation=explanation,
            margin=margin
        )
    
    def _generate_explanation(self, breed: str, morph_features: MorphologicalFeatures,
                            contributions: Dict[str, float], confidence_level: str,
                            margin: float) -> str:
        """Generate human-readable explanation for the prediction"""
        
        explanations = []
        
        # Analyze dominant features
        cnn_contrib = contributions['cnn_contribution']
        morph_contrib = contributions['morph_contribution']
        
        if morph_contrib > cnn_contrib:
            explanations.append("Strong morphological match")
        else:
            explanations.append("Strong global appearance match")
        
        # Analyze specific morphological features
        feature_strengths = []
        
        # Horn features
        if morph_features.confidence_scores.get('horn', 0) > 0.3:
            dominant_horn = max(morph_features.horn_features.items(), 
                              key=lambda x: x[1]) if morph_features.horn_features else None
            if dominant_horn and dominant_horn[1] > 0.3:
                feature_strengths.append(f"horn shape ({dominant_horn[0]})")
        
        # Coat features  
        if morph_features.confidence_scores.get('coat', 0) > 0.3:
            dominant_coat = max(morph_features.coat_features.items(),
                              key=lambda x: x[1]) if morph_features.coat_features else None
            if dominant_coat and dominant_coat[1] > 0.3:
                feature_strengths.append(f"coat color ({dominant_coat[0]})")
        
        # Body features
        if morph_features.confidence_scores.get('body', 0) > 0.3:
            dominant_body = max(morph_features.body_features.items(),
                              key=lambda x: x[1]) if morph_features.body_features else None
            if dominant_body and dominant_body[1] > 0.3:
                feature_strengths.append(f"body type ({dominant_body[0]})")
        
        if feature_strengths:
            explanations.append(f"Key features: {', '.join(feature_strengths)}")
        
        # Confidence assessment
        if confidence_level == 'exact':
            explanations.append(f"EXACT identification (margin: {margin:.3f})")
        elif confidence_level == 'high':
            explanations.append(f"High confidence (margin: {margin:.3f})")
        elif margin < self.min_margin:
            explanations.append("Multiple breeds show similar characteristics")
        
        return " | ".join(explanations)
    
    def predict_with_fusion(self, image, model, prototypes, device) -> Tuple[List[Tuple[str, float]], FusedPrediction]:
        """
        Main prediction function that combines CNN and morphological features
        """
        
        # Step 1: Extract global CNN features (existing system)
        from utils import get_transforms, load_and_preprocess_image
        
        # Preprocess image for CNN
        if isinstance(image, str):
            image_tensor = load_and_preprocess_image(image)
        else:
            transform = get_transforms(augment=False)
            image_tensor = transform(image)
        
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Extract CNN features
        with torch.no_grad():
            features = model(image_batch)
            features_normalized = F.normalize(features, p=2, dim=1)
            query_embedding = features_normalized.squeeze(0)
        
        # Compute CNN similarities
        from utils import compute_similarities
        cnn_similarities = compute_similarities(query_embedding, prototypes)
        
        # Step 2: Extract morphological features
        morph_features = self.extract_morphological_features(image)
        
        # Step 3: Fuse predictions
        fused_prediction = self.fuse_predictions(cnn_similarities, morph_features, image)
        
        # Step 4: Convert to standard format for compatibility
        predictions = list(fused_prediction.breed_scores.items())
        
        return predictions, fused_prediction
    
    def get_feature_visualization_data(self, morph_features: MorphologicalFeatures) -> Dict:
        """Get data for visualizing detected features in Streamlit"""
        viz_data = {
            'horn_analysis': {
                'detected_features': morph_features.horn_features,
                'confidence': morph_features.confidence_scores.get('horn', 0.0),
                'interpretation': 'Horn shape and structure analysis'
            },
            'coat_analysis': {
                'detected_features': morph_features.coat_features,
                'confidence': morph_features.confidence_scores.get('coat', 0.0),
                'interpretation': 'Coat color and pattern analysis'
            },
            'body_analysis': {
                'detected_features': morph_features.body_features,
                'confidence': morph_features.confidence_scores.get('body', 0.0),
                'interpretation': 'Body size and morphology analysis'
            }
        }
        return viz_data

class BreedExpertSystem:
    """
    Expert system that provides breed-specific insights and recommendations
    """
    
    def __init__(self):
        self.breed_characteristics = {
            'murrah': {
                'key_features': ['black_coat', 'curved_horns', 'large_body'],
                'distinguishing_factors': 'Largest milk producer, distinctive black color',
                'confusion_breeds': ['jaffarbadi', 'surti'],
                'confidence_boosters': ['coat_color', 'body_size']
            },
            'gir': {
                'key_features': ['lyre_horns', 'spotted_coat', 'drooping_ears'],
                'distinguishing_factors': 'Sacred cattle with distinctive lyre-shaped horns',
                'confusion_breeds': ['kankrej', 'sahiwal'],
                'confidence_boosters': ['horn_shape', 'coat_pattern']
            },
            'sahiwal': {
                'key_features': ['reddish_brown_coat', 'small_horns', 'heat_tolerance'],
                'distinguishing_factors': 'Reddish-brown color with heat adaptation',
                'confusion_breeds': ['gir', 'tharparkar'],
                'confidence_boosters': ['coat_color', 'body_proportions']
            }
            # Add more breeds as needed
        }
    
    def get_breed_insights(self, breed: str, fused_prediction: FusedPrediction) -> Dict[str, str]:
        """Get expert insights for a predicted breed"""
        breed_key = breed.lower()
        insights = {}
        
        if breed_key in self.breed_characteristics:
            char = self.breed_characteristics[breed_key]
            
            insights['key_features'] = f"Key identifying features: {', '.join(char['key_features'])}"
            insights['distinguishing_factors'] = char['distinguishing_factors']
            
            # Check for confusion with similar breeds
            if fused_prediction.margin < 0.1:
                similar_breeds = char.get('confusion_breeds', [])
                if similar_breeds:
                    insights['warning'] = f"May be confused with: {', '.join(similar_breeds)}"
        
        return insights