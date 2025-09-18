#!/usr/bin/env python3
"""
Smart CNN-First Fusion System

This system prioritizes the proven accurate CNN predictions and only uses 
morphological features for tie-breaking or confidence enhancement.

Strategy:
1. If CNN confidence > 0.85: Use CNN prediction (proven accurate)
2. If CNN margin > 0.05: Use CNN prediction (clear winner)
3. If CNN margin < 0.01: Use morphological features to break tie
4. Always preserve CNN ranking order when confidence is high
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .morphology import MorphologicalFeatureExtractor, MorphologicalFeatures

@dataclass
class SmartFusedPrediction:
    """Container for smart fused prediction results"""
    breed_scores: Dict[str, float]
    fusion_strategy: str  # 'cnn_dominant', 'morpho_tiebreak', 'full_fusion'
    confidence_level: str
    explanation: str
    margin: float
    cnn_confidence: float
    morpho_confidence: float

class SmartCNNFirstFusion:
    """
    Smart fusion system that prioritizes proven CNN accuracy
    """
    
    def __init__(self, breeds: List[str]):
        self.breeds = breeds
        self.morphology_extractor = MorphologicalFeatureExtractor()
        
        # CNN dominance thresholds (these preserve proven accuracy)
        self.high_confidence_threshold = 0.85  # Above this: CNN is dominant
        self.clear_margin_threshold = 0.05     # Above this: CNN decision is clear
        self.tie_break_threshold = 0.01        # Below this: use morphology to break tie
        
        # Confidence levels
        self.confidence_thresholds = {
            'exact': 0.90,       # Very high confidence
            'high': 0.85,        # High confidence  
            'medium': 0.75,      # Medium confidence
            'low': 0.65,         # Low confidence
            'uncertain': 0.0     # Below low threshold
        }
    
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
            return {breed: 1.0 / len(scores) for breed in scores.keys()}
    
    def smart_fuse_predictions(self, cnn_similarities: Dict[str, float], 
                              morph_features: MorphologicalFeatures,
                              image) -> SmartFusedPrediction:
        """
        Smart fusion that preserves CNN accuracy when confidence is high
        """
        
        # Use raw CNN similarities for decisions (they are already cosine similarities 0-1)
        sorted_cnn_raw = sorted(cnn_similarities.items(), key=lambda x: x[1], reverse=True)
        best_cnn_breed, best_cnn_score_raw = sorted_cnn_raw[0]
        second_cnn_score_raw = sorted_cnn_raw[1][1] if len(sorted_cnn_raw) > 1 else 0.0
        cnn_margin_raw = best_cnn_score_raw - second_cnn_score_raw
        
        # Normalize CNN similarities for final scoring
        cnn_scores = self.normalize_scores(cnn_similarities)
        sorted_cnn = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
        best_cnn_score = sorted_cnn[0][1]
        cnn_margin = best_cnn_score - sorted_cnn[1][1] if len(sorted_cnn) > 1 else 0.0
        
        # Compute morphological confidence
        total_morph_confidence = sum(morph_features.confidence_scores.values())
        
        print(f"🧠 CNN Top: {best_cnn_breed} ({best_cnn_score_raw:.6f})")
        print(f"📏 CNN Margin: {cnn_margin_raw:.6f}")
        print(f"🔬 Morph Confidence: {total_morph_confidence:.3f}")
        
        # STRATEGY 1: High CNN confidence - Use CNN prediction directly (use raw scores)
        if best_cnn_score_raw >= self.high_confidence_threshold:
            print("✅ Strategy: CNN_DOMINANT (High confidence)")
            fusion_strategy = "cnn_dominant"
            final_scores = dict(sorted_cnn_raw)  # Use raw similarities for high confidence
            explanation = f"High CNN confidence ({best_cnn_score_raw:.4f}) - Using proven CNN prediction"
        
        # STRATEGY 2: Clear CNN margin - Use CNN prediction (use raw scores)
        elif cnn_margin_raw >= self.clear_margin_threshold:
            print("✅ Strategy: CNN_DOMINANT (Clear margin)")
            fusion_strategy = "cnn_dominant"
            final_scores = dict(sorted_cnn_raw)  # Use raw similarities for clear margin
            explanation = f"Clear CNN margin ({cnn_margin_raw:.4f}) - Using CNN decision"
        
        # STRATEGY 3: Small CNN margin - Use morphology to break tie (use raw margin)
        elif cnn_margin_raw <= self.tie_break_threshold and total_morph_confidence > 1.0:
            print("🔧 Strategy: MORPHO_TIEBREAK (Breaking CNN tie)")
            fusion_strategy = "morpho_tiebreak"
            
            # Get morphological scores for top CNN candidates
            morph_scores = self.compute_morphological_scores(morph_features)
            morph_scores_normalized = self.normalize_scores(morph_scores)
            
            # Apply morphology only to top 3 CNN candidates
            top_cnn_candidates = sorted_cnn[:3]
            enhanced_scores = {}
            
            for breed, cnn_score in top_cnn_candidates:
                breed_lower = breed.lower()
                morph_boost = morph_scores_normalized.get(breed_lower, 0.0) * 0.1  # Small boost
                enhanced_scores[breed] = cnn_score + morph_boost
            
            # Add remaining breeds with original CNN scores
            for breed, cnn_score in sorted_cnn[3:]:
                enhanced_scores[breed] = cnn_score
            
            final_scores = dict(sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True))
            explanation = f"CNN tie ({cnn_margin_raw:.4f}) - Morphology tiebreaker applied"
        
        # STRATEGY 4: Medium confidence - Light fusion
        else:
            print("⚖️ Strategy: LIGHT_FUSION (Balanced approach)")
            fusion_strategy = "light_fusion"
            
            # Compute morphological scores
            morph_scores = self.compute_morphological_scores(morph_features)
            morph_scores_normalized = self.normalize_scores(morph_scores)
            
            # Light fusion: 80% CNN, 20% morphology (preserves CNN dominance)
            fused_scores = {}
            for breed in self.breeds:
                breed_lower = breed.lower()
                cnn_score = cnn_scores.get(breed, 0.0)
                morph_score = morph_scores_normalized.get(breed_lower, 0.0)
                
                # Light fusion weights
                fused_score = 0.8 * cnn_score + 0.2 * morph_score
                fused_scores[breed] = fused_score
            
            final_scores = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
            explanation = f"Light fusion (80% CNN, 20% morphology) for balanced accuracy"
        
        # Calculate final metrics
        final_sorted = list(final_scores.items())
        best_breed, best_score = final_sorted[0]
        second_score = final_sorted[1][1] if len(final_sorted) > 1 else 0.0
        final_margin = best_score - second_score
        
        # Determine confidence level based on final score and strategy
        if fusion_strategy == "cnn_dominant" and best_cnn_score_raw >= 0.90:
            confidence_level = 'exact'
        elif fusion_strategy == "cnn_dominant" and best_cnn_score_raw >= 0.85:
            confidence_level = 'high'
        elif best_score >= 0.75 or best_cnn_score_raw >= 0.80:
            confidence_level = 'medium'
        elif best_score >= 0.65 or best_cnn_score_raw >= 0.70:
            confidence_level = 'low'
        else:
            confidence_level = 'uncertain'
        
        return SmartFusedPrediction(
            breed_scores=final_scores,
            fusion_strategy=fusion_strategy,
            confidence_level=confidence_level,
            explanation=explanation,
            margin=final_margin,
            cnn_confidence=best_cnn_score_raw,
            morpho_confidence=total_morph_confidence
        )
    
    def predict_with_smart_fusion(self, image, model, prototypes, device) -> Tuple[List[Tuple[str, float]], SmartFusedPrediction]:
        """
        Main prediction function with smart CNN-first fusion
        """
        
        # Step 1: Extract global CNN features (proven accurate system)
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
        
        # Compute CNN similarities (the proven accurate part)
        from utils import compute_similarities
        cnn_similarities = compute_similarities(query_embedding, prototypes)
        
        # Step 2: Extract morphological features (for enhancement only)
        morph_features = self.extract_morphological_features(image)
        
        # Step 3: Smart fusion that preserves CNN accuracy
        smart_prediction = self.smart_fuse_predictions(cnn_similarities, morph_features, image)
        
        # Step 4: Convert to standard format
        predictions = list(smart_prediction.breed_scores.items())
        
        return predictions, smart_prediction