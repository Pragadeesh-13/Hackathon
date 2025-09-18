#!/usr/bin/env python3
"""
MAXIMUM 10-BREED INFERENCE SYSTEM
Near-Perfect Cattle & Buffalo Breed Recognition

This system provides near-perfect accuracy for 10 Indian breeds:
Buffalo: Bhadawari, Jaffarbadi, Mehsana, Murrah, Surti
Cattle: Gir, Kankrej, Ongole, Sahiwal, Tharparkar

Usage:
    python infer_maximum_10breed.py "path/to/image.jpg"
    python infer_maximum_10breed.py "dataset/Murrah/murrah_001.jpg" --confidence-threshold 0.7
"""

import os
import sys
import pickle
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple

# Add utils to path
sys.path.append('.')
from utils import (
    FeatureExtractor,
    get_device,
    get_transforms,
    compute_similarities,
    get_top_predictions
)

class Maximum10BreedInference:
    """Near-perfect inference for 10 cattle & buffalo breeds"""
    
    def __init__(self, prototypes_path: str = "models/prototypes_maximum_10breed.pkl"):
        self.prototypes_path = prototypes_path
        self.device = get_device()
        
        # Load prototypes and model
        self.prototypes, self.breeds, self.model_name, self.config = self._load_prototypes()
        
        # Load model
        self.model = FeatureExtractor(model_name=self.model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Breed categories
        self.buffalo_breeds = ['Bhadawari', 'Jaffarbadi', 'Mehsana', 'Murrah', 'Surti']
        self.cattle_breeds = ['Gir', 'Kankrej', 'Ongole', 'Sahiwal', 'Tharparkar']
        
        # Transform
        self.transform = get_transforms(augment=False)
        
        print("=" * 90)
        print("🚀 MAXIMUM 10-BREED RECOGNITION - NEAR-PERFECT ACCURACY")
        print("=" * 90)
        print(f"🤖 Model: {self.model_name.upper()} (Maximum Optimization)")
        print(f"📊 Breeds: {len(self.breeds)} total")
        print(f"🐃 Buffalo: {len([b for b in self.breeds if b in self.buffalo_breeds])} breeds")
        print(f"🐄 Cattle: {len([b for b in self.breeds if b in self.cattle_breeds])} breeds")
        
        if self.config.get('optimization_level') == 'maximum_discrimination_10breed':
            print("🔥 Optimization: MAXIMUM DISCRIMINATION")
            if self.config.get('near_perfect_targeting'):
                print("🎯 Target: NEAR-PERFECT ACCURACY")
            if self.config.get('cross_species_discrimination'):
                print("⚖️ Cross-species enhancement: ACTIVE")
        
        print(f"💾 Using device: {self.device}")
        print("=" * 90)
    
    def _load_prototypes(self) -> Tuple[Dict, List, str, Dict]:
        """Load maximum prototypes"""
        try:
            with open(self.prototypes_path, 'rb') as f:
                data = pickle.load(f)
            
            prototypes = data['prototypes']
            breeds = data['breeds']
            model_name = data['model_name']
            config = data.get('config', {})
            
            print(f"📂 Loaded: {self.prototypes_path}")
            print(f"🧬 Breeds: {', '.join(breeds)}")
            
            if 'total_features' in data:
                print(f"🔥 Training features: {data['total_features']}")
            
            return prototypes, breeds, model_name, config
            
        except FileNotFoundError:
            print(f"❌ Prototypes not found: {self.prototypes_path}")
            print("   Run: python prototype_maximum_10breed.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading prototypes: {e}")
            sys.exit(1)
    
    def predict(self, image_path: str, confidence_threshold: float = 0.7, 
                top_k: int = 5) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        """Predict breed with maximum accuracy"""
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            features_normalized = F.normalize(features, p=2, dim=1)
            query_embedding = features_normalized.squeeze(0)
        
        # Compute similarities with maximum prototypes
        similarities = {}
        for breed in self.breeds:
            prototype = self.prototypes[breed].to(self.device)
            similarity = F.cosine_similarity(query_embedding.unsqueeze(0), 
                                           prototype.unsqueeze(0)).item()
            similarities[breed] = similarity
        
        # Get predictions
        predictions = get_top_predictions(similarities, top_k=len(self.breeds))
        
        return predictions, similarities
    
    def analyze_prediction_quality(self, predictions: List[Tuple[str, float]], 
                                 similarities: Dict[str, float]) -> Dict[str, str]:
        """Analyze prediction quality and confidence"""
        
        if not predictions:
            return {"quality": "ERROR", "confidence": "NONE"}
        
        best_breed, best_score = predictions[0]
        second_score = predictions[1][1] if len(predictions) > 1 else 0.0
        margin = best_score - second_score
        
        # Enhanced quality assessment for 10 breeds
        if best_score >= 0.90 and margin >= 0.05:
            quality = "EXCEPTIONAL"
            confidence_level = "MAXIMUM"
        elif best_score >= 0.85 and margin >= 0.03:
            quality = "EXCELLENT"
            confidence_level = "HIGH"
        elif best_score >= 0.80 and margin >= 0.02:
            quality = "VERY GOOD"
            confidence_level = "GOOD"
        elif best_score >= 0.75 and margin >= 0.01:
            quality = "GOOD"
            confidence_level = "MODERATE"
        elif best_score >= 0.70:
            quality = "ACCEPTABLE"
            confidence_level = "LOW"
        else:
            quality = "UNCERTAIN"
            confidence_level = "VERY LOW"
        
        # Species consistency check
        species_analysis = self._analyze_species_consistency(similarities)
        
        return {
            "quality": quality,
            "confidence": confidence_level,
            "margin": f"{margin:.4f}",
            "species_analysis": species_analysis
        }
    
    def _analyze_species_consistency(self, similarities: Dict[str, float]) -> Dict:
        """Analyze species-level prediction consistency"""
        
        buffalo_scores = [similarities[breed] for breed in self.buffalo_breeds if breed in similarities]
        cattle_scores = [similarities[breed] for breed in self.cattle_breeds if breed in similarities]
        
        buffalo_avg = sum(buffalo_scores) / len(buffalo_scores) if buffalo_scores else 0
        cattle_avg = sum(cattle_scores) / len(cattle_scores) if cattle_scores else 0
        
        species_margin = abs(buffalo_avg - cattle_avg)
        dominant_species = "Buffalo" if buffalo_avg > cattle_avg else "Cattle"
        
        if species_margin >= 0.05:
            species_confidence = "STRONG"
        elif species_margin >= 0.03:
            species_confidence = "MODERATE"
        else:
            species_confidence = "WEAK"
        
        return {
            "dominant_species": dominant_species,
            "species_confidence": species_confidence,
            "species_margin": f"{species_margin:.4f}",
            "buffalo_avg": f"{buffalo_avg:.4f}",
            "cattle_avg": f"{cattle_avg:.4f}"
        }
    
    def format_breed_info(self, breed_name: str) -> str:
        """Get formatted breed information"""
        breed_info = {
            # Buffalo breeds
            "Bhadawari": "🐃 Hardy buffalo, waterlogged-adapted, Uttar Pradesh origin",
            "Jaffarbadi": "🐃 Large buffalo, high milk yield, Gujarat commercial breed",
            "Mehsana": "🐃 Heat-tolerant buffalo, semi-arid adapted, Gujarat origin",
            "Murrah": "🐃 World-famous buffalo, highest milk producer, Haryana breed",
            "Surti": "🐃 Drought-resistant buffalo, high-fat milk, Gujarat origin",
            
            # Cattle breeds
            "Gir": "🐄 Sacred cattle, disease-resistant, Gujarat Gir forest origin",
            "Kankrej": "🐄 Dual-purpose cattle, drought-tolerant, Gujarat-Rajasthan",
            "Ongole": "🐄 Large cattle, heat-resistant, Andhra Pradesh origin",
            "Sahiwal": "🐄 High milk yield cattle, heat-tolerant, Punjab origin",
            "Tharparkar": "🐄 Desert cattle, extreme heat-resistant, Rajasthan origin"
        }
        
        return breed_info.get(breed_name, f"🐾 {breed_name} - Indian breed")

def main():
    parser = argparse.ArgumentParser(description="Maximum 10-Breed Inference")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--prototypes-path", default="models/prototypes_maximum_10breed.pkl")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-all", action="store_true", help="Show all breed scores")
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        return
    
    # Initialize inference system
    try:
        inference = Maximum10BreedInference(args.prototypes_path)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Make prediction
    try:
        predictions, similarities = inference.predict(
            args.image_path,
            confidence_threshold=args.confidence_threshold,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Analyze quality
    quality_analysis = inference.analyze_prediction_quality(predictions, similarities)
    
    # Display results
    print(f"\n📸 Image: {args.image_path}")
    print(f"🔍 Analysis: {quality_analysis['quality']} quality, {quality_analysis['confidence']} confidence")
    
    # Species analysis
    species_info = quality_analysis['species_analysis']
    print(f"🧬 Species: {species_info['dominant_species']} ({species_info['species_confidence']} confidence)")
    
    # Top predictions
    print(f"\n🏆 Top {args.top_k} Predictions:")
    for i, (breed, confidence) in enumerate(predictions[:args.top_k]):
        bar_length = int(confidence * 80)
        bar = "█" * bar_length + "░" * (80 - bar_length)
        
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        
        species = "Buffalo" if breed in inference.buffalo_breeds else "Cattle"
        print(f"  {emoji} {breed:<12} │{bar}│ {confidence:.4f} ({species})")
    
    # Final prediction
    best_breed, best_confidence = predictions[0]
    margin = float(quality_analysis['margin'])
    
    print(f"\n{'='*90}")
    if best_confidence >= args.confidence_threshold:
        print(f"✅ MAXIMUM CONFIDENCE IDENTIFICATION: {best_breed}")
        print(f"   🎯 Confidence: {best_confidence:.4f} (≥ {args.confidence_threshold} threshold)")
        print(f"   📏 Margin: {margin:.4f}")
        print(f"   ℹ️  {inference.format_breed_info(best_breed)}")
    else:
        print(f"⚠️  BELOW THRESHOLD IDENTIFICATION")
        print(f"   🎯 Best match: {best_breed} ({best_confidence:.4f})")
        print(f"   📏 Threshold: {args.confidence_threshold}")
        print(f"   💡 Consider lowering threshold or adding more training data")
    
    print(f"{'='*90}")
    
    # Show all breeds if requested
    if args.show_all:
        print(f"\n📊 All Breed Similarities:")
        all_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print("   Buffalo Breeds:")
        for breed, sim in all_similarities:
            if breed in inference.buffalo_breeds:
                print(f"      {breed:<12}: {sim:.4f}")
        
        print("   Cattle Breeds:")
        for breed, sim in all_similarities:
            if breed in inference.cattle_breeds:
                print(f"      {breed:<12}: {sim:.4f}")

if __name__ == "__main__":
    main()