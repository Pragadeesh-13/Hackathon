#!/usr/bin/env python3
"""
Morphological Feature Extraction System

This module implements breed-specific morphological feature extraction:
- Horn shape analysis (curved, straight, lyre, polled)
- Coat color and pattern detection
- Body size and proportion analysis

These features are combined with global CNN features for exact breed prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class MorphologicalFeatures:
    """Container for extracted morphological features"""
    horn_features: Dict[str, float]
    coat_features: Dict[str, float] 
    body_features: Dict[str, float]
    confidence_scores: Dict[str, float]
    
class HornShapeAnalyzer:
    """Analyzes horn shapes and characteristics"""
    
    def __init__(self):
        self.horn_patterns = {
            'curved_forward': {'murrah': 0.8, 'surti': 0.9, 'bhadawari': 0.7},
            'lyre_shaped': {'gir': 0.95, 'kankrej': 0.9, 'sahiwal': 0.3},
            'straight_upward': {'ongole': 0.8, 'tharparkar': 0.6},
            'curved_backward': {'jaffarbadi': 0.9, 'mehsana': 0.7},
            'small_polled': {'murrah': 0.4, 'sahiwal': 0.7}
        }
    
    def detect_head_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop head region for horn analysis"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use cascade classifier for animal face detection (simplified)
        # In production, you'd use a trained animal head detector
        
        # For now, assume head is in upper portion of image
        height, width = gray.shape
        head_region = image[:int(height * 0.4), :]  # Top 40% of image
        
        return head_region
    
    def analyze_horn_contours(self, head_region: np.ndarray) -> Dict[str, float]:
        """Analyze horn contours and shapes"""
        gray = cv2.cvtColor(head_region, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for horn outlines
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        horn_features = {
            'horn_curvature': 0.0,
            'horn_length_ratio': 0.0,
            'horn_spacing': 0.0,
            'horn_thickness': 0.0,
            'horn_angle': 0.0
        }
        
        # Filter contours that could be horns (based on shape and position)
        potential_horns = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Filter by reasonable horn size
                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if 1.5 < aspect_ratio < 10:  # Horn-like aspect ratio
                    potential_horns.append(contour)
        
        if len(potential_horns) >= 1:
            # Analyze the most prominent horn-like contours
            horn_contour = max(potential_horns, key=cv2.contourArea)
            
            # Calculate curvature
            if len(horn_contour) > 10:
                # Fit ellipse to measure curvature
                try:
                    ellipse = cv2.fitEllipse(horn_contour)
                    # Use ellipse parameters to estimate curvature
                    (center, axes, angle) = ellipse
                    major_axis, minor_axis = max(axes), min(axes)
                    
                    horn_features['horn_curvature'] = 1.0 - (minor_axis / major_axis) if major_axis > 0 else 0.0
                    horn_features['horn_length_ratio'] = major_axis / head_region.shape[1]  # Relative to head width
                    horn_features['horn_angle'] = abs(angle) / 90.0  # Normalized angle
                    
                except:
                    pass
            
            # Calculate thickness (average width of contour)
            if len(horn_contour) > 5:
                hull = cv2.convexHull(horn_contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(horn_contour)
                horn_features['horn_thickness'] = contour_area / hull_area if hull_area > 0 else 0.0
        
        # If two horns detected, calculate spacing
        if len(potential_horns) >= 2:
            # Sort by area and take two largest
            sorted_horns = sorted(potential_horns, key=cv2.contourArea, reverse=True)[:2]
            
            # Calculate centers
            moments1 = cv2.moments(sorted_horns[0])
            moments2 = cv2.moments(sorted_horns[1])
            
            if moments1['m00'] > 0 and moments2['m00'] > 0:
                center1 = (moments1['m10'] / moments1['m00'], moments1['m01'] / moments1['m00'])
                center2 = (moments2['m10'] / moments2['m00'], moments2['m01'] / moments2['m00'])
                
                # Calculate distance between horn centers
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                horn_features['horn_spacing'] = distance / head_region.shape[1]  # Normalized to head width
        
        return horn_features
    
    def classify_horn_type(self, horn_features: Dict[str, float]) -> Dict[str, float]:
        """Classify horn type based on extracted features"""
        horn_scores = {}
        
        curvature = horn_features.get('horn_curvature', 0.0)
        angle = horn_features.get('horn_angle', 0.0)
        length_ratio = horn_features.get('horn_length_ratio', 0.0)
        
        # Curved forward (typical buffalo horns)
        if curvature > 0.3 and angle < 0.3:
            horn_scores['curved_forward'] = min(1.0, curvature + (1.0 - angle))
        
        # Lyre-shaped (typical Gir, Kankrej)
        if curvature > 0.4 and 0.2 < angle < 0.8:
            horn_scores['lyre_shaped'] = min(1.0, curvature * 2 * angle)
        
        # Straight upward (Ongole, some Tharparkar)
        if curvature < 0.2 and angle > 0.7:
            horn_scores['straight_upward'] = min(1.0, (1.0 - curvature) * angle)
        
        # Curved backward (Jaffarbadi)
        if curvature > 0.5 and angle > 0.6:
            horn_scores['curved_backward'] = min(1.0, curvature * angle)
        
        # Small/polled (some Murrah, Sahiwal)
        if length_ratio < 0.1:
            horn_scores['small_polled'] = 1.0 - length_ratio * 10
        
        return horn_scores

class CoatColorAnalyzer:
    """Analyzes coat color and patterns"""
    
    def __init__(self):
        self.breed_colors = {
            'black': {'murrah': 0.9, 'jaffarbadi': 0.8},
            'brown': {'surti': 0.8, 'sahiwal': 0.9, 'gir': 0.3},
            'white': {'ongole': 0.9, 'tharparkar': 0.8, 'gir': 0.6},
            'grey': {'kankrej': 0.9, 'mehsana': 0.6},
            'reddish_brown': {'sahiwal': 0.95, 'gir': 0.7},
            'spotted': {'gir': 0.8, 'sahiwal': 0.3}
        }
    
    def extract_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[np.ndarray, float]]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove very dark pixels (shadows) and very bright pixels (overexposure)
        brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(brightness > 30) & (brightness < 225)]
        
        if len(valid_pixels) < 100:
            valid_pixels = pixels
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=min(num_colors, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Get colors and their percentages
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            color_percentages = []
            for i in range(len(colors)):
                percentage = np.sum(labels == i) / len(labels)
                color_percentages.append((colors[i], percentage))
            
            # Sort by percentage
            color_percentages.sort(key=lambda x: x[1], reverse=True)
            
            return color_percentages
        
        except:
            # Fallback: return average color
            avg_color = np.mean(valid_pixels, axis=0)
            return [(avg_color, 1.0)]
    
    def classify_color(self, rgb_color: np.ndarray) -> str:
        """Classify RGB color into breed-relevant categories"""
        r, g, b = rgb_color
        
        # Convert to HSV for better color classification
        hsv = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        # Color classification based on HSV values
        if v < 60:  # Very dark
            return 'black'
        elif v > 200 and s < 30:  # Very light with low saturation
            return 'white'
        elif s < 50:  # Low saturation (greyish)
            if v > 150:
                return 'light_grey'
            else:
                return 'grey'
        else:  # Saturated colors
            if h < 15 or h > 165:  # Red range
                if v > 100:
                    return 'reddish_brown'
                else:
                    return 'brown'
            elif 15 <= h < 35:  # Orange/brown range
                return 'brown'
            elif 35 <= h < 85:  # Yellow/green range (rare in cattle)
                return 'brown'  # Classify as brown
            else:  # Blue range (very rare)
                return 'grey'
    
    def detect_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Detect coat patterns like spots, stripes, etc."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        patterns = {
            'solid': 0.0,
            'spotted': 0.0,
            'patched': 0.0,
            'uniform': 0.0
        }
        
        # Calculate color variance to detect patterns
        # Use local variance to detect spots/patches
        kernel = np.ones((15, 15), np.float32) / 225
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_img = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance_img = sqr_img - mean_img**2
        
        avg_variance = np.mean(variance_img)
        
        # High variance indicates patterns/spots
        if avg_variance > 500:
            patterns['spotted'] = min(1.0, avg_variance / 1000)
        elif avg_variance > 200:
            patterns['patched'] = min(1.0, avg_variance / 500)
        else:
            patterns['solid'] = 1.0 - (avg_variance / 200)
        
        # Uniformity check
        overall_std = np.std(gray)
        patterns['uniform'] = max(0.0, 1.0 - (overall_std / 50))
        
        return patterns
    
    def analyze_coat(self, image: np.ndarray) -> Dict[str, float]:
        """Comprehensive coat analysis"""
        # Extract dominant colors
        dominant_colors = self.extract_dominant_colors(image, num_colors=3)
        
        # Classify each color
        color_features = {}
        for color, percentage in dominant_colors:
            color_class = self.classify_color(color)
            if color_class in color_features:
                color_features[color_class] += percentage
            else:
                color_features[color_class] = percentage
        
        # Detect patterns
        pattern_features = self.detect_patterns(image)
        
        # Combine color and pattern features
        coat_features = {**color_features, **pattern_features}
        
        return coat_features

class BodyMorphologyAnalyzer:
    """Analyzes body size and proportions"""
    
    def __init__(self):
        self.breed_body_types = {
            'large_heavy': {'jaffarbadi': 0.9, 'murrah': 0.8, 'ongole': 0.8},
            'medium_compact': {'surti': 0.8, 'mehsana': 0.8, 'gir': 0.7},
            'medium_lean': {'sahiwal': 0.8, 'tharparkar': 0.7, 'kankrej': 0.8},
            'small_hardy': {'bhadawari': 0.9}
        }
    
    def segment_animal_body(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment animal body from background (simplified)"""
        # Convert to HSV for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for animal (non-green, non-sky colors)
        # This is a simplified approach - in production, use semantic segmentation
        lower_bound = np.array([0, 30, 30])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def analyze_body_proportions(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze body proportions and size indicators"""
        mask = self.segment_animal_body(image)
        
        if mask is None:
            return {'body_size_score': 0.5, 'height_width_ratio': 0.5, 'compactness': 0.5}
        
        # Find contours of the animal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'body_size_score': 0.5, 'height_width_ratio': 0.5, 'compactness': 0.5}
        
        # Get the largest contour (main body)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Body proportion features
        height_width_ratio = h / w if w > 0 else 1.0
        
        # Body size relative to image
        image_area = image.shape[0] * image.shape[1]
        body_area = cv2.contourArea(main_contour)
        body_size_score = body_area / image_area
        
        # Compactness (how much the animal fills its bounding box)
        bounding_area = w * h
        compactness = body_area / bounding_area if bounding_area > 0 else 0.0
        
        # Additional shape analysis
        try:
            # Fit ellipse to get more shape information
            ellipse = cv2.fitEllipse(main_contour)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            ellipse_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
        except:
            ellipse_ratio = 0.5
        
        body_features = {
            'body_size_score': min(1.0, body_size_score * 2),  # Normalize
            'height_width_ratio': min(1.0, height_width_ratio / 2),  # Normalize
            'compactness': compactness,
            'ellipse_ratio': ellipse_ratio,
            'relative_width': min(1.0, w / image.shape[1]),
            'relative_height': min(1.0, h / image.shape[0])
        }
        
        return body_features
    
    def classify_body_type(self, body_features: Dict[str, float]) -> Dict[str, float]:
        """Classify body type based on measurements"""
        body_type_scores = {}
        
        size_score = body_features.get('body_size_score', 0.5)
        compactness = body_features.get('compactness', 0.5)
        hw_ratio = body_features.get('height_width_ratio', 0.5)
        
        # Large heavy (high size, high compactness)
        if size_score > 0.6 and compactness > 0.6:
            body_type_scores['large_heavy'] = min(1.0, size_score * compactness)
        
        # Medium compact (medium size, high compactness)
        if 0.3 < size_score < 0.7 and compactness > 0.5:
            body_type_scores['medium_compact'] = min(1.0, (1.0 - abs(size_score - 0.5)) * compactness)
        
        # Medium lean (medium size, lower compactness, higher height/width ratio)
        if 0.3 < size_score < 0.7 and compactness < 0.6 and hw_ratio > 0.4:
            body_type_scores['medium_lean'] = min(1.0, (1.0 - abs(size_score - 0.5)) * hw_ratio)
        
        # Small hardy (lower size, variable compactness)
        if size_score < 0.4:
            body_type_scores['small_hardy'] = 1.0 - size_score
        
        return body_type_scores

class MorphologicalFeatureExtractor:
    """Main class that combines all morphological feature extractors"""
    
    def __init__(self):
        self.horn_analyzer = HornShapeAnalyzer()
        self.coat_analyzer = CoatColorAnalyzer()
        self.body_analyzer = BodyMorphologyAnalyzer()
        
        # Breed-specific feature weights (learned from data)
        self.breed_feature_weights = {
            'murrah': {'horn': 0.3, 'coat': 0.4, 'body': 0.3},
            'surti': {'horn': 0.4, 'coat': 0.3, 'body': 0.3},
            'gir': {'horn': 0.5, 'coat': 0.3, 'body': 0.2},
            'sahiwal': {'horn': 0.2, 'coat': 0.5, 'body': 0.3},
            'jaffarbadi': {'horn': 0.4, 'coat': 0.2, 'body': 0.4},
            'bhadawari': {'horn': 0.3, 'coat': 0.3, 'body': 0.4},
            'mehsana': {'horn': 0.4, 'coat': 0.3, 'body': 0.3},
            'kankrej': {'horn': 0.4, 'coat': 0.4, 'body': 0.2},
            'ongole': {'horn': 0.3, 'coat': 0.3, 'body': 0.4},
            'tharparkar': {'horn': 0.3, 'coat': 0.4, 'body': 0.3}
        }
    
    def extract_all_features(self, image: np.ndarray) -> MorphologicalFeatures:
        """Extract all morphological features from an image"""
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Extract features from each analyzer
        horn_features = {}
        coat_features = {}
        body_features = {}
        confidence_scores = {}
        
        try:
            # Horn analysis
            head_region = self.horn_analyzer.detect_head_region(image)
            if head_region is not None:
                horn_raw = self.horn_analyzer.analyze_horn_contours(head_region)
                horn_features = self.horn_analyzer.classify_horn_type(horn_raw)
                confidence_scores['horn'] = min(1.0, sum(horn_features.values()))
            else:
                confidence_scores['horn'] = 0.0
                
        except Exception as e:
            print(f"Horn analysis failed: {e}")
            confidence_scores['horn'] = 0.0
        
        try:
            # Coat analysis
            coat_features = self.coat_analyzer.analyze_coat(image)
            confidence_scores['coat'] = min(1.0, sum(v for k, v in coat_features.items() if not k.startswith('pattern')))
        except Exception as e:
            print(f"Coat analysis failed: {e}")
            confidence_scores['coat'] = 0.0
        
        try:
            # Body analysis
            body_raw = self.body_analyzer.analyze_body_proportions(image)
            body_features = self.body_analyzer.classify_body_type(body_raw)
            confidence_scores['body'] = min(1.0, sum(body_features.values()))
        except Exception as e:
            print(f"Body analysis failed: {e}")
            confidence_scores['body'] = 0.0
        
        return MorphologicalFeatures(
            horn_features=horn_features,
            coat_features=coat_features,
            body_features=body_features,
            confidence_scores=confidence_scores
        )
    
    def compute_breed_scores(self, morph_features: MorphologicalFeatures) -> Dict[str, float]:
        """Compute breed-specific scores based on morphological features"""
        breed_scores = {}
        
        for breed in self.breed_feature_weights.keys():
            weights = self.breed_feature_weights[breed]
            total_score = 0.0
            total_weight = 0.0
            
            # Horn contribution
            if morph_features.confidence_scores.get('horn', 0) > 0.1:
                horn_score = 0.0
                for horn_type, score in morph_features.horn_features.items():
                    if horn_type in self.horn_analyzer.horn_patterns:
                        breed_match = self.horn_analyzer.horn_patterns[horn_type].get(breed.lower(), 0.0)
                        horn_score += score * breed_match
                
                total_score += horn_score * weights['horn']
                total_weight += weights['horn']
            
            # Coat contribution
            if morph_features.confidence_scores.get('coat', 0) > 0.1:
                coat_score = 0.0
                for color_type, score in morph_features.coat_features.items():
                    if color_type in self.coat_analyzer.breed_colors:
                        breed_match = self.coat_analyzer.breed_colors[color_type].get(breed.lower(), 0.0)
                        coat_score += score * breed_match
                
                total_score += coat_score * weights['coat']
                total_weight += weights['coat']
            
            # Body contribution
            if morph_features.confidence_scores.get('body', 0) > 0.1:
                body_score = 0.0
                for body_type, score in morph_features.body_features.items():
                    if body_type in self.body_analyzer.breed_body_types:
                        breed_match = self.body_analyzer.breed_body_types[body_type].get(breed.lower(), 0.0)
                        body_score += score * breed_match
                
                total_score += body_score * weights['body']
                total_weight += weights['body']
            
            # Normalize by total weight
            if total_weight > 0:
                breed_scores[breed] = total_score / total_weight
            else:
                breed_scores[breed] = 0.0
        
        return breed_scores