#!/usr/bin/env python3
"""
UPDATED SMART FUSION WITH NAGPURI

Add Nagpuri to existing smart fusion system wit            # Morphological features
            image = cv2.imread(img_path)
            if image is not None:
                morphological_features = morphology_extractor.extract_all_features(image)
                # Convert to numerical array (you may need to adjust this based on the MorphologicalFeatures structure)
                morph_array = np.array([
                    morphological_features.area,
                    morphological_features.perimeter,
                    morphological_features.aspect_ratio,
                    morphological_features.extent,
                    morphological_features.solidity,
                    morphological_features.compactness,
                    morphological_features.circularity,
                    morphological_features.rectangularity,
                    morphological_features.symmetry,
                    morphological_features.eccentricity,
                    morphological_features.major_axis_length,
                    morphological_features.minor_axis_length,
                ])
                # Add head features
                if hasattr(morphological_features, 'head_features') and morphological_features.head_features:
                    head_array = np.array([
                        morphological_features.head_features.head_width,
                        morphological_features.head_features.head_height,
                        morphological_features.head_features.head_aspect_ratio,
                        morphological_features.head_features.forehead_width,
                        morphological_features.head_features.eye_distance,
                        morphological_features.head_features.nose_width,
                        morphological_features.head_features.mouth_width,
                    ])
                    morph_array = np.concatenate([morph_array, head_array])
                
                nagpuri_morph_features.append(morph_array)changing the core architecture.
Updates prototypes to include all 11 breeds for use with infer_smart_fusion.py.
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import cv2

# Add utils to path
sys.path.append('.')
from utils import (
    FeatureExtractor,
    get_device,
    get_transforms,
    load_and_preprocess_image,
    extract_features,
    compute_prototypes,
    compute_similarities,
    discover_breeds,
    get_breed_images,
    split_dataset
)
from utils.morphology import MorphologicalFeatureExtractor

def load_existing_prototypes():
    """Load existing 10-breed prototypes"""
    
    prototype_path = 'models/prototypes_maximum_10breed.pkl'
    
    if os.path.exists(prototype_path):
        with open(prototype_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded existing prototypes with {len(data['breeds'])} breeds")
        print(f"📋 Current breeds: {', '.join(data['breeds'])}")
        return data
    else:
        print("❌ No existing prototypes found")
        return None

def add_nagpuri_to_prototypes():
    """Add Nagpuri breed to existing prototypes"""
    
    print("🔄 Adding Nagpuri to existing smart fusion system...")
    
    # Load existing data
    existing_data = load_existing_prototypes()
    if existing_data is None:
        print("❌ Cannot proceed without existing prototypes")
        return False
    
    # Check if Nagpuri already exists
    if 'Nagpuri' in existing_data['breeds']:
        print("✅ Nagpuri already in prototypes!")
        return True
    
    # Setup device and model
    device = get_device()
    
    # Load the existing model configuration
    model_name = existing_data.get('model_name', 'resnet50')
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model_name, pretrained=True)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Create morphological extractor
    morph_extractor = MorphologicalFeatureExtractor()
    
    # Get transforms
    transform = get_transforms()
    
    print("📊 Extracting features for Nagpuri breed...")
    
    # Get Nagpuri images
    nagpuri_images = get_breed_images("dataset", "Nagpuri")
    print(f"🖼️  Found {len(nagpuri_images)} Nagpuri images")
    
    if len(nagpuri_images) == 0:
        print("❌ No Nagpuri images found in dataset/Nagpuri/")
        return False
    
    # Extract CNN features for Nagpuri
    nagpuri_cnn_features = []
    nagpuri_morph_features = []
    
    print("🧠 Extracting CNN features...")
    for img_path in tqdm(nagpuri_images, desc="CNN Features"):
        try:
            # CNN features
            image_tensor = load_and_preprocess_image(img_path, transform)
            image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            # Extract features directly from model
            with torch.no_grad():
                features = feature_extractor(image_tensor).squeeze(0)
            nagpuri_cnn_features.append(features.cpu().numpy())
            
            # Morphological features
            image = cv2.imread(img_path)
            if image is not None:
                morphological_features = morph_extractor.extract_all_features(image)
                # Convert to numerical array (you may need to adjust this based on the MorphologicalFeatures structure)
                morph_array = np.array([
                    morphological_features.area,
                    morphological_features.perimeter,
                    morphological_features.aspect_ratio,
                    morphological_features.extent,
                    morphological_features.solidity,
                    morphological_features.compactness,
                    morphological_features.circularity,
                    morphological_features.rectangularity,
                    morphological_features.symmetry,
                    morphological_features.eccentricity,
                    morphological_features.major_axis_length,
                    morphological_features.minor_axis_length,
                ])
                # Add head features
                if hasattr(morphological_features, 'head_features') and morphological_features.head_features:
                    head_array = np.array([
                        morphological_features.head_features.head_width,
                        morphological_features.head_features.head_height,
                        morphological_features.head_features.head_aspect_ratio,
                        morphological_features.head_features.forehead_width,
                        morphological_features.head_features.eye_distance,
                        morphological_features.head_features.nose_width,
                        morphological_features.head_features.mouth_width,
                    ])
                    morph_array = np.concatenate([morph_array, head_array])
                
                nagpuri_morph_features.append(morph_array)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    if len(nagpuri_cnn_features) == 0:
        print("❌ No valid features extracted for Nagpuri")
        return False

    # Compute Nagpuri prototype
    nagpuri_cnn_features = np.array(nagpuri_cnn_features)
    nagpuri_cnn_prototype = np.mean(nagpuri_cnn_features, axis=0)
    
    print(f"✅ Nagpuri CNN prototype shape: {nagpuri_cnn_prototype.shape}")
    
    # Add Nagpuri to existing prototypes
    print("📊 Adding Nagpuri to existing prototypes...")
    
    # Add Nagpuri CNN prototype directly to the prototypes dict
    existing_data['prototypes']['Nagpuri'] = torch.tensor(nagpuri_cnn_prototype)
    
    # Update breeds list
    existing_data['breeds'].append('Nagpuri')
    
    # Update metadata
    existing_data['metadata']['breeds_count'] = len(existing_data['breeds'])
    existing_data['metadata']['last_updated'] = datetime.now().isoformat()
    existing_data['metadata']['nagpuri_images'] = len(nagpuri_images)
    existing_data['metadata']['nagpuri_integration'] = 'Added via update_prototypes_with_nagpuri.py'    # Create updated data structure
    updated_data = {
        'prototypes': {
            'cnn': updated_cnn_prototypes,
            'morphological': updated_morph_prototypes
        },
        'breeds': updated_breeds,
        'model_name': existing_data['model_name'],
        'config': existing_data['config'],
        'metadata': {
            **existing_data.get('metadata', {}),
            'nagpuri_added': time.strftime('%Y-%m-%d %H:%M:%S'),
            'nagpuri_images': len(nagpuri_images),
            'total_breeds': len(updated_breeds)
        },
        'augmentation_rounds': existing_data.get('augmentation_rounds', 1),
        'dataset_splits': existing_data.get('dataset_splits', {}),
        'total_features': existing_data.get('total_features', 0),
        'optimization_type': existing_data.get('optimization_type', 'maximum_discrimination')
    }
    
    # Save updated prototypes
    updated_path = 'models/prototypes_enhanced.pkl'
    with open(updated_path, 'wb') as f:
        pickle.dump(updated_data, f)
    
    print(f"✅ Updated prototypes saved to: {updated_path}")
    print(f"🧬 Total breeds: {len(updated_breeds)}")
    print(f"📋 All breeds: {', '.join(updated_breeds)}")
    
    # Also update the maximum prototypes file for compatibility
    maximum_path = 'models/prototypes_maximum_11breed.pkl'
    with open(maximum_path, 'wb') as f:
        pickle.dump(updated_data, f)
    
    print(f"✅ Also saved as: {maximum_path}")
    
    return True

def update_smart_fusion_config():
    """Update smart fusion configuration for 11 breeds"""
    
    config = {
        'breeds': ['Bhadawari', 'Gir', 'Jaffarbadi', 'Kankrej', 'Mehsana', 'Murrah', 
                   'Nagpuri', 'Ongole', 'Sahiwal', 'Surti', 'Tharparkar'],
        'num_classes': 11,
        'model_path': 'models/prototypes_enhanced.pkl',
        'fusion_strategies': {
            'CNN_DOMINANT': 'confidence >= 0.85',
            'MORPHO_TIEBREAK': 'confidence_margin <= 0.01', 
            'LIGHT_FUSION': 'balanced approach'
        },
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'nagpuri_integration': True
    }
    
    with open('models/smart_fusion_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Smart fusion config updated for 11 breeds")

def test_nagpuri_integration():
    """Test Nagpuri integration with existing tools"""
    
    print("\n🧪 Testing Nagpuri integration...")
    
    # Find a Nagpuri test image
    nagpuri_images = get_breed_images("dataset", "Nagpuri")
    
    if not nagpuri_images:
        print("❌ No Nagpuri images found for testing")
        return
    
    test_image = nagpuri_images[0]
    print(f"🔍 Testing with: {test_image}")
    
    # Test with updated smart fusion
    try:
        from infer_smart_fusion import smart_enhanced_inference
        
        # Update the inference to use new prototypes
        result = smart_enhanced_inference(
            test_image, 
            model_path='models/prototypes_enhanced.pkl'
        )
        
        print(f"✅ Smart fusion test successful!")
        print(f"🎯 Predicted: {result.get('predicted_breed', 'Unknown')}")
        print(f"🎪 Confidence: {result.get('confidence', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Smart fusion test failed: {e}")
        return False

def create_updated_inference_script():
    """Create updated inference script for 11 breeds"""
    
    script_content = '''#!/usr/bin/env python3
"""
Updated Smart Fusion Inference with Nagpuri

Usage: python infer_smart_fusion_11breed.py <image_path>
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def main():
    if len(sys.argv) != 2:
        print("Usage: python infer_smart_fusion_11breed.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Import inference function
    from infer_smart_fusion import smart_enhanced_inference
    
    # Use updated prototypes with Nagpuri
    result = smart_enhanced_inference(
        image_path, 
        model_path='models/prototypes_enhanced.pkl'
    )
    
    # Display results
    print(f"\\n🎯 PREDICTION RESULTS")
    print("=" * 40)
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"🧬 Predicted Breed: {result.get('predicted_breed', 'Unknown')}")
    print(f"🎪 Confidence: {result.get('confidence', 0):.3f}")
    print(f"🔧 Strategy: {result.get('fusion_strategy', 'Unknown')}")
    
    # Show top predictions if available
    if 'top_predictions' in result:
        print(f"\\n📊 Top 3 Predictions:")
        for i, (breed, conf) in enumerate(result['top_predictions'][:3]):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {rank} {breed}: {conf:.3f}")

if __name__ == "__main__":
    main()
'''
    
    with open('infer_smart_fusion_11breed.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created: infer_smart_fusion_11breed.py")

def main():
    """Main function to add Nagpuri to smart fusion system"""
    
    print("🎯 INTEGRATING NAGPURI INTO SMART FUSION SYSTEM")
    print("=" * 60)
    print("📋 Updating existing prototypes without changing core system")
    print()
    
    # Add Nagpuri to prototypes
    success = add_nagpuri_to_prototypes()
    
    if not success:
        print("❌ Failed to add Nagpuri to prototypes")
        return
    
    # Update configuration
    update_smart_fusion_config()
    
    # Create updated inference script
    create_updated_inference_script()
    
    # Test integration
    test_nagpuri_integration()
    
    print(f"\n🏆 NAGPURI INTEGRATION COMPLETE!")
    print("=" * 60)
    print("✅ Nagpuri added to smart fusion system")
    print("🧬 Total breeds: 11 (including Nagpuri)")
    print("💾 Updated prototypes: models/prototypes_enhanced.pkl")
    print("🔧 Config updated: models/smart_fusion_config.json")
    print()
    print("🧪 Test with existing tools:")
    print("   python test_accuracy.py")
    print("   python infer_smart_fusion.py <image> --model-path models/prototypes_enhanced.pkl")
    print("   python infer_smart_fusion_11breed.py <image>")
    print("=" * 60)

if __name__ == "__main__":
    main()