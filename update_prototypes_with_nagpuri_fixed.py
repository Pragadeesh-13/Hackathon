#!/usr/bin/env python3
"""
Add Nagpuri to Existing Smart Fusion System

This script adds Nagpuri breed to the existing 10-breed prototypes 
without changing the core smart fusion architecture.
"""

import os
import pickle
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

# Import existing utilities
from utils import (
    FeatureExtractor,
    get_device,
    get_transforms,
    load_and_preprocess_image
)

def main():
    """Main function to add Nagpuri to existing prototypes"""
    
    print("🎯 INTEGRATING NAGPURI INTO SMART FUSION SYSTEM")
    print("=" * 60)
    print("📋 Updating existing prototypes without changing core system")
    print()
    
    # Add Nagpuri to prototypes
    success = add_nagpuri_to_prototypes()
    
    if success:
        print()
        print("🎉 SUCCESS! Nagpuri integration completed")
        print("✅ Updated prototypes saved")
        print("📋 Ready to use with infer_smart_fusion.py and test_accuracy.py")
        print()
        print("🔄 Next steps:")
        print("  1. Run: python test_accuracy.py")
        print("  2. Test: python infer_smart_fusion.py <image_path>")
    else:
        print("❌ Failed to integrate Nagpuri")
        
def add_nagpuri_to_prototypes():
    """Add Nagpuri breed to existing prototypes"""
    
    # Load existing prototypes
    prototypes_path = 'models/prototypes_maximum_10breed.pkl'
    
    if not os.path.exists(prototypes_path):
        print(f"❌ Prototypes file not found: {prototypes_path}")
        return False
    
    print("🔄 Adding Nagpuri to existing smart fusion system...")
    
    with open(prototypes_path, 'rb') as f:
        existing_data = pickle.load(f)
    
    print(f"✅ Loaded existing prototypes with {len(existing_data['breeds'])} breeds")
    print(f"📋 Current breeds: {', '.join(existing_data['breeds'])}")
    
    # Check if Nagpuri already exists
    if 'Nagpuri' in existing_data['breeds']:
        print("⚠️  Nagpuri already exists in prototypes")
        return True
    
    # Set up device and model
    device = get_device()
    print(f"Using {device}")
    
    # Create feature extractor
    model_name = existing_data.get('model_name', 'resnet50')
    feature_extractor = FeatureExtractor(model_name=model_name, pretrained=True)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Get transform
    transform = get_transforms(augment=False)
    
    # Find Nagpuri images
    dataset_path = 'dataset'
    nagpuri_path = os.path.join(dataset_path, 'Nagpuri')
    
    if not os.path.exists(nagpuri_path):
        print(f"❌ Nagpuri dataset not found: {nagpuri_path}")
        return False
    
    # Get Nagpuri image paths
    nagpuri_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        nagpuri_images.extend(list(Path(nagpuri_path).glob(ext)))
    
    nagpuri_images = [str(img) for img in nagpuri_images]
    
    print(f"📊 Extracting features for Nagpuri breed...")
    print(f"🖼️  Found {len(nagpuri_images)} Nagpuri images")
    
    if len(nagpuri_images) == 0:
        print("❌ No Nagpuri images found")
        return False
    
    # Extract CNN features for Nagpuri
    nagpuri_cnn_features = []
    
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
    existing_data['metadata']['nagpuri_integration'] = 'Added via update_prototypes_with_nagpuri.py'
    
    # Save updated prototypes with multiple names for compatibility
    save_paths = [
        'models/prototypes_maximum_10breed.pkl',  # Update original
        'models/prototypes_enhanced.pkl',         # Alternative name
        'models/prototypes_maximum_11breed.pkl'   # New 11-breed name
    ]
    
    for save_path in save_paths:
        with open(save_path, 'wb') as f:
            pickle.dump(existing_data, f)
        print(f"✅ Updated prototypes saved to: {save_path}")
    
    print(f"🧬 Total breeds: {len(existing_data['breeds'])}")
    print(f"📋 All breeds: {', '.join(existing_data['breeds'])}")
    
    return True

if __name__ == "__main__":
    main()