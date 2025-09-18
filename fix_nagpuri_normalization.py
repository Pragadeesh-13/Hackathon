#!/usr/bin/env python3
"""
Fix Nagpuri Prototype Normalization

The Nagpuri prototype was added without proper normalization.
This script normalizes it to match the existing prototypes.
"""

import pickle
import torch

def fix_nagpuri_normalization():
    """Normalize Nagpuri prototype to unit norm like others"""
    
    print("🔧 FIXING NAGPURI PROTOTYPE NORMALIZATION")
    print("=" * 50)
    
    # Load prototypes
    prototypes_path = 'models/prototypes_maximum_11breed.pkl'
    
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check current norms
    print("📊 Current prototype norms:")
    for breed, tensor in data['prototypes'].items():
        norm = torch.norm(tensor).item()
        print(f"  {breed}: {norm:.6f}")
    
    # Normalize Nagpuri prototype
    nagpuri_tensor = data['prototypes']['Nagpuri']
    nagpuri_norm = torch.norm(nagpuri_tensor)
    normalized_nagpuri = nagpuri_tensor / nagpuri_norm
    
    # Update the prototype
    data['prototypes']['Nagpuri'] = normalized_nagpuri
    
    print(f"\n✅ Normalized Nagpuri: {torch.norm(normalized_nagpuri).item():.6f}")
    
    # Save all versions
    save_paths = [
        'models/prototypes_maximum_10breed.pkl',
        'models/prototypes_enhanced.pkl', 
        'models/prototypes_maximum_11breed.pkl'
    ]
    
    for save_path in save_paths:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Updated: {save_path}")
    
    print("\n🎉 Nagpuri prototype normalization fixed!")
    print("📋 Ready to test with proper similarity scores")

if __name__ == "__main__":
    fix_nagpuri_normalization()