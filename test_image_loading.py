#!/usr/bin/env python3
"""
Test script to verify image loading functionality.
"""

import io
from PIL import Image
import numpy as np

def test_image_loading():
    """Test image loading with different scenarios."""
    print("Testing image loading scenarios...")
    
    # Create a test image in memory (simulating uploaded file)
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Save to bytes buffer (simulating Streamlit UploadedFile)
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    buffer.seek(0)  # Reset to beginning
    
    # Test loading from buffer
    try:
        loaded_image = Image.open(buffer).convert('RGB')
        print("✅ Successfully loaded image from buffer")
        print(f"   Image size: {loaded_image.size}")
        print(f"   Image mode: {loaded_image.mode}")
    except Exception as e:
        print(f"❌ Failed to load image from buffer: {e}")
        return False
    
    # Test with transforms
    try:
        from utils.model import get_transforms
        transform = get_transforms(augment=False)
        tensor = transform(loaded_image)
        print(f"✅ Successfully applied transforms: {tensor.shape}")
    except Exception as e:
        print(f"❌ Failed to apply transforms: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("IMAGE LOADING TEST")
    print("=" * 50)
    
    success = test_image_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Image loading should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    print("=" * 50)