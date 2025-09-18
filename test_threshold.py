#!/usr/bin/env python3
"""
Test script to demonstrate threshold-based breed recognition.
This script tests the system with different confidence thresholds.
"""

import os
import subprocess
import sys

def test_inference_with_thresholds():
    """Test inference with different confidence thresholds."""
    
    print("=" * 70)
    print("TESTING THRESHOLD-BASED BREED RECOGNITION")
    print("=" * 70)
    
    # Find a test image
    test_image = None
    murrah_dir = "dataset/Murrah"
    
    if os.path.exists(murrah_dir):
        for filename in os.listdir(murrah_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                test_image = os.path.join(murrah_dir, filename)
                break
    
    if not test_image:
        print("❌ No test images found in dataset/Murrah/")
        return
    
    print(f"Testing with image: {test_image}")
    print(f"Expected: Should be identified as Murrah with high confidence")
    
    # Test with different thresholds
    thresholds = [0.9, 0.7, 0.5, 0.3]
    
    for threshold in thresholds:
        print(f"\n" + "🔬" + " " * 5 + f"TESTING WITH THRESHOLD: {threshold}" + " " * 5 + "🔬")
        print("-" * 70)
        
        try:
            # Run inference with specific threshold
            cmd = [
                "python", "infer.py", 
                test_image,
                "--confidence-threshold", str(threshold),
                "--show-breed-info"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error running inference: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("• High threshold (0.9): Should identify Murrah only if very confident")
    print("• Medium threshold (0.7): Should identify Murrah with good confidence")  
    print("• Low threshold (0.5): Should identify Murrah with reasonable confidence")
    print("• Very low threshold (0.3): Should identify Murrah even with low confidence")
    print("\nFor testing with non-Murrah images:")
    print("1. Add images of other animals/objects to test rejection")
    print("2. Use threshold around 0.6-0.7 for good discrimination")
    print("3. Images below threshold will be classified as 'Unknown'")

def show_usage():
    """Show usage instructions."""
    print("\n" + "📋" + " " * 5 + "USAGE INSTRUCTIONS" + " " * 5 + "📋")
    print("=" * 70)
    print("1. Test with Murrah image (should be recognized):")
    print("   python infer.py dataset/Murrah/murrah_0002.jpg --confidence-threshold 0.6")
    print("")
    print("2. Test with non-Murrah image (should be rejected):")
    print("   python infer.py path/to/dog.jpg --confidence-threshold 0.6")
    print("")
    print("3. Adjust threshold for sensitivity:")
    print("   --confidence-threshold 0.8  # Strict (fewer false positives)")
    print("   --confidence-threshold 0.5  # Lenient (more detections)")
    print("")
    print("4. Use the Streamlit app for interactive testing:")
    print("   streamlit run app.py")
    print("   (Use the confidence threshold slider in the sidebar)")

if __name__ == "__main__":
    test_inference_with_thresholds()
    show_usage()