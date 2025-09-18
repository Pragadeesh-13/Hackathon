#!/usr/bin/env python3
"""
Environment Test Script

Quick test to check if all dependencies are properly installed.
Run this before using the main scripts.

Usage:
    python test_environment.py
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def check_torch_gpu():
    """Check PyTorch GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU available: {gpu_name}")
        else:
            print("💻 GPU not available, using CPU")
        return True
    except ImportError:
        return False

def main():
    print("=" * 50)
    print("ENVIRONMENT TEST")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print()
    
    print("Testing core dependencies:")
    
    # Core ML dependencies
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
    ]
    
    core_success = True
    for module, package in deps:
        if not test_import(module, package):
            core_success = False
    
    print("\nTesting visualization dependencies:")
    
    viz_deps = [
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("cv2", "OpenCV"),
        ("plotly", "Plotly"),
    ]
    
    viz_success = True
    for module, package in viz_deps:
        if not test_import(module, package):
            viz_success = False
    
    print("\nTesting web interface:")
    streamlit_success = test_import("streamlit", "Streamlit")
    
    print("\nTesting GPU support:")
    check_torch_gpu()
    
    print("\nTesting project utilities:")
    utils_success = test_import("utils", "Project utilities")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if core_success:
        print("✅ Core ML dependencies: Ready")
    else:
        print("❌ Core ML dependencies: Missing packages")
        print("   Run: pip install torch torchvision numpy pillow scikit-learn pandas")
    
    if viz_success:
        print("✅ Visualization dependencies: Ready")
    else:
        print("❌ Visualization dependencies: Missing packages")
        print("   Run: pip install matplotlib seaborn opencv-python plotly")
    
    if streamlit_success:
        print("✅ Web interface: Ready")
    else:
        print("❌ Web interface: Missing Streamlit")
        print("   Run: pip install streamlit")
    
    if utils_success:
        print("✅ Project utilities: Ready")
    else:
        print("❌ Project utilities: Check project structure")
    
    if core_success and viz_success and streamlit_success and utils_success:
        print("\n🎉 Environment is ready! You can now run:")
        print("   python prototype.py")
        print("   python infer.py <image_path>")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some dependencies are missing. Install them using:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()