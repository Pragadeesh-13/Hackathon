#!/usr/bin/env python3
"""
HACKATHON SETUP SCRIPT

Automated setup for competition-grade training environment.
Ensures all directories, dependencies, and configurations are ready.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_gpu_availability():
    """Check GPU availability"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            print(f"🔥 CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower but functional)")
            return True  # Changed to True since CPU training is still valid
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0"
    ]
    
    for package in requirements:
        try:
            print(f"📥 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "models",
        "results",
        "exports",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified: {directory}/")
    
    return True

def verify_dataset():
    """Verify hackathon dataset"""
    print("🧬 Verifying hackathon dataset...")
    
    # Check if analysis exists
    if not os.path.exists('hackathon_analysis.json'):
        print("❌ Hackathon analysis not found. Running dataset analysis...")
        try:
            subprocess.check_call([sys.executable, "analyze_hackathon_dataset.py"])
            print("✅ Dataset analysis completed")
        except subprocess.CalledProcessError:
            print("❌ Failed to analyze dataset")
            return False
    
    # Load and verify analysis
    try:
        with open('hackathon_analysis.json', 'r') as f:
            analysis = json.load(f)
        
        breeds = analysis['breeds_analysis']
        total_images = sum(breed['count'] for breed in breeds.values())
        
        print(f"✅ Dataset verified:")
        print(f"   📊 {len(breeds)} breeds detected")
        print(f"   🖼️  {total_images} total images")
        print(f"   🎯 Target accuracy: {analysis['hackathon_config']['hackathon_optimizations']['target_accuracy']*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False

def create_utils_module():
    """Ensure utils module exists with all necessary functions"""
    print("🛠️  Checking utils module...")
    
    utils_path = Path("utils")
    if not utils_path.exists():
        print("❌ Utils module not found")
        return False
    
    required_files = [
        "__init__.py",
        "feature_extractor.py",
        "smart_fusion.py"
    ]
    
    for file in required_files:
        file_path = utils_path / file
        if not file_path.exists():
            print(f"❌ Missing utils file: {file}")
            return False
    
    print("✅ Utils module verified")
    return True

def run_system_checks():
    """Run comprehensive system checks"""
    print("🔍 Running system checks...")
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu_availability),
        ("Required Packages", install_requirements),
        ("Directory Structure", create_directory_structure),
        ("Utils Module", create_utils_module),
        ("Dataset Verification", verify_dataset)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"🧪 {check_name}")
        print('='*50)
        
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
            all_passed = False
    
    return all_passed, results

def display_hackathon_readiness(results):
    """Display hackathon readiness status"""
    print("\n" + "="*60)
    print("🏆 HACKATHON READINESS REPORT")
    print("="*60)
    
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {check}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 SYSTEM READY FOR HACKATHON TRAINING!")
        print("="*60)
        print("Next steps:")
        print("1. 🚀 Run: python train_hackathon_pipeline.py")
        print("2. 🧪 Test: python infer_hackathon_ensemble.py")
        print("3. 🌐 Launch: streamlit run app_hackathon.py")
        print("="*60)
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("Please resolve the failed checks above before proceeding.")
    
    return all_passed

def create_quick_start_guide():
    """Create quick start guide"""
    guide_content = """
# 🏆 HACKATHON QUICK START GUIDE

## System Status
✅ Setup completed successfully
🎯 Target: 95%+ accuracy competition system

## Training Pipeline
1. **Start Training**: `python train_hackathon_pipeline.py`
   - Trains ensemble of ResNet50 models
   - Uses advanced augmentation
   - Targets 95%+ accuracy

2. **Test Inference**: `python infer_hackathon_ensemble.py`
   - Tests ensemble voting methods
   - Validates accuracy on samples
   - Benchmarks inference speed

3. **Launch Web App**: `streamlit run app_hackathon.py`
   - Interactive breed recognition
   - Real-time confidence display
   - Ensemble voting visualization

## Dataset Information
- **Breeds**: 11 (including Sahiwal)
- **Images**: 600+ total
- **Quality**: Competition-grade dataset

## Model Architecture
- **Base**: ResNet50 (pretrained)
- **Ensemble**: Multiple models voting
- **Fusion**: Smart CNN-first approach
- **Accuracy**: 95%+ target

## Competition Features
- ⚡ Fast inference (< 100ms)
- 🎯 High confidence scoring
- 📊 Detailed breed analysis
- 🏆 Hackathon-optimized training

## Support
- Check `logs/` for training progress
- Monitor `results/` for outputs
- Use `temp/` for temporary files

Good luck with your hackathon! 🚀
"""
    
    with open("HACKATHON_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("📖 Created HACKATHON_GUIDE.md")

def main():
    """Main setup function"""
    print("🏆 HACKATHON SETUP INITIALIZING")
    print("="*60)
    
    # Run system checks
    all_passed, results = run_system_checks()
    
    # Display results
    display_hackathon_readiness(results)
    
    if all_passed:
        create_quick_start_guide()
        
        print("\n🎊 HACKATHON ENVIRONMENT READY!")
        print("🚀 You can now start training with maximum accuracy!")
    else:
        print("\n❌ Please resolve setup issues before proceeding.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)