#!/usr/bin/env python3
"""
🚀 ULTIMATE CATTLE BREED RECOGNITION LAUNCHER 🚀
=================================================

Quick launcher for the ultimate cattle breed recognition system.
This script provides easy deployment options for the production app.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'opencv-python', 
        'pillow', 'numpy', 'scipy', 'scikit-learn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("\n🎉 All dependencies satisfied!")
    return True

def check_models():
    """Check if all required model files exist"""
    required_files = [
        'models/enhanced_11breed_model.pth',
        'models/enhanced_11breed_info.json', 
        'models/prototypes_maximum_11breed.pkl',
        'smart_fusion_fixed.py',
        'ensemble_optimizer.py'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            missing.append(file_path)
            print(f"❌ {file_path} - MISSING")
    
    if missing:
        print(f"\n⚠️  Missing model files: {', '.join(missing)}")
        return False
    
    print("\n🎉 All model files present!")
    return True

def launch_ultimate_app():
    """Launch the ultimate cattle breed recognition app"""
    print("\n🚀 Launching Ultimate Cattle Breed Recognition App...")
    print("📊 Expected Performance: 87.1% accuracy with 11-breed support")
    print("🔗 The app will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'app_ultimate.py',
            '--theme.base', 'light',
            '--theme.primaryColor', '#2E8B57',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Ultimate app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching app: {e}")

def show_menu():
    """Show the main menu"""
    print("""
🐄 ULTIMATE CATTLE BREED RECOGNITION SYSTEM 🐄
===============================================

🏆 FEATURES:
✅ 87.1% accuracy with advanced AI ensemble
✅ 11 cattle breeds including Nagpuri support  
✅ Dual prediction methods (Neural + Smart Fusion)
✅ Real-time morphological analysis
✅ Production-ready Streamlit interface

🎯 SUPPORTED BREEDS:
Bhadawari, Gir, Jaffarbadi, Kankrej, Mehsana, 
Murrah, Nagpuri, Ongole, Sahiwal, Surti, Tharparkar

📋 MENU OPTIONS:
1. 🚀 Launch Ultimate App (Recommended)
2. 🔍 Check Dependencies
3. 📁 Check Model Files
4. 📖 Show System Info
5. ❌ Exit

Choose an option (1-5): """)

def show_system_info():
    """Show comprehensive system information"""
    print("\n📊 ULTIMATE SYSTEM SPECIFICATIONS")
    print("=" * 50)
    
    print("🤖 AI COMPONENTS:")
    print("• Enhanced ResNet50 with 11-breed classification")
    print("• Smart Fusion with morphological analysis")
    print("• Advanced ensemble with optimized weights")
    print("• CNN feature extraction from trained backbone")
    
    print("\n🎯 PERFORMANCE METRICS:")
    print("• Overall Accuracy: 87.1%")
    print("• Perfect Breeds (100%): Kankrej, Nagpuri, Ongole")
    print("• Excellent Performance (91%+): Mehsana, Murrah")
    print("• Method Agreement: 79.5%")
    
    print("\n⚙️ ENSEMBLE CONFIGURATION:")
    print("• Trained Model Weight: 0.4")
    print("• Smart Fusion Weight: 0.6") 
    print("• Confidence Boost: 0.1")
    print("• Agreement Bonus: 0.2")
    
    print("\n🔬 FUSION STRATEGIES:")
    print("• CNN_DOMINANT: High confidence neural predictions")
    print("• MORPHO_TIEBREAK: Morphological analysis for ties")
    print("• LIGHT_FUSION: Balanced multi-modal approach")
    
    print("\n📂 SYSTEM FILES:")
    print("• Main App: app_ultimate.py")
    print("• Smart Fusion: smart_fusion_fixed.py") 
    print("• Ensemble: ensemble_optimizer.py")
    print("• Model: models/enhanced_11breed_model.pth")
    print("• Prototypes: models/prototypes_maximum_11breed.pkl")

def main():
    """Main launcher application"""
    while True:
        show_menu()
        
        try:
            choice = input().strip()
            
            if choice == '1':
                print("\n🔍 Pre-flight checks...")
                if check_dependencies() and check_models():
                    launch_ultimate_app()
                else:
                    print("\n❌ System not ready. Please resolve missing components.")
                    input("\nPress Enter to continue...")
                    
            elif choice == '2':
                print("\n🔍 CHECKING DEPENDENCIES")
                print("=" * 30)
                check_dependencies()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                print("\n📁 CHECKING MODEL FILES")
                print("=" * 30)
                check_models()
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                show_system_info()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                print("\n👋 Thanks for using Ultimate Cattle Recognition!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter 1-5.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting Ultimate Launcher")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()