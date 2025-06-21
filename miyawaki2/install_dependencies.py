#!/usr/bin/env python3
"""
Installation script for Miyawaki2 dependencies
Automatically installs required packages with fallbacks
"""

import subprocess
import sys
import importlib

def install_package(package_name, pip_name=None):
    """Install a package using pip"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def install_git_package(package_name, git_url):
    """Install a package from git repository"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name} from git...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{git_url}"])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    """Main installation function"""
    print("🚀 Installing Miyawaki2 Dependencies...")
    print("=" * 50)
    
    # Core dependencies
    core_packages = [
        ("torch", "torch>=1.9.0"),
        ("torchvision", "torchvision>=0.10.0"),
        ("scipy", "scipy>=1.9.0"),
        ("numpy", "numpy>=1.21.0"),
        ("matplotlib", "matplotlib>=3.5.0"),
    ]
    
    # Advanced dependencies
    advanced_packages = [
        ("diffusers", "diffusers>=0.21.0"),
        ("transformers", "transformers>=4.21.0"),
        ("torchmetrics", "torchmetrics>=0.11.0"),
        ("lpips", "lpips>=0.1.4"),
        ("skimage", "scikit-image>=0.19.0"),
    ]
    
    # Git dependencies
    git_packages = [
        ("clip", "https://github.com/openai/CLIP.git"),
    ]
    
    print("📦 Installing core dependencies...")
    core_success = 0
    for package, pip_name in core_packages:
        if install_package(package, pip_name):
            core_success += 1
    
    print(f"\n📊 Core packages: {core_success}/{len(core_packages)} installed")
    
    print("\n📦 Installing advanced dependencies...")
    advanced_success = 0
    for package, pip_name in advanced_packages:
        if install_package(package, pip_name):
            advanced_success += 1
    
    print(f"\n📊 Advanced packages: {advanced_success}/{len(advanced_packages)} installed")
    
    print("\n📦 Installing git dependencies...")
    git_success = 0
    for package, git_url in git_packages:
        if install_git_package(package, git_url):
            git_success += 1
    
    print(f"\n📊 Git packages: {git_success}/{len(git_packages)} installed")
    
    # Summary
    total_packages = len(core_packages) + len(advanced_packages) + len(git_packages)
    total_success = core_success + advanced_success + git_success
    
    print("\n" + "=" * 50)
    print(f"🎯 INSTALLATION SUMMARY")
    print(f"Total packages: {total_success}/{total_packages} installed")
    
    if total_success == total_packages:
        print("✅ All dependencies installed successfully!")
        print("🚀 Ready to run Miyawaki2 implementation")
    else:
        print("⚠️  Some dependencies failed to install")
        print("💡 The implementation will use fallbacks for missing packages")
    
    print("\n🔧 Next steps:")
    print("1. cd miyawaki2")
    print("2. python test_dependencies.py")
    print("3. python main.py  # (once integration is complete)")

if __name__ == "__main__":
    main()
