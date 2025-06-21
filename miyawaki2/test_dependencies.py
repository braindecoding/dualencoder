#!/usr/bin/env python3
"""
Test script untuk memverifikasi semua dependencies Miyawaki2
Checks imports dan basic functionality
"""

import importlib

def test_import(module_name, description=""):
    """Test importing a module"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name:<20} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<20} - {description} (Error: {e})")
        return False

def test_torch_functionality():
    """Test basic PyTorch functionality"""
    try:
        import torch
        import torch.nn as nn

        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        _ = x + y  # Test addition

        # Test neural network
        model = nn.Linear(3, 1)
        _ = model(x)  # Test forward pass

        print("✅ PyTorch basic functionality working")
        return True
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_clip_functionality():
    """Test CLIP functionality"""
    try:
        # Now that we renamed clip.py to clipcorrtrain.py, no conflict
        import clip

        # Try to load CLIP model
        device = "cpu"  # Use CPU for testing
        _, _ = clip.load("ViT-B/32", device=device)  # Test loading

        print("✅ CLIP model loading working")
        return True
    except Exception as e:
        print(f"❌ CLIP functionality test failed: {e}")
        return False

def test_diffusers_functionality():
    """Test diffusers functionality"""
    try:
        from diffusers import UNet2DModel, DDPMScheduler
        
        # Test creating models
        _ = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
        )

        _ = DDPMScheduler(num_train_timesteps=1000)
        
        print("✅ Diffusers models creation working")
        return True
    except Exception as e:
        print(f"❌ Diffusers functionality test failed: {e}")
        return False

def test_metrics_functionality():
    """Test metrics functionality"""
    try:
        
        # Test torchmetrics
        try:
            from torchmetrics import StructuralSimilarityIndexMeasure
            _ = StructuralSimilarityIndexMeasure()  # Test creation
            print("✅ TorchMetrics working")
        except ImportError:
            print("⚠️  TorchMetrics not available - using fallbacks")
        
        # Test LPIPS
        try:
            import lpips
            _ = lpips.LPIPS(net='alex')  # Test creation
            print("✅ LPIPS working")
        except ImportError:
            print("⚠️  LPIPS not available - using fallbacks")
        
        # Test scikit-image
        try:
            from skimage.metrics import structural_similarity  # noqa: F401
            print("✅ Scikit-image working")
        except ImportError:
            print("⚠️  Scikit-image not available - using fallbacks")
        
        return True
    except Exception as e:
        print(f"❌ Metrics functionality test failed: {e}")
        return False

def test_component_imports():
    """Test importing our custom components"""
    try:
        # Test core components
        from fmriencoder import fMRI_Encoder  # noqa: F401
        from stimencoder import Shape_Encoder  # noqa: F401
        from clipcorrtrain import CLIP_Correlation  # noqa: F401
        from diffusion import Diffusion_Decoder  # noqa: F401
        from gan import GAN_Decoder  # noqa: F401

        print("✅ Core components import successfully")

        # Test utility components
        from metriks import evaluate_decoding_performance  # noqa: F401
        from miyawakidataset import load_miyawaki_dataset_corrected  # noqa: F401
        
        print("✅ Utility components import successfully")
        return True
    except Exception as e:
        print(f"❌ Component import test failed: {e}")
        return False

def test_basic_model_creation():
    """Test creating basic models"""
    try:
        from fmriencoder import fMRI_Encoder
        from stimencoder import Shape_Encoder
        from clipcorrtrain import CLIP_Correlation
        
        # Create models
        fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512)
        stim_encoder = Shape_Encoder(latent_dim=512)
        clip_corr = CLIP_Correlation(latent_dim=512)
        
        print("✅ Basic model creation working")
        
        # Test forward pass with dummy data
        import torch
        dummy_fmri = torch.randn(2, 967)
        dummy_stim = torch.randn(2, 1, 28, 28)

        fmri_lat = fmri_encoder(dummy_fmri)
        stim_lat = stim_encoder(dummy_stim)
        corr_emb = clip_corr(fmri_lat, stim_lat)
        
        print(f"✅ Forward pass working - shapes: fMRI {fmri_lat.shape}, Stim {stim_lat.shape}, Corr {corr_emb.shape}")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Miyawaki2 Dependencies")
    print("=" * 50)
    
    # Test core imports
    print("\n📦 Testing Core Dependencies:")
    core_tests = [
        ("torch", "PyTorch deep learning framework"),
        ("torchvision", "PyTorch computer vision"),
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting library"),
    ]
    
    core_success = 0
    for module, desc in core_tests:
        if test_import(module, desc):
            core_success += 1
    
    # Test advanced imports
    print("\n📦 Testing Advanced Dependencies:")
    advanced_tests = [
        ("diffusers", "Diffusion models"),
        ("transformers", "Transformer models"),
        ("clip", "OpenAI CLIP"),
        ("torchmetrics", "PyTorch metrics"),
        ("lpips", "Perceptual similarity"),
        ("skimage", "Image processing"),
    ]
    
    advanced_success = 0
    for module, desc in advanced_tests:
        if test_import(module, desc):
            advanced_success += 1
    
    # Test functionality
    print("\n🔧 Testing Functionality:")
    functionality_tests = [
        ("PyTorch", test_torch_functionality),
        ("CLIP", test_clip_functionality),
        ("Diffusers", test_diffusers_functionality),
        ("Metrics", test_metrics_functionality),
        ("Components", test_component_imports),
        ("Models", test_basic_model_creation),
    ]
    
    functionality_success = 0
    for name, test_func in functionality_tests:
        print(f"\n🧪 Testing {name}:")
        if test_func():
            functionality_success += 1
    
    # Summary
    total_core = len(core_tests)
    total_advanced = len(advanced_tests)
    total_functionality = len(functionality_tests)
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Core dependencies: {core_success}/{total_core}")
    print(f"Advanced dependencies: {advanced_success}/{total_advanced}")
    print(f"Functionality tests: {functionality_success}/{total_functionality}")
    
    if core_success == total_core and functionality_success == total_functionality:
        print("\n✅ ALL TESTS PASSED!")
        print("🚀 Miyawaki2 is ready for use")
    elif core_success == total_core:
        print("\n⚠️  Core dependencies OK, some advanced features may be limited")
        print("💡 Consider installing missing packages for full functionality")
    else:
        print("\n❌ Some core dependencies missing")
        print("🔧 Run: python install_dependencies.py")
    
    print(f"\n🎯 Device info: {'CUDA available' if test_import('torch') and __import__('torch').cuda.is_available() else 'CPU only'}")

if __name__ == "__main__":
    main()
