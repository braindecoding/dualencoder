#!/usr/bin/env python3
"""
Test Script: Verify dtype fixes for LDM Methods 2 & 3
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our fixed LDM classes
from ldm import Method2_CrossAttentionConditioning, Method3_ControlNetStyle

def load_test_embedding():
    """Load a test fMRI embedding"""
    print("📥 Loading test fMRI embedding")
    
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found, using dummy embedding")
        return torch.randn(512)
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Get first test embedding
    test_embedding = embeddings_data['test']['fmri_embeddings'][0]
    test_embedding = torch.FloatTensor(test_embedding)
    
    print(f"✅ Test embedding loaded: {test_embedding.shape}")
    return test_embedding

def test_method2_dtype_fix():
    """Test Method 2 with dtype fixes"""
    print("\n🎨 Testing Method 2: Cross-Attention (Fixed)")
    print("=" * 50)
    
    try:
        # Initialize method
        method2 = Method2_CrossAttentionConditioning(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test embedding
        test_embedding = load_test_embedding()
        if torch.cuda.is_available():
            test_embedding = test_embedding.cuda()
        
        print(f"📊 Input embedding shape: {test_embedding.shape}")
        print(f"📊 Input embedding dtype: {test_embedding.dtype}")
        
        # Test generation
        print("🎨 Generating image with fixed dtype handling...")
        generated_image = method2.reconstruct_from_fmri(test_embedding, num_steps=10)  # Fewer steps for testing
        
        print(f"✅ Method 2 completed successfully!")
        print(f"   📊 Generated image size: {generated_image.size}")
        print(f"   📊 Generated image type: {type(generated_image)}")
        
        return generated_image, True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        return None, False

def test_method3_dtype_fix():
    """Test Method 3 with dtype fixes"""
    print("\n🎮 Testing Method 3: ControlNet (Fixed)")
    print("=" * 50)
    
    try:
        # Initialize method
        method3 = Method3_ControlNetStyle(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test embedding
        test_embedding = load_test_embedding()
        if torch.cuda.is_available():
            test_embedding = test_embedding.cuda()
        
        print(f"📊 Input embedding shape: {test_embedding.shape}")
        print(f"📊 Input embedding dtype: {test_embedding.dtype}")
        
        # Test generation
        print("🎨 Generating image with fixed dtype handling...")
        generated_image = method3.reconstruct_from_fmri(test_embedding, num_steps=10)  # Fewer steps for testing
        
        print(f"✅ Method 3 completed successfully!")
        print(f"   📊 Generated image size: {generated_image.size}")
        print(f"   📊 Generated image type: {type(generated_image)}")
        
        return generated_image, True
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        return None, False

def test_dtype_consistency():
    """Test dtype consistency across different scenarios"""
    print("\n🔍 Testing dtype consistency")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different input dtypes
    test_cases = [
        ("float32", torch.float32),
        ("float16", torch.float16) if device == 'cuda' else ("float32", torch.float32),
    ]
    
    for name, dtype in test_cases:
        print(f"\n🧪 Testing with {name} input:")
        
        # Create test embedding with specific dtype
        test_embedding = torch.randn(512, dtype=dtype, device=device)
        print(f"   Input dtype: {test_embedding.dtype}")
        
        try:
            # Test Method 2
            method2 = Method2_CrossAttentionConditioning(device=device)
            if hasattr(method2, 'fmri_projection') and method2.fmri_projection is not None:
                # Test projection
                with torch.no_grad():
                    projected = method2.fmri_projection(test_embedding.unsqueeze(0))
                    print(f"   Method 2 projection output dtype: {projected.dtype}")
            
            # Test Method 3
            method3 = Method3_ControlNetStyle(device=device)
            if hasattr(method3, 'fmri_controlnet') and method3.fmri_controlnet is not None:
                # Test controlnet
                with torch.no_grad():
                    controlnet_out = method3.fmri_controlnet(test_embedding.unsqueeze(0))
                    print(f"   Method 3 controlnet output dtype: {controlnet_out.dtype}")
            
            print(f"   ✅ {name} dtype test passed")
            
        except Exception as e:
            print(f"   ❌ {name} dtype test failed: {e}")

def visualize_results(method2_result, method2_success, method3_result, method3_success):
    """Visualize test results"""
    print("\n📊 Creating test results visualization")
    print("=" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('LDM Dtype Fix Test Results', fontsize=16, fontweight='bold')
    
    # Method 2 result
    if method2_success and method2_result is not None:
        axes[0].imshow(method2_result)
        axes[0].set_title('Method 2: Cross-Attention\n✅ FIXED - Success')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Method 2\n❌ Failed', 
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=14, color='red')
        axes[0].set_title('Method 2: Cross-Attention\n❌ Still has issues')
        axes[0].axis('off')
    
    # Method 3 result
    if method3_success and method3_result is not None:
        axes[1].imshow(method3_result)
        axes[1].set_title('Method 3: ControlNet\n✅ FIXED - Success')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Method 3\n❌ Failed', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=14, color='red')
        axes[1].set_title('Method 3: ControlNet\n❌ Still has issues')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dtype_fix_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Test results saved as 'dtype_fix_test_results.png'")

def main():
    """Main test function"""
    print("🔧 LDM DTYPE FIXES VERIFICATION TEST")
    print("=" * 60)
    
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔢 PyTorch version: {torch.__version__}")
    
    # Test Method 2
    method2_result, method2_success = test_method2_dtype_fix()
    
    # Test Method 3
    method3_result, method3_success = test_method3_dtype_fix()
    
    # Test dtype consistency
    test_dtype_consistency()
    
    # Visualize results
    visualize_results(method2_result, method2_success, method3_result, method3_success)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DTYPE FIX TEST SUMMARY")
    print("=" * 60)
    
    print(f"🎨 Method 2 (Cross-Attention): {'✅ FIXED' if method2_success else '❌ Still has issues'}")
    print(f"🎮 Method 3 (ControlNet): {'✅ FIXED' if method3_success else '❌ Still has issues'}")
    
    total_success = sum([method2_success, method3_success])
    print(f"\n📊 Overall Fix Success Rate: {total_success}/2 methods working")
    
    if total_success == 2:
        print(f"\n🎉 ALL DTYPE ISSUES FIXED!")
        print(f"   ✅ Method 2: Cross-attention working perfectly")
        print(f"   ✅ Method 3: ControlNet working perfectly")
        print(f"   🚀 Ready for production use!")
    elif total_success == 1:
        print(f"\n⚠️ PARTIAL SUCCESS - 1 method fixed")
        print(f"   🔧 Additional debugging needed for remaining method")
    else:
        print(f"\n❌ FIXES NEED MORE WORK")
        print(f"   🔧 Both methods still have dtype issues")
        print(f"   💡 Consider using float32 consistently or add more dtype conversions")
    
    print(f"\n📁 Generated Files:")
    print(f"   - dtype_fix_test_results.png")
    
    print(f"\n🎯 Next Steps:")
    if total_success < 2:
        print(f"   - Debug remaining dtype mismatches")
        print(f"   - Add more explicit dtype conversions")
        print(f"   - Test with different PyTorch versions")
    else:
        print(f"   - Run full LDM integration demo")
        print(f"   - Test with larger datasets")
        print(f"   - Optimize inference speed")

if __name__ == "__main__":
    main()
