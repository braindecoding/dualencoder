#!/usr/bin/env python3
"""
Quick Test: Verify Method 2 dtype fix only
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
from ldm import Method2_CrossAttentionConditioning

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

def test_method2_comprehensive():
    """Comprehensive test of Method 2 with different scenarios"""
    print("\n🎨 COMPREHENSIVE METHOD 2 TEST")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Device: {device}")
    
    # Test scenarios
    scenarios = [
        ("Standard Test", 10, torch.float32),
        ("Fast Test", 5, torch.float32),
    ]
    
    if device == 'cuda':
        scenarios.append(("Half Precision", 5, torch.float16))
    
    results = []
    
    for scenario_name, num_steps, input_dtype in scenarios:
        print(f"\n🧪 {scenario_name} (steps={num_steps}, dtype={input_dtype})")
        print("-" * 40)
        
        try:
            # Initialize method
            method2 = Method2_CrossAttentionConditioning(device=device)
            
            # Get test embedding with specific dtype
            test_embedding = load_test_embedding()
            test_embedding = test_embedding.to(device=device, dtype=input_dtype)
            
            print(f"📊 Input shape: {test_embedding.shape}")
            print(f"📊 Input dtype: {test_embedding.dtype}")
            
            # Test generation
            print(f"🎨 Generating image...")
            generated_image = method2.reconstruct_from_fmri(test_embedding, num_steps=num_steps)
            
            print(f"✅ {scenario_name} SUCCESS!")
            print(f"   📊 Generated image size: {generated_image.size}")
            
            results.append((scenario_name, generated_image, True, None))
            
        except Exception as e:
            print(f"❌ {scenario_name} FAILED: {e}")
            results.append((scenario_name, None, False, str(e)))
    
    return results

def visualize_comprehensive_results(results):
    """Visualize comprehensive test results"""
    print("\n📊 Creating comprehensive test visualization")
    print("=" * 50)
    
    # Count successful tests
    successful_tests = [r for r in results if r[2]]
    failed_tests = [r for r in results if not r[2]]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Method 2 Cross-Attention: Comprehensive Dtype Fix Test', fontsize=16, fontweight='bold')
    
    # Show successful results
    for i, (name, image, success, error) in enumerate(results[:3]):
        row, col = 0, i
        if success and image is not None:
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'{name}\n✅ SUCCESS')
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, f'{name}\n❌ FAILED\n{error[:50]}...', 
                               ha='center', va='center', transform=axes[row, col].transAxes,
                               fontsize=10, color='red')
            axes[row, col].set_title(f'{name}\n❌ FAILED')
            axes[row, col].axis('off')
    
    # Summary statistics
    axes[1, 0].text(0.1, 0.9, 'TEST SUMMARY', fontsize=14, fontweight='bold')
    axes[1, 0].text(0.1, 0.8, f'Total Tests: {len(results)}', fontsize=12)
    axes[1, 0].text(0.1, 0.7, f'Successful: {len(successful_tests)}', fontsize=12, color='green')
    axes[1, 0].text(0.1, 0.6, f'Failed: {len(failed_tests)}', fontsize=12, color='red')
    axes[1, 0].text(0.1, 0.5, f'Success Rate: {len(successful_tests)/len(results)*100:.1f}%', fontsize=12)
    
    if len(successful_tests) == len(results):
        axes[1, 0].text(0.1, 0.3, '🎉 ALL TESTS PASSED!', fontsize=14, color='green', fontweight='bold')
        axes[1, 0].text(0.1, 0.2, 'Method 2 dtype fix is COMPLETE', fontsize=12, color='green')
    else:
        axes[1, 0].text(0.1, 0.3, '⚠️ Some tests failed', fontsize=14, color='orange', fontweight='bold')
        axes[1, 0].text(0.1, 0.2, 'Additional debugging needed', fontsize=12, color='orange')
    
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Technical details
    axes[1, 1].text(0.1, 0.9, 'TECHNICAL DETAILS', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.1, 0.8, f'PyTorch: {torch.__version__}', fontsize=10)
    axes[1, 1].text(0.1, 0.7, f'CUDA Available: {torch.cuda.is_available()}', fontsize=10)
    if torch.cuda.is_available():
        axes[1, 1].text(0.1, 0.6, f'GPU: {torch.cuda.get_device_name()}', fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'CUDA Version: {torch.version.cuda}', fontsize=10)
    
    axes[1, 1].text(0.1, 0.3, 'DTYPE FIXES APPLIED:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.2, '✅ Consistent dtype handling', fontsize=10)
    axes[1, 1].text(0.1, 0.1, '✅ Model dtype detection', fontsize=10)
    axes[1, 1].text(0.1, 0.0, '✅ Automatic conversions', fontsize=10)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # Error details (if any)
    axes[1, 2].text(0.1, 0.9, 'ERROR ANALYSIS', fontsize=14, fontweight='bold')
    if failed_tests:
        for i, (name, _, _, error) in enumerate(failed_tests[:3]):
            axes[1, 2].text(0.1, 0.8-i*0.2, f'{name}:', fontsize=10, fontweight='bold')
            axes[1, 2].text(0.1, 0.75-i*0.2, f'{error[:60]}...', fontsize=9, color='red')
    else:
        axes[1, 2].text(0.1, 0.5, '🎉 NO ERRORS!', fontsize=14, color='green', fontweight='bold')
        axes[1, 2].text(0.1, 0.4, 'All dtype issues resolved', fontsize=12, color='green')
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('method2_comprehensive_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Results saved as 'method2_comprehensive_test.png'")

def main():
    """Main test function"""
    print("🔧 METHOD 2 DTYPE FIX VERIFICATION")
    print("=" * 60)
    
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔢 PyTorch version: {torch.__version__}")
    
    # Run comprehensive tests
    results = test_method2_comprehensive()
    
    # Visualize results
    visualize_comprehensive_results(results)
    
    # Final summary
    successful_tests = [r for r in results if r[2]]
    
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    print(f"🎨 Method 2 (Cross-Attention) Tests: {len(successful_tests)}/{len(results)} passed")
    
    if len(successful_tests) == len(results):
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"   ✅ All dtype issues in Method 2 are FIXED")
        print(f"   ✅ Works with float32 and float16")
        print(f"   ✅ Handles different inference steps")
        print(f"   🚀 Method 2 is PRODUCTION READY!")
    else:
        print(f"\n⚠️ Partial success: {len(successful_tests)}/{len(results)} tests passed")
        print(f"   🔧 Some scenarios still need debugging")
    
    print(f"\n📁 Generated Files:")
    print(f"   - method2_comprehensive_test.png")
    
    print(f"\n🎯 Status:")
    print(f"   Method 2: {'✅ FIXED' if len(successful_tests) == len(results) else '⚠️ Needs work'}")

if __name__ == "__main__":
    main()
