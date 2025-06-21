#!/usr/bin/env python3
"""
Demo: LDM Integration with Miyawaki4 Embeddings
Test fMRI-to-Image generation using Latent Diffusion Models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our LDM classes
from ldm import fMRIToLDMAdapter, Method1_DirectConditioning, Method2_CrossAttentionConditioning, Method3_ControlNetStyle

def load_miyawaki4_embeddings():
    """Load miyawaki4 embeddings for testing"""
    print("📥 Loading Miyawaki4 Embeddings")
    print("=" * 40)
    
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        print("💡 Please run embedding_converter.py first")
        return None
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    print(f"✅ Embeddings loaded successfully")
    print(f"   📊 Training samples: {len(embeddings_data['train']['fmri_embeddings'])}")
    print(f"   📊 Test samples: {len(embeddings_data['test']['fmri_embeddings'])}")
    
    return embeddings_data

def test_method1_direct_conditioning(embeddings_data):
    """Test Method 1: Direct Conditioning"""
    print("\n🎯 Testing Method 1: Direct Conditioning")
    print("=" * 50)
    
    try:
        # Initialize method
        method1 = Method1_DirectConditioning(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test fMRI embedding
        test_fmri = embeddings_data['test']['fmri_embeddings'][0]  # First test sample
        test_fmri_tensor = torch.FloatTensor(test_fmri)
        
        if torch.cuda.is_available():
            test_fmri_tensor = test_fmri_tensor.cuda()
        
        print(f"📊 Input fMRI embedding shape: {test_fmri_tensor.shape}")
        
        # Test reconstruction
        print("🎨 Generating image...")
        generated_image = method1.reconstruct_from_fmri(test_fmri_tensor)
        
        print(f"✅ Method 1 completed successfully")
        print(f"   📊 Generated image size: {generated_image.size}")
        
        return generated_image
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        return None

def test_method2_cross_attention(embeddings_data):
    """Test Method 2: Cross Attention Conditioning"""
    print("\n🎨 Testing Method 2: Cross Attention Conditioning")
    print("=" * 50)
    
    try:
        # Initialize method
        method2 = Method2_CrossAttentionConditioning(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test fMRI embedding
        test_fmri = embeddings_data['test']['fmri_embeddings'][1]  # Second test sample
        test_fmri_tensor = torch.FloatTensor(test_fmri)
        
        if torch.cuda.is_available():
            test_fmri_tensor = test_fmri_tensor.cuda()
        
        print(f"📊 Input fMRI embedding shape: {test_fmri_tensor.shape}")
        
        # Test reconstruction
        print("🎨 Generating image...")
        generated_image = method2.reconstruct_from_fmri(test_fmri_tensor)
        
        print(f"✅ Method 2 completed successfully")
        print(f"   📊 Generated image size: {generated_image.size}")
        
        return generated_image
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        return None

def test_method3_controlnet(embeddings_data):
    """Test Method 3: ControlNet Style"""
    print("\n🎮 Testing Method 3: ControlNet Style")
    print("=" * 50)
    
    try:
        # Initialize method
        method3 = Method3_ControlNetStyle(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get test fMRI embedding
        test_fmri = embeddings_data['test']['fmri_embeddings'][2]  # Third test sample
        test_fmri_tensor = torch.FloatTensor(test_fmri)
        
        if torch.cuda.is_available():
            test_fmri_tensor = test_fmri_tensor.cuda()
        
        print(f"📊 Input fMRI embedding shape: {test_fmri_tensor.shape}")
        
        # Test reconstruction
        print("🎨 Generating image...")
        generated_image = method3.reconstruct_from_fmri(test_fmri_tensor)
        
        print(f"✅ Method 3 completed successfully")
        print(f"   📊 Generated image size: {generated_image.size}")
        
        return generated_image
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        return None

def visualize_results(embeddings_data, results):
    """Visualize original images vs generated images"""
    print("\n📊 Creating Visualization")
    print("=" * 30)
    
    # Get original test images
    original_images = embeddings_data['test']['original_images'][:3]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('fMRI-to-Image Generation Results', fontsize=16, fontweight='bold')
    
    methods = ['Direct Conditioning', 'Cross Attention', 'ControlNet Style']
    
    for i, (method_name, generated_image) in enumerate(zip(methods, results)):
        # Original image (top row)
        if i < len(original_images):
            original = original_images[i].transpose(1, 2, 0)  # CHW -> HWC
            axes[0, i].imshow(original)
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
        
        # Generated image (bottom row)
        if generated_image is not None:
            axes[1, i].imshow(generated_image)
            axes[1, i].set_title(f'{method_name}\n(Generated)')
            axes[1, i].axis('off')
        else:
            axes[1, i].text(0.5, 0.5, 'Generation Failed', 
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           fontsize=12, color='red')
            axes[1, i].set_title(f'{method_name}\n(Failed)')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ldm_generation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Results saved as 'ldm_generation_results.png'")

def test_adapter_integration():
    """Test fMRIToLDMAdapter integration"""
    print("\n🔗 Testing fMRIToLDMAdapter Integration")
    print("=" * 50)
    
    try:
        # Initialize adapter
        adapter = fMRIToLDMAdapter(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test loading embeddings
        embeddings_data = adapter.load_miyawaki4_embeddings("miyawaki4_embeddings.pkl")
        
        if embeddings_data is not None:
            print("✅ Adapter integration successful")
            return True
        else:
            print("❌ Adapter integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Adapter integration error: {e}")
        return False

def create_embedding_analysis(embeddings_data):
    """Analyze embedding properties for LDM conditioning"""
    print("\n🔍 Analyzing Embeddings for LDM Conditioning")
    print("=" * 50)
    
    # Get embeddings
    train_fmri = embeddings_data['train']['fmri_embeddings']
    test_fmri = embeddings_data['test']['fmri_embeddings']
    
    # Compute statistics
    train_mean = np.mean(train_fmri, axis=0)
    train_std = np.std(train_fmri, axis=0)
    test_mean = np.mean(test_fmri, axis=0)
    test_std = np.std(test_fmri, axis=0)
    
    print(f"📊 Embedding Statistics:")
    print(f"   Training embeddings: {train_fmri.shape}")
    print(f"   Test embeddings: {test_fmri.shape}")
    print(f"   Train mean range: [{train_mean.min():.3f}, {train_mean.max():.3f}]")
    print(f"   Train std range: [{train_std.min():.3f}, {train_std.max():.3f}]")
    print(f"   Test mean range: [{test_mean.min():.3f}, {test_mean.max():.3f}]")
    print(f"   Test std range: [{test_std.min():.3f}, {test_std.max():.3f}]")
    
    # Visualize embedding distributions
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(train_fmri.flatten(), bins=50, alpha=0.7, label='Training', density=True)
    plt.hist(test_fmri.flatten(), bins=50, alpha=0.7, label='Test', density=True)
    plt.xlabel('Embedding Value')
    plt.ylabel('Density')
    plt.title('Embedding Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_mean, label='Training Mean', alpha=0.7)
    plt.plot(test_mean, label='Test Mean', alpha=0.7)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Mean Value')
    plt.title('Mean Values per Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_std, label='Training Std', alpha=0.7)
    plt.plot(test_std, label='Test Std', alpha=0.7)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Std Deviation per Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis_for_ldm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Analysis saved as 'embedding_analysis_for_ldm.png'")

def main():
    """Main demo function"""
    print("🎯 LDM Integration Demo with Miyawaki4")
    print("=" * 60)
    
    # Load embeddings
    embeddings_data = load_miyawaki4_embeddings()
    if embeddings_data is None:
        print("❌ Cannot proceed without embeddings")
        return
    
    # Test adapter integration
    adapter_ok = test_adapter_integration()
    
    # Analyze embeddings
    create_embedding_analysis(embeddings_data)
    
    # Test all three methods
    print("\n🚀 Testing All LDM Methods")
    print("=" * 40)
    
    results = []
    
    # Method 1: Direct Conditioning
    result1 = test_method1_direct_conditioning(embeddings_data)
    results.append(result1)
    
    # Method 2: Cross Attention
    result2 = test_method2_cross_attention(embeddings_data)
    results.append(result2)
    
    # Method 3: ControlNet Style
    result3 = test_method3_controlnet(embeddings_data)
    results.append(result3)
    
    # Visualize results
    visualize_results(embeddings_data, results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 LDM INTEGRATION DEMO SUMMARY")
    print("=" * 60)
    
    print(f"🔗 Adapter Integration: {'✅ Success' if adapter_ok else '❌ Failed'}")
    print(f"🎯 Method 1 (Direct): {'✅ Success' if results[0] is not None else '❌ Failed'}")
    print(f"🎨 Method 2 (Cross-Attention): {'✅ Success' if results[1] is not None else '❌ Failed'}")
    print(f"🎮 Method 3 (ControlNet): {'✅ Success' if results[2] is not None else '❌ Failed'}")
    
    successful_methods = sum(1 for r in results if r is not None)
    print(f"\n📊 Overall Success Rate: {successful_methods}/3 methods working")
    
    print(f"\n📁 Generated Files:")
    print(f"   - ldm_generation_results.png")
    print(f"   - embedding_analysis_for_ldm.png")
    
    if successful_methods > 0:
        print(f"\n🎯 Next Steps:")
        print(f"   - Fine-tune conditioning networks")
        print(f"   - Implement training pipelines")
        print(f"   - Add evaluation metrics")
        print(f"   - Optimize for real-time generation")
    else:
        print(f"\n💡 Troubleshooting:")
        print(f"   - Check GPU memory availability")
        print(f"   - Install missing dependencies")
        print(f"   - Verify Stable Diffusion model access")

if __name__ == "__main__":
    main()
