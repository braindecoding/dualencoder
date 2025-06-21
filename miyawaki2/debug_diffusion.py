#!/usr/bin/env python3
"""
Debug Diffusion Model Performance
Analyze why diffusion performs poorly
"""

import torch
import torch.nn as nn
from diffusion import Diffusion_Decoder
from gan import GAN_Decoder
import matplotlib.pyplot as plt

def debug_diffusion_architecture():
    """Debug diffusion model architecture"""
    print("🔍 Debugging Diffusion Model Architecture")
    print("=" * 50)
    
    # Create diffusion decoder
    diffusion = Diffusion_Decoder(correlation_dim=512)
    
    # Check if using real U-Net or MLP fallback
    print(f"Diffusion decoder type: {type(diffusion.unet)}")
    print(f"Scheduler available: {diffusion.scheduler is not None}")
    
    if hasattr(diffusion.unet, 'config'):
        print("✅ Using real U-Net from diffusers")
        print(f"U-Net config: {diffusion.unet.config}")
    else:
        print("❌ Using MLP fallback")
        print(f"MLP architecture: {diffusion.unet}")
    
    # Count parameters
    diffusion_params = sum(p.numel() for p in diffusion.parameters())
    print(f"Diffusion parameters: {diffusion_params:,}")
    
    return diffusion

def debug_forward_pass():
    """Debug forward pass differences"""
    print("\n🔍 Debugging Forward Pass")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    diffusion = Diffusion_Decoder(correlation_dim=512).to(device)
    gan = GAN_Decoder(correlation_dim=512).to(device)
    
    # Create dummy inputs
    batch_size = 4
    correlation = torch.randn(batch_size, 512, device=device)
    fmri_latent = torch.randn(batch_size, 512, device=device)
    
    print(f"Input shapes:")
    print(f"  Correlation: {correlation.shape}")
    print(f"  fMRI latent: {fmri_latent.shape}")
    
    # Forward pass
    with torch.no_grad():
        diff_output = diffusion(correlation, fmri_latent)
        gan_output = gan(correlation, fmri_latent)
    
    print(f"\nOutput shapes:")
    print(f"  Diffusion: {diff_output.shape}")
    print(f"  GAN: {gan_output.shape}")
    
    # Analyze output statistics
    print(f"\nOutput statistics:")
    print(f"  Diffusion - Mean: {diff_output.mean():.4f}, Std: {diff_output.std():.4f}")
    print(f"  Diffusion - Min: {diff_output.min():.4f}, Max: {diff_output.max():.4f}")
    print(f"  GAN - Mean: {gan_output.mean():.4f}, Std: {gan_output.std():.4f}")
    print(f"  GAN - Min: {gan_output.min():.4f}, Max: {gan_output.max():.4f}")
    
    return diff_output, gan_output

def analyze_diffusion_problems():
    """Analyze potential problems with diffusion"""
    print("\n🚨 Analyzing Diffusion Problems")
    print("=" * 50)
    
    problems = []
    
    # Problem 1: Conditioning
    print("1. Conditioning Issues:")
    print("   ❌ Diffusion tidak menggunakan proper conditioning")
    print("   ❌ Line 71: noise_pred = self.unet(noise, t).sample")
    print("   ❌ Tidak ada conditioning pada correlation/fmri_latent")
    print("   ✅ Seharusnya: noise_pred = self.unet(noise, t, condition).sample")
    problems.append("No proper conditioning")
    
    # Problem 2: Training mismatch
    print("\n2. Training vs Inference Mismatch:")
    print("   ❌ Training: MSE loss langsung pada output")
    print("   ❌ Inference: Denoising process dengan 50 steps")
    print("   ❌ Training tidak mengajarkan denoising process")
    problems.append("Training/inference mismatch")
    
    # Problem 3: Architecture complexity
    print("\n3. Architecture Complexity:")
    print("   ❌ U-Net terlalu complex untuk 28x28 images")
    print("   ❌ Attention blocks mungkin overkill")
    print("   ❌ 50 denoising steps terlalu banyak")
    problems.append("Over-complex architecture")
    
    # Problem 4: Loss function
    print("\n4. Loss Function Issues:")
    print("   ❌ MSE loss tidak cocok untuk diffusion training")
    print("   ❌ Seharusnya noise prediction loss")
    print("   ❌ Tidak ada noise scheduling dalam training")
    problems.append("Wrong loss function")
    
    return problems

def compare_architectures():
    """Compare diffusion vs GAN architectures"""
    print("\n⚖️ Architecture Comparison")
    print("=" * 50)
    
    # Create models
    diffusion = Diffusion_Decoder(correlation_dim=512)
    gan = GAN_Decoder(correlation_dim=512)
    
    # Parameter count
    diff_params = sum(p.numel() for p in diffusion.parameters())
    gan_params = sum(p.numel() for p in gan.parameters())
    
    print(f"Parameter Count:")
    print(f"  Diffusion: {diff_params:,}")
    print(f"  GAN: {gan_params:,}")
    print(f"  Ratio: {diff_params/gan_params:.1f}x")
    
    # Architecture complexity
    print(f"\nArchitecture Complexity:")
    print(f"  Diffusion: U-Net + Scheduler + 50 inference steps")
    print(f"  GAN: Simple ConvTranspose + 1 forward pass")
    
    # Training approach
    print(f"\nTraining Approach:")
    print(f"  Diffusion: Direct MSE (wrong for diffusion)")
    print(f"  GAN: Direct MSE (correct for reconstruction)")

def propose_fixes():
    """Propose fixes for diffusion model"""
    print("\n🔧 Proposed Fixes")
    print("=" * 50)
    
    fixes = [
        "1. Fix Conditioning: Add proper conditioning in U-Net forward pass",
        "2. Fix Training: Use noise prediction loss instead of MSE",
        "3. Simplify Architecture: Use smaller U-Net or fewer attention blocks",
        "4. Reduce Inference Steps: Use 10-20 steps instead of 50",
        "5. Match Training/Inference: Train with noise scheduling",
        "6. Alternative: Use simpler diffusion or replace with VAE"
    ]
    
    for fix in fixes:
        print(f"  {fix}")
    
    return fixes

def create_comparison_plot(diff_output, gan_output):
    """Create comparison plot"""
    print("\n📊 Creating Comparison Plot")
    
    # Convert to numpy for plotting
    diff_np = diff_output[0, 0].cpu().numpy()
    gan_np = gan_output[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Diffusion output
    im1 = axes[0].imshow(diff_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Diffusion Output\nMean: {diff_np.mean():.3f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # GAN output
    im2 = axes[1].imshow(gan_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'GAN Output\nMean: {gan_np.mean():.3f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff_map = abs(diff_np - gan_np)
    im3 = axes[2].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title(f'Absolute Difference\nMax: {diff_map.max():.3f}')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('diffusion_vs_gan_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Comparison saved as 'diffusion_vs_gan_debug.png'")

def main():
    """Main debug function"""
    print("🔍 Diffusion Model Debug Analysis")
    print("=" * 60)
    
    # Debug architecture
    diffusion = debug_diffusion_architecture()
    
    # Debug forward pass
    diff_output, gan_output = debug_forward_pass()
    
    # Analyze problems
    problems = analyze_diffusion_problems()
    
    # Compare architectures
    compare_architectures()
    
    # Propose fixes
    fixes = propose_fixes()
    
    # Create visualization
    create_comparison_plot(diff_output, gan_output)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEBUG SUMMARY")
    print("=" * 60)
    
    print(f"🚨 Problems Found ({len(problems)}):")
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. {problem}")
    
    print(f"\n🔧 Fixes Needed ({len(fixes)}):")
    for fix in fixes[:3]:  # Show top 3
        print(f"  {fix}")
    
    print(f"\n🎯 Root Cause:")
    print(f"  Diffusion model menggunakan wrong training approach!")
    print(f"  Training dengan MSE tapi inference dengan denoising process")
    print(f"  Tidak ada proper conditioning dalam U-Net")
    
    print(f"\n💡 Quick Fix:")
    print(f"  Ganti diffusion dengan VAE atau simple CNN decoder")
    print(f"  Atau fix conditioning dan training approach")

if __name__ == "__main__":
    main()
