#!/usr/bin/env python3
"""
Test Fixed Diffusion Model
Compare original vs fixed diffusion implementation
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Import both versions
from diffusion import Diffusion_Decoder as OriginalDiffusion
from diffusion_fixed import FixedDiffusion_Decoder, SimpleCNN_Decoder
from gan import GAN_Decoder
from miyawakidataset import load_miyawaki_dataset_corrected, create_dataloaders_corrected
from metriks import evaluate_decoding_performance

def compare_architectures():
    """Compare original vs fixed diffusion architectures"""
    print("🔍 Comparing Diffusion Architectures")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    original_diff = OriginalDiffusion(correlation_dim=512).to(device)
    fixed_diff = FixedDiffusion_Decoder(correlation_dim=512).to(device)
    simple_cnn = SimpleCNN_Decoder(correlation_dim=512).to(device)
    gan = GAN_Decoder(correlation_dim=512).to(device)
    
    # Parameter counts
    orig_params = sum(p.numel() for p in original_diff.parameters())
    fixed_params = sum(p.numel() for p in fixed_diff.parameters())
    cnn_params = sum(p.numel() for p in simple_cnn.parameters())
    gan_params = sum(p.numel() for p in gan.parameters())
    
    print(f"Parameter Counts:")
    print(f"  Original Diffusion: {orig_params:,}")
    print(f"  Fixed Diffusion:    {fixed_params:,}")
    print(f"  Simple CNN:         {cnn_params:,}")
    print(f"  GAN:                {gan_params:,}")
    
    return original_diff, fixed_diff, simple_cnn, gan

def test_forward_passes():
    """Test forward passes of all models"""
    print("\n🧪 Testing Forward Passes")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    original_diff, fixed_diff, simple_cnn, gan = compare_architectures()
    
    # Create dummy inputs
    batch_size = 4
    correlation = torch.randn(batch_size, 512, device=device)
    fmri_latent = torch.randn(batch_size, 512, device=device)
    target_images = torch.rand(batch_size, 1, 28, 28, device=device)
    
    print(f"\nInput shapes:")
    print(f"  Correlation: {correlation.shape}")
    print(f"  fMRI latent: {fmri_latent.shape}")
    print(f"  Target images: {target_images.shape}")
    
    # Test inference mode
    print(f"\n📤 Inference Mode:")
    with torch.no_grad():
        # Original diffusion (inference)
        original_diff.eval()
        orig_output = original_diff(correlation, fmri_latent)
        
        # Fixed diffusion (inference)
        fixed_diff.eval()
        fixed_output = fixed_diff(correlation, fmri_latent, num_inference_steps=10)
        
        # Simple CNN
        simple_cnn.eval()
        cnn_output = simple_cnn(correlation, fmri_latent)
        
        # GAN
        gan.eval()
        gan_output = gan(correlation, fmri_latent)
    
    print(f"  Original Diffusion: {orig_output.shape}, Mean: {orig_output.mean():.4f}")
    print(f"  Fixed Diffusion:    {fixed_output.shape}, Mean: {fixed_output.mean():.4f}")
    print(f"  Simple CNN:         {cnn_output.shape}, Mean: {cnn_output.mean():.4f}")
    print(f"  GAN:                {gan_output.shape}, Mean: {gan_output.mean():.4f}")
    
    # Test training mode
    print(f"\n📥 Training Mode:")
    
    # Fixed diffusion (training with proper loss)
    fixed_diff.train()
    fixed_pred, fixed_loss = fixed_diff(correlation, fmri_latent, target_images)
    
    # Simple CNN (training)
    simple_cnn.train()
    cnn_pred, cnn_loss = simple_cnn(correlation, fmri_latent, target_images)
    
    # GAN (training)
    gan.train()
    gan_pred = gan(correlation, fmri_latent)
    gan_loss = nn.MSELoss()(gan_pred, target_images)
    
    print(f"  Fixed Diffusion Loss: {fixed_loss.item():.4f}")
    print(f"  Simple CNN Loss:      {cnn_loss.item():.4f}")
    print(f"  GAN Loss:             {gan_loss.item():.4f}")
    
    return {
        'original': orig_output,
        'fixed': fixed_output,
        'cnn': cnn_output,
        'gan': gan_output
    }

def quick_training_test():
    """Quick training test to see convergence"""
    print("\n🏋️ Quick Training Test")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load small dataset
    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")
    if not dataset_path.exists():
        print("❌ Dataset not found, skipping training test")
        return
    
    train_dataset, test_dataset, _ = load_miyawaki_dataset_corrected(dataset_path)
    train_loader, _ = create_dataloaders_corrected(train_dataset, test_dataset, batch_size=8)
    
    # Create models
    fixed_diff = FixedDiffusion_Decoder(correlation_dim=512).to(device)
    simple_cnn = SimpleCNN_Decoder(correlation_dim=512).to(device)
    
    # Optimizers
    fixed_opt = torch.optim.Adam(fixed_diff.parameters(), lr=1e-3)
    cnn_opt = torch.optim.Adam(simple_cnn.parameters(), lr=1e-3)
    
    # Quick training (5 epochs)
    print("Training for 5 epochs...")
    
    fixed_losses = []
    cnn_losses = []
    
    for epoch in range(5):
        fixed_epoch_loss = 0
        cnn_epoch_loss = 0
        
        for batch in train_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            
            # Dummy correlation (for testing)
            correlation = torch.randn(fmri.shape[0], 512, device=device)
            
            # Fixed diffusion training
            fixed_opt.zero_grad()
            _, fixed_loss = fixed_diff(correlation, fmri, stimulus)
            fixed_loss.backward()
            fixed_opt.step()
            fixed_epoch_loss += fixed_loss.item()
            
            # CNN training
            cnn_opt.zero_grad()
            _, cnn_loss = simple_cnn(correlation, fmri, stimulus)
            cnn_loss.backward()
            cnn_opt.step()
            cnn_epoch_loss += cnn_loss.item()
        
        fixed_epoch_loss /= len(train_loader)
        cnn_epoch_loss /= len(train_loader)
        
        fixed_losses.append(fixed_epoch_loss)
        cnn_losses.append(cnn_epoch_loss)
        
        print(f"  Epoch {epoch+1}: Fixed={fixed_epoch_loss:.4f}, CNN={cnn_epoch_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fixed_losses, label='Fixed Diffusion', marker='o')
    plt.plot(cnn_losses, label='Simple CNN', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test inference
    fixed_diff.eval()
    simple_cnn.eval()
    
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        fmri = test_batch['fmri'][:4].to(device)
        stimulus = test_batch['stimulus'][:4].to(device)
        correlation = torch.randn(4, 512, device=device)
        
        fixed_output = fixed_diff(correlation, fmri, num_inference_steps=10)
        cnn_output = simple_cnn(correlation, fmri)
    
    # Show sample outputs
    plt.subplot(1, 2, 2)
    
    # Show first sample
    plt.subplot(2, 3, 4)
    plt.imshow(stimulus[0, 0].cpu(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(fixed_output[0, 0].cpu(), cmap='gray')
    plt.title('Fixed Diffusion')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cnn_output[0, 0].cpu(), cmap='gray')
    plt.title('Simple CNN')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fixed_diffusion_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Test results saved as 'fixed_diffusion_test.png'")
    
    return fixed_losses, cnn_losses

def main():
    """Main test function"""
    print("🧪 Testing Fixed Diffusion Model")
    print("=" * 60)
    
    # Compare architectures
    compare_architectures()
    
    # Test forward passes
    outputs = test_forward_passes()
    
    # Quick training test
    try:
        fixed_losses, cnn_losses = quick_training_test()
        
        print(f"\n📊 Training Results:")
        print(f"  Fixed Diffusion - Start: {fixed_losses[0]:.4f}, End: {fixed_losses[-1]:.4f}")
        print(f"  Simple CNN - Start: {cnn_losses[0]:.4f}, End: {cnn_losses[-1]:.4f}")
        
        improvement_fixed = (fixed_losses[0] - fixed_losses[-1]) / fixed_losses[0] * 100
        improvement_cnn = (cnn_losses[0] - cnn_losses[-1]) / cnn_losses[0] * 100
        
        print(f"  Fixed Diffusion Improvement: {improvement_fixed:.1f}%")
        print(f"  Simple CNN Improvement: {improvement_cnn:.1f}%")
        
    except Exception as e:
        print(f"⚠️ Training test failed: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📋 FIXED DIFFUSION TEST SUMMARY")
    print(f"=" * 60)
    
    print(f"✅ Fixed Issues:")
    print(f"  1. Added proper conditioning network")
    print(f"  2. Implemented noise prediction training")
    print(f"  3. Reduced complexity (no attention)")
    print(f"  4. Fewer inference steps (10 vs 50)")
    print(f"  5. Training/inference consistency")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Replace original diffusion with fixed version")
    print(f"  2. Run full training comparison")
    print(f"  3. Compare with GAN performance")

if __name__ == "__main__":
    main()
