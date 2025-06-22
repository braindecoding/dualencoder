#!/usr/bin/env python3
"""
Test SSIM-Focused Configuration
Quick test to verify SSIM-focused model is working correctly
"""

import torch
import numpy as np
from eeg_ldm_ssim_focused import SSIMFocusedEEGDataset, SSIMFocusedEEGDiffusion, SSIMFocusedLoss

def test_ssim_focused_config():
    """Test SSIM-focused configuration"""
    print("🧪 TESTING SSIM-FOCUSED CONFIGURATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    try:
        # Test dataset loading
        print("\n📊 Testing SSIM-focused dataset...")
        
        # Test train split
        train_dataset = SSIMFocusedEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="train", 
            target_size=64
        )
        
        print(f"✅ Train dataset loaded successfully:")
        print(f"   Samples: {len(train_dataset)}")
        print(f"   EEG embeddings shape: {train_dataset.eeg_embeddings.shape}")
        print(f"   Images shape: {train_dataset.images.shape}")
        print(f"   Image range: [{train_dataset.images.min():.3f}, {train_dataset.images.max():.3f}]")
        print(f"   Expected range: [0, 1] for SSIM optimization")
        
        # Test data item
        eeg_emb, image, label = train_dataset[0]
        print(f"\n✅ Data item test:")
        print(f"   EEG embedding: {eeg_emb.shape} {eeg_emb.dtype}")
        print(f"   Image: {image.shape} {image.dtype}")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"   Label: {label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ssim_focused_model():
    """Test SSIM-focused model"""
    print(f"\n🧪 TESTING SSIM-FOCUSED MODEL")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test model initialization
        model = SSIMFocusedEEGDiffusion(
            condition_dim=512,
            image_size=64,
            num_timesteps=100
        ).to(device)
        
        print(f"✅ SSIM-focused model initialized:")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Image size: 64x64")
        print(f"   Beta schedule: Linear (optimized for SSIM)")
        
        # Test forward pass
        batch_size = 4
        eeg_emb = torch.randn(batch_size, 512).to(device)
        images = torch.rand(batch_size, 1, 64, 64).to(device)  # [0, 1] range for SSIM
        t = torch.randint(0, 100, (batch_size,)).to(device)
        
        # Test forward diffusion
        noisy_images, noise = model.forward_diffusion(images, t)
        noise_pred = model(noisy_images, t, eeg_emb)
        
        print(f"✅ Forward pass test:")
        print(f"   Input images: {images.shape} range [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Noisy images: {noisy_images.shape}")
        print(f"   Predicted noise: {noise_pred.shape}")
        
        # Test sampling
        with torch.no_grad():
            generated = model.sample(eeg_emb, num_inference_steps=5)  # Quick test
        
        print(f"✅ Sampling test:")
        print(f"   Generated: {generated.shape} range [{generated.min():.3f}, {generated.max():.3f}]")
        print(f"   Expected range: [0, 1] for SSIM")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ssim_focused_loss():
    """Test SSIM-focused loss function"""
    print(f"\n🧪 TESTING SSIM-FOCUSED LOSS")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test loss initialization
        loss_fn = SSIMFocusedLoss(device=device)
        
        print(f"✅ SSIM-focused loss initialized:")
        print(f"   Components: SSIM + MS-SSIM + L1")
        print(f"   Weights: 70% + 20% + 10%")
        
        # Test loss computation
        batch_size = 4
        predicted = torch.rand(batch_size, 1, 64, 64).to(device)  # [0, 1] range
        target = torch.rand(batch_size, 1, 64, 64).to(device)     # [0, 1] range
        
        total_loss, ssim_loss, l1_loss = loss_fn(predicted, target)
        
        print(f"✅ Loss computation test:")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   SSIM loss: {ssim_loss.item():.4f}")
        print(f"   L1 loss: {l1_loss.item():.4f}")
        
        # Test loss backward
        total_loss.backward()
        print(f"   Backward pass: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ssim_training_step():
    """Test SSIM-focused training step"""
    print(f"\n🧪 TESTING SSIM TRAINING STEP")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load dataset
        train_dataset = SSIMFocusedEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="train", 
            target_size=64
        )
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Initialize model and loss
        model = SSIMFocusedEEGDiffusion(
            condition_dim=512,
            image_size=64,
            num_timesteps=100
        ).to(device)
        
        loss_fn = SSIMFocusedLoss(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test one training step
        eeg_emb, images, labels = next(iter(train_loader))
        eeg_emb = eeg_emb.to(device)
        images = images.to(device)
        
        print(f"✅ Training step test:")
        print(f"   Batch size: {images.shape[0]}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Forward pass
        batch_size = images.shape[0]
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        
        noisy_images, noise = model.forward_diffusion(images, t)
        noise_pred = model(noisy_images, t, eeg_emb)
        
        # Reconstruct predicted image for SSIM loss
        alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
        sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
        predicted_x0 = torch.clamp(predicted_x0, 0, 1)
        
        # SSIM-focused loss
        total_loss, ssim_loss, l1_loss = loss_fn(predicted_x0, images)
        
        print(f"   Forward pass: SUCCESS")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   SSIM loss: {ssim_loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"   Backward pass: SUCCESS")
        print(f"   Optimizer step: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_ssim_vs_other_models():
    """Compare SSIM-focused vs other approaches"""
    print(f"\n📊 SSIM-FOCUSED VS OTHER APPROACHES")
    print("=" * 60)
    
    approaches = {
        "Baseline": {
            "Loss": "L1/L2 regression",
            "Focus": "Pixel-wise accuracy",
            "Strength": "Simple, fast training",
            "Weakness": "Poor perceptual quality"
        },
        "CLIP-Guided": {
            "Loss": "CLIP semantic loss",
            "Focus": "Semantic accuracy",
            "Strength": "Letter recognition",
            "Weakness": "May ignore visual quality"
        },
        "Hybrid CLIP-SSIM": {
            "Loss": "SSIM + Classification + CLIP",
            "Focus": "Balanced approach",
            "Strength": "Multiple objectives",
            "Weakness": "Loss conflicts"
        },
        "Scaled Architecture": {
            "Loss": "L1 loss",
            "Focus": "Model capacity",
            "Strength": "Large model, high resolution",
            "Weakness": "No perceptual optimization"
        },
        "SSIM-Focused": {
            "Loss": "SSIM (70%) + MS-SSIM (20%) + L1 (10%)",
            "Focus": "Perceptual quality",
            "Strength": "Human-like visual similarity",
            "Weakness": "May sacrifice semantic accuracy"
        }
    }
    
    print(f"🔍 Approach Comparison:")
    for approach, details in approaches.items():
        print(f"\n🎯 {approach}:")
        for key, value in details.items():
            print(f"   {key}: {value}")
    
    print(f"\n🎯 SSIM-Focused Expected Advantages:")
    print(f"   ✅ Better perceptual quality (human-like similarity)")
    print(f"   ✅ Higher SSIM scores (structural similarity)")
    print(f"   ✅ More natural-looking letters")
    print(f"   ✅ Optimized for visual perception")
    print(f"   ✅ Better edge and texture preservation")
    print(f"   ✅ Multi-scale structural analysis (MS-SSIM)")

def main():
    """Main test function"""
    print("🔍 SSIM-FOCUSED COMPREHENSIVE TEST")
    print("=" * 70)
    print("Testing SSIM-focused approach for perceptual quality optimization")
    print("=" * 70)
    
    # Run tests
    config_success = test_ssim_focused_config()
    model_success = test_ssim_focused_model()
    loss_success = test_ssim_focused_loss()
    training_success = test_ssim_training_step()
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    if config_success:
        print(f"   ✅ SSIM Configuration: PASSED")
    else:
        print(f"   ❌ SSIM Configuration: FAILED")
        
    if model_success:
        print(f"   ✅ SSIM Model: PASSED")
    else:
        print(f"   ❌ SSIM Model: FAILED")
        
    if loss_success:
        print(f"   ✅ SSIM Loss: PASSED")
    else:
        print(f"   ❌ SSIM Loss: FAILED")
        
    if training_success:
        print(f"   ✅ SSIM Training: PASSED")
    else:
        print(f"   ❌ SSIM Training: FAILED")
    
    # Show comparison
    compare_ssim_vs_other_models()
    
    if all([config_success, model_success, loss_success, training_success]):
        print(f"\n🚀 SSIM-FOCUSED MODEL READY FOR TRAINING!")
        print(f"\n📊 Expected SSIM advantages:")
        print(f"   🎯 Better perceptual quality (human visual similarity)")
        print(f"   📈 Higher SSIM scores (structural similarity index)")
        print(f"   🎨 More natural-looking letter generation")
        print(f"   🔄 Optimized for visual perception rather than pixel accuracy")
        print(f"   💪 Better preservation of edges, textures, and structures")
        print(f"   ⚡ Multi-scale analysis with MS-SSIM")
        
        print(f"\n🎯 Key SSIM Features:")
        print(f"   • Primary loss: SSIM (70% weight)")
        print(f"   • Secondary loss: MS-SSIM (20% weight)")
        print(f"   • Auxiliary loss: L1 (10% weight)")
        print(f"   • Image range: [0, 1] (optimal for SSIM)")
        print(f"   • Focus: Perceptual quality over pixel accuracy")
        print(f"   • Strength: Human-like visual similarity assessment")
    else:
        print(f"\n❌ Please fix issues before training.")

if __name__ == "__main__":
    main()
