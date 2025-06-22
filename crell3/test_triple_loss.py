#!/usr/bin/env python3
"""
Test Triple Loss Configuration
Quick test to verify Triple Loss model is working correctly
"""

import torch
import numpy as np
from eeg_ldm_triple_loss import TripleLossEEGDataset, TripleLossEEGDiffusion, TripleLoss

def test_triple_loss_config():
    """Test Triple Loss configuration"""
    print("🧪 TESTING TRIPLE LOSS CONFIGURATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    try:
        # Test dataset loading
        print("\n📊 Testing Triple Loss dataset...")
        
        # Test train split
        train_dataset = TripleLossEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="train", 
            target_size=64
        )
        
        print(f"✅ Train dataset loaded successfully:")
        print(f"   Samples: {len(train_dataset)}")
        print(f"   EEG embeddings shape: {train_dataset.eeg_embeddings.shape}")
        print(f"   Images shape: {train_dataset.images.shape}")
        print(f"   Image range: [{train_dataset.images.min():.3f}, {train_dataset.images.max():.3f}]")
        print(f"   Expected range: [0, 1] for SSIM + CLIP + MSE")
        
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

def test_triple_loss_model():
    """Test Triple Loss model"""
    print(f"\n🧪 TESTING TRIPLE LOSS MODEL")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test model initialization
        model = TripleLossEEGDiffusion(
            condition_dim=512,
            image_size=64,
            num_timesteps=100
        ).to(device)
        
        print(f"✅ Triple Loss model initialized:")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Image size: 64x64")
        print(f"   Beta schedule: Cosine (optimized for CLIP)")
        print(f"   Architecture: UNet with attention")
        
        # Test forward pass
        batch_size = 4
        eeg_emb = torch.randn(batch_size, 512).to(device)
        images = torch.rand(batch_size, 1, 64, 64).to(device)  # [0, 1] range
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
        print(f"   Expected range: [0, 1] for Triple Loss")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triple_loss_function():
    """Test Triple Loss function"""
    print(f"\n🧪 TESTING TRIPLE LOSS FUNCTION")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test loss initialization
        loss_fn = TripleLoss(device=device)
        
        print(f"✅ Triple Loss initialized:")
        print(f"   Components: SSIM + CLIP + MSE")
        print(f"   Weights: 50% + 30% + 20%")
        print(f"   CLIP model: openai/clip-vit-base-patch32")
        print(f"   Letter templates: 10 letters (a,d,e,f,j,n,o,s,t,v)")
        
        # Test loss computation
        batch_size = 2  # Small batch for CLIP test
        predicted = torch.rand(batch_size, 1, 64, 64).to(device)  # [0, 1] range
        target = torch.rand(batch_size, 1, 64, 64).to(device)     # [0, 1] range
        labels = torch.tensor([0, 1]).to(device)  # Letter labels
        
        total_loss, ssim_loss, clip_loss, mse_loss = loss_fn(predicted, target, labels)
        
        print(f"✅ Loss computation test:")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   SSIM loss: {ssim_loss.item():.4f}")
        print(f"   CLIP loss: {clip_loss.item():.4f}")
        print(f"   MSE loss: {mse_loss.item():.4f}")
        
        # Test loss backward
        total_loss.backward()
        print(f"   Backward pass: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triple_loss_training_step():
    """Test Triple Loss training step"""
    print(f"\n🧪 TESTING TRIPLE LOSS TRAINING STEP")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load dataset
        train_dataset = TripleLossEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="train", 
            target_size=64
        )
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Small batch
        
        # Initialize model and loss
        model = TripleLossEEGDiffusion(
            condition_dim=512,
            image_size=64,
            num_timesteps=100
        ).to(device)
        
        loss_fn = TripleLoss(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Test one training step
        eeg_emb, images, labels = next(iter(train_loader))
        eeg_emb = eeg_emb.to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        print(f"✅ Training step test:")
        print(f"   Batch size: {images.shape[0]}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Labels: {labels}")
        
        # Forward pass
        batch_size = images.shape[0]
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        
        noisy_images, noise = model.forward_diffusion(images, t)
        noise_pred = model(noisy_images, t, eeg_emb)
        
        # Reconstruct predicted image for Triple Loss
        alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
        sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
        predicted_x0 = torch.clamp(predicted_x0, 0, 1)
        
        # Triple Loss computation
        total_loss, ssim_loss, clip_loss, mse_loss = loss_fn(predicted_x0, images, labels)
        
        print(f"   Forward pass: SUCCESS")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   SSIM loss: {ssim_loss.item():.4f}")
        print(f"   CLIP loss: {clip_loss.item():.4f}")
        print(f"   MSE loss: {mse_loss.item():.4f}")
        
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

def compare_triple_loss_vs_others():
    """Compare Triple Loss vs other approaches"""
    print(f"\n📊 TRIPLE LOSS VS OTHER APPROACHES")
    print("=" * 60)
    
    approaches = {
        "Baseline": {
            "Loss": "L1/L2 regression",
            "Focus": "Pixel-wise accuracy",
            "Strength": "Simple, fast",
            "Weakness": "Poor perceptual quality"
        },
        "CLIP-Only": {
            "Loss": "CLIP semantic loss",
            "Focus": "Semantic accuracy",
            "Strength": "Letter recognition",
            "Weakness": "May ignore visual quality"
        },
        "SSIM-Only": {
            "Loss": "SSIM perceptual loss",
            "Focus": "Perceptual quality",
            "Strength": "Human-like similarity",
            "Weakness": "May sacrifice semantic accuracy"
        },
        "Scaled Architecture": {
            "Loss": "L1 loss",
            "Focus": "Model capacity",
            "Strength": "Large model, high resolution",
            "Weakness": "No perceptual/semantic optimization"
        },
        "Triple Loss": {
            "Loss": "SSIM (50%) + CLIP (30%) + MSE (20%)",
            "Focus": "Comprehensive optimization",
            "Strength": "Best of all worlds",
            "Weakness": "More complex training"
        }
    }
    
    print(f"🔍 Approach Comparison:")
    for approach, details in approaches.items():
        print(f"\n🎯 {approach}:")
        for key, value in details.items():
            print(f"   {key}: {value}")
    
    print(f"\n🎯 Triple Loss Expected Advantages:")
    print(f"   ✅ Perceptual quality (SSIM 50%)")
    print(f"   ✅ Semantic accuracy (CLIP 30%)")
    print(f"   ✅ Reconstruction quality (MSE 20%)")
    print(f"   ✅ Balanced optimization approach")
    print(f"   ✅ Best of all worlds combination")
    print(f"   ✅ Comprehensive loss function")

def main():
    """Main test function"""
    print("🔍 TRIPLE LOSS COMPREHENSIVE TEST")
    print("=" * 70)
    print("Testing Triple Loss approach: SSIM + CLIP + MSE")
    print("=" * 70)
    
    # Run tests
    config_success = test_triple_loss_config()
    model_success = test_triple_loss_model()
    loss_success = test_triple_loss_function()
    training_success = test_triple_loss_training_step()
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    if config_success:
        print(f"   ✅ Triple Loss Configuration: PASSED")
    else:
        print(f"   ❌ Triple Loss Configuration: FAILED")
        
    if model_success:
        print(f"   ✅ Triple Loss Model: PASSED")
    else:
        print(f"   ❌ Triple Loss Model: FAILED")
        
    if loss_success:
        print(f"   ✅ Triple Loss Function: PASSED")
    else:
        print(f"   ❌ Triple Loss Function: FAILED")
        
    if training_success:
        print(f"   ✅ Triple Loss Training: PASSED")
    else:
        print(f"   ❌ Triple Loss Training: FAILED")
    
    # Show comparison
    compare_triple_loss_vs_others()
    
    if all([config_success, model_success, loss_success, training_success]):
        print(f"\n🚀 TRIPLE LOSS MODEL READY FOR TRAINING!")
        print(f"\n📊 Expected Triple Loss advantages:")
        print(f"   🎯 Comprehensive optimization (perceptual + semantic + reconstruction)")
        print(f"   📈 Higher SSIM scores (structural similarity)")
        print(f"   🔤 Better letter recognition (CLIP guidance)")
        print(f"   📐 Good reconstruction quality (MSE)")
        print(f"   ⚖️ Balanced approach (50% + 30% + 20%)")
        print(f"   🏆 Best of all worlds combination")
        
        print(f"\n🎯 Key Triple Loss Features:")
        print(f"   • Primary loss: SSIM (50% weight) - Perceptual quality")
        print(f"   • Secondary loss: CLIP (30% weight) - Semantic accuracy")
        print(f"   • Auxiliary loss: MSE (20% weight) - Reconstruction quality")
        print(f"   • Image range: [0, 1] (optimal for all losses)")
        print(f"   • Focus: Comprehensive multi-objective optimization")
        print(f"   • Strength: Combines advantages of all approaches")
    else:
        print(f"\n❌ Please fix issues before training.")

if __name__ == "__main__":
    main()
