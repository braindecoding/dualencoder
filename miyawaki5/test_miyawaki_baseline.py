#!/usr/bin/env python3
"""
Test Miyawaki Baseline Model Configuration
Verify that the modified baseline model works with Miyawaki embeddings
"""

import torch
import numpy as np
from simple_baseline_model import MiyawakiBaselineDataset, SimpleRegressionModel

def test_dataset_loading():
    """Test dataset loading with Miyawaki embeddings"""
    print("🧪 TESTING MIYAWAKI DATASET LOADING")
    print("=" * 50)
    
    try:
        # Test train dataset
        print("\n📊 Testing train dataset...")
        train_dataset = MiyawakiBaselineDataset(
            model_path="miyawaki_contrastive_clip.pth",
            split="train", 
            target_size=28
        )
        
        print(f"✅ Train dataset loaded successfully:")
        print(f"   Samples: {len(train_dataset)}")
        
        # Test data item
        fmri_emb, image = train_dataset[0]
        print(f"   fMRI embedding: {fmri_emb.shape} {fmri_emb.dtype}")
        print(f"   Image: {image.shape} {image.dtype}")
        print(f"   fMRI range: [{fmri_emb.min():.3f}, {fmri_emb.max():.3f}]")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Test test dataset
        print("\n📊 Testing test dataset...")
        test_dataset = MiyawakiBaselineDataset(
            model_path="miyawaki_contrastive_clip.pth",
            split="test", 
            target_size=28
        )
        
        print(f"✅ Test dataset loaded successfully:")
        print(f"   Samples: {len(test_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test model architecture"""
    print(f"\n🧪 TESTING MODEL ARCHITECTURE")
    print("=" * 40)
    
    try:
        # Test model initialization
        model = SimpleRegressionModel(
            fmri_dim=512, 
            image_size=28, 
            hidden_dims=[1024, 2048, 1024]
        )
        
        print(f"✅ Model initialized successfully:")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        fmri_emb = torch.randn(batch_size, 512)
        
        with torch.no_grad():
            output = model(fmri_emb)
        
        print(f"✅ Forward pass test:")
        print(f"   Input: {fmri_emb.shape}")
        print(f"   Output: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   Expected output shape: ({batch_size}, 1, 28, 28)")
        
        # Verify output shape
        expected_shape = (batch_size, 1, 28, 28)
        if output.shape == expected_shape:
            print(f"   ✅ Output shape correct: {output.shape}")
        else:
            print(f"   ❌ Output shape incorrect: {output.shape}, expected: {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test one training step"""
    print(f"\n🧪 TESTING TRAINING STEP")
    print("=" * 35)
    
    try:
        # Load small dataset
        train_dataset = MiyawakiBaselineDataset(
            model_path="miyawaki_contrastive_clip.pth",
            split="train", 
            target_size=28
        )
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Initialize model
        model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[512, 1024, 512])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Test one training step
        fmri_emb, images = next(iter(train_loader))
        
        print(f"✅ Training step test:")
        print(f"   Batch size: {fmri_emb.shape[0]}")
        print(f"   fMRI embeddings: {fmri_emb.shape}")
        print(f"   Target images: {images.shape}")
        
        # Forward pass
        predicted_images = model(fmri_emb)
        loss = criterion(predicted_images, images)
        
        print(f"   Predicted images: {predicted_images.shape}")
        print(f"   Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Backward pass: SUCCESS")
        print(f"   Optimizer step: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """Compare with original dataset structure"""
    print(f"\n📊 MIYAWAKI VS ORIGINAL COMPARISON")
    print("=" * 50)
    
    print(f"🔍 Original (Digit69) vs Miyawaki Dataset:")
    print(f"   📊 Original Dataset:")
    print(f"      - Source: digit69_embeddings.pkl")
    print(f"      - Pre-computed embeddings")
    print(f"      - EEG → CLIP space")
    print(f"      - Digit classification task")
    
    print(f"   🧠 Miyawaki Dataset:")
    print(f"      - Source: miyawaki_contrastive_clip.pth")
    print(f"      - Real-time embedding generation")
    print(f"      - fMRI → CLIP space")
    print(f"      - Natural image reconstruction")
    print(f"      - Training: 107 samples, Test: 12 samples")
    print(f"      - Image size: 28x28 grayscale")
    print(f"      - Embedding dim: 512 (CLIP space)")

def main():
    """Main test function"""
    print("🔍 MIYAWAKI BASELINE MODEL TEST")
    print("=" * 60)
    print("Testing modified baseline model with Miyawaki embeddings")
    print("=" * 60)
    
    # Run tests
    dataset_success = test_dataset_loading()
    model_success = test_model_architecture()
    training_success = test_training_step()
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    if dataset_success:
        print(f"   ✅ Dataset Loading: PASSED")
    else:
        print(f"   ❌ Dataset Loading: FAILED")
        
    if model_success:
        print(f"   ✅ Model Architecture: PASSED")
    else:
        print(f"   ❌ Model Architecture: FAILED")
        
    if training_success:
        print(f"   ✅ Training Step: PASSED")
    else:
        print(f"   ❌ Training Step: FAILED")
    
    # Show comparison
    compare_with_original()
    
    if all([dataset_success, model_success, training_success]):
        print(f"\n🚀 MIYAWAKI BASELINE MODEL READY!")
        print(f"\n📊 Key Features:")
        print(f"   ✅ Uses trained miyawaki_contrastive_clip.pth")
        print(f"   ✅ Real-time embedding generation from fMRI")
        print(f"   ✅ Direct fMRI embedding → Image regression")
        print(f"   ✅ 512-dim CLIP space embeddings")
        print(f"   ✅ 28x28 grayscale image output")
        print(f"   ✅ Compatible with existing training pipeline")
        
        print(f"\n🎯 Ready to run:")
        print(f"   python simple_baseline_model.py")
    else:
        print(f"\n❌ Please fix issues before running full training.")

if __name__ == "__main__":
    main()
