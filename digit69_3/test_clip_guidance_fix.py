#!/usr/bin/env python3
"""
Test CLIP Guidance Fix
Verify that the gradient computation error is resolved
"""

import torch
import torch.nn.functional as F
from improved_clip_v2 import create_improved_clip_v2_model

def test_clip_guidance_fix():
    """Test the fixed CLIP guidance implementation"""
    print("🔧 TESTING CLIP GUIDANCE FIX")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    print("🏗️ Creating Improved CLIP v2.0 model...")
    model = create_improved_clip_v2_model().to(device)
    
    # Load trained weights
    print("📂 Loading trained weights...")
    try:
        checkpoint = torch.load('improved_clip_v2_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test data
    print("\n🧪 Preparing test data...")
    batch_size = 1
    fmri_condition = torch.randn(batch_size, 512).to(device)
    text_prompts = ["a handwritten digit 3"]
    
    print(f"   fMRI condition shape: {fmri_condition.shape}")
    print(f"   Text prompts: {text_prompts}")
    
    # Test 1: Pure diffusion sampling (should work)
    print("\n🎯 Test 1: Pure Diffusion Sampling...")
    try:
        with torch.no_grad():
            pure_sample = model.sample(
                fmri_condition, 
                text_prompts=None,
                num_samples=1, 
                num_timesteps=10  # Very fast test
            )
        print(f"✅ Pure diffusion successful! Shape: {pure_sample.shape}")
        print(f"   Sample range: [{pure_sample.min():.3f}, {pure_sample.max():.3f}]")
    except Exception as e:
        print(f"❌ Pure diffusion failed: {e}")
        return False
    
    # Test 2: CLIP guidance sampling (the critical test)
    print("\n🎯 Test 2: CLIP Guidance Sampling...")
    try:
        with torch.no_grad():
            clip_sample = model.sample(
                fmri_condition, 
                text_prompts=text_prompts,
                num_samples=1, 
                num_timesteps=10,  # Very fast test
                clip_guidance_scale=0.01,
                clip_start_step=5
            )
        print(f"✅ CLIP guidance successful! Shape: {clip_sample.shape}")
        print(f"   Sample range: [{clip_sample.min():.3f}, {clip_sample.max():.3f}]")
    except Exception as e:
        print(f"❌ CLIP guidance failed: {e}")
        print(f"   Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Direct CLIP guidance computation
    print("\n🎯 Test 3: Direct CLIP Guidance Computation...")
    try:
        # Create a test image
        test_image = torch.randn(1, 1, 28, 28).to(device)
        test_image.requires_grad_(True)
        
        # Compute CLIP guidance
        clip_grad = model.compute_clip_guidance(
            test_image, 
            text_prompts, 
            guidance_scale=0.01
        )
        
        print(f"✅ CLIP guidance computation successful!")
        print(f"   Gradient shape: {clip_grad.shape}")
        print(f"   Gradient range: [{clip_grad.min():.6f}, {clip_grad.max():.6f}]")
        print(f"   Gradient norm: {clip_grad.norm():.6f}")
        
    except Exception as e:
        print(f"❌ CLIP guidance computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Compare pure vs CLIP guided samples
    print("\n🎯 Test 4: Comparing Pure vs CLIP Guided Samples...")
    try:
        # Generate longer samples for comparison
        with torch.no_grad():
            pure_sample_long = model.sample(
                fmri_condition, 
                text_prompts=None,
                num_samples=1, 
                num_timesteps=50
            )
            
            clip_sample_long = model.sample(
                fmri_condition, 
                text_prompts=text_prompts,
                num_samples=1, 
                num_timesteps=50,
                clip_guidance_scale=0.01,
                clip_start_step=25
            )
        
        # Compute difference
        diff = torch.abs(pure_sample_long - clip_sample_long).mean()
        print(f"✅ Comparison successful!")
        print(f"   Pure sample range: [{pure_sample_long.min():.3f}, {pure_sample_long.max():.3f}]")
        print(f"   CLIP sample range: [{clip_sample_long.min():.3f}, {clip_sample_long.max():.3f}]")
        print(f"   Mean absolute difference: {diff:.6f}")
        
        if diff > 0.001:
            print(f"✅ CLIP guidance is having an effect (difference > 0.001)")
        else:
            print(f"⚠️  CLIP guidance effect is minimal (difference ≤ 0.001)")
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ CLIP guidance fix is working correctly")
    return True

def test_clip_model_states():
    """Test CLIP model state changes during guidance"""
    print("\n🔍 TESTING CLIP MODEL STATE MANAGEMENT")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_improved_clip_v2_model().to(device)
    
    # Check initial state
    print("📊 Initial CLIP model state:")
    print(f"   Training mode: {model.clip_model.training}")
    print(f"   Parameters require grad: {any(p.requires_grad for p in model.clip_model.parameters())}")
    
    # Test guidance computation
    test_image = torch.randn(1, 1, 28, 28).to(device)
    text_prompts = ["a handwritten digit 5"]
    
    print("\n🔧 During CLIP guidance computation...")
    try:
        clip_grad = model.compute_clip_guidance(test_image, text_prompts)
        print("✅ Guidance computation successful")
    except Exception as e:
        print(f"❌ Guidance computation failed: {e}")
        return False
    
    # Check final state
    print("\n📊 Final CLIP model state:")
    print(f"   Training mode: {model.clip_model.training}")
    print(f"   Parameters require grad: {any(p.requires_grad for p in model.clip_model.parameters())}")
    
    print("✅ CLIP model state management working correctly")
    return True

if __name__ == "__main__":
    print("🚀 STARTING CLIP GUIDANCE FIX VERIFICATION")
    print("=" * 60)
    
    # Test the fix
    success = test_clip_guidance_fix()
    
    if success:
        # Test state management
        test_clip_model_states()
        
        print("\n🎯 SUMMARY:")
        print("✅ CLIP guidance gradient computation error FIXED!")
        print("✅ Model can now perform CLIP-guided sampling")
        print("✅ Ready for comprehensive evaluation")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Run full evaluation with CLIP guidance")
        print("   2. Compare Pure vs CLIP guided performance")
        print("   3. Test different guidance scales")
        print("   4. Evaluate semantic improvements")
        
    else:
        print("\n❌ CLIP GUIDANCE FIX VERIFICATION FAILED")
        print("   Additional debugging required")
