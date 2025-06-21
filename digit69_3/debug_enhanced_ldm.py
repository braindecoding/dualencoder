#!/usr/bin/env python3
"""
Debug Enhanced LDM loading issues
"""

import torch
import pickle

def debug_enhanced_ldm_checkpoint():
    """Debug Enhanced LDM checkpoint structure"""
    print("🔍 DEBUGGING ENHANCED LDM CHECKPOINT")
    print("=" * 50)
    
    # Load checkpoint
    print("📂 Loading Enhanced LDM checkpoint...")
    checkpoint = torch.load('enhanced_ldm_best.pth', map_location='cpu')
    
    print(f"📊 Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"📊 Model state dict type: {type(model_state)}")
            print(f"📋 Model state keys (first 10): {list(model_state.keys())[:10]}")
            print(f"📊 Total model parameters: {len(model_state.keys())}")
            
            # Check for UNet prefix
            unet_keys = [k for k in model_state.keys() if k.startswith('unet.')]
            non_unet_keys = [k for k in model_state.keys() if not k.startswith('unet.')]
            
            print(f"📊 UNet keys: {len(unet_keys)}")
            print(f"📊 Non-UNet keys: {len(non_unet_keys)}")
            
            if unet_keys:
                print(f"📋 Sample UNet keys: {unet_keys[:5]}")
            if non_unet_keys:
                print(f"📋 Sample Non-UNet keys: {non_unet_keys[:5]}")
                
        # Check other checkpoint info
        for key, value in checkpoint.items():
            if key != 'model_state_dict':
                print(f"📊 {key}: {type(value)} - {value if not torch.is_tensor(value) else f'Tensor {value.shape}'}")
    
    else:
        # Direct state dict
        print(f"📋 Direct state dict keys (first 10): {list(checkpoint.keys())[:10]}")
        print(f"📊 Total parameters: {len(checkpoint.keys())}")

def test_enhanced_ldm_loading():
    """Test different loading approaches"""
    print("\n🧪 TESTING ENHANCED LDM LOADING APPROACHES")
    print("=" * 50)
    
    from improved_unet import ImprovedUNet
    from enhanced_training_pipeline import EnhancedDiffusionModel
    
    # Create model
    print("🏗️ Creating Enhanced LDM model...")
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )
    
    model = EnhancedDiffusionModel(
        unet=unet,
        num_timesteps=1000
    )
    
    print(f"📊 Model created successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loading approaches
    checkpoint = torch.load('enhanced_ldm_best.pth', map_location='cpu')
    
    print("\n🔧 Approach 1: Direct loading")
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Approach 1 successful!")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Approach 1 successful!")
    except Exception as e:
        print(f"❌ Approach 1 failed: {str(e)[:100]}...")
    
    print("\n🔧 Approach 2: Remove 'unet.' prefix")
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'unet.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                new_key = key[5:]  # Remove 'unet.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.unet.load_state_dict(new_state_dict)
        print("✅ Approach 2 successful!")
    except Exception as e:
        print(f"❌ Approach 2 failed: {str(e)[:100]}...")
    
    print("\n🔧 Approach 3: Load only UNet part")
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Extract only UNet parameters
        unet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                new_key = key[5:]  # Remove 'unet.' prefix
                unet_state_dict[new_key] = value
        
        if unet_state_dict:
            model.unet.load_state_dict(unet_state_dict)
            print("✅ Approach 3 successful!")
        else:
            print("❌ Approach 3 failed: No UNet parameters found")
    except Exception as e:
        print(f"❌ Approach 3 failed: {str(e)[:100]}...")

def test_enhanced_ldm_sampling():
    """Test Enhanced LDM sampling with device fixes"""
    print("\n🎯 TESTING ENHANCED LDM SAMPLING")
    print("=" * 50)
    
    from improved_unet import ImprovedUNet
    from enhanced_training_pipeline import EnhancedDiffusionModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Create model
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )
    
    model = EnhancedDiffusionModel(
        unet=unet,
        num_timesteps=1000
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('enhanced_ldm_best.pth', map_location=device)
    
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Extract only UNet parameters and remove prefix
        unet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                new_key = key[5:]  # Remove 'unet.' prefix
                unet_state_dict[new_key] = value
        
        model.unet.load_state_dict(unet_state_dict)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Test sampling
        print("🎯 Testing sampling...")
        test_condition = torch.randn(1, 512).to(device)
        
        with torch.no_grad():
            # Test with fewer timesteps first
            sample = model.sample(test_condition, image_size=28)
            print(f"✅ Sampling successful! Output shape: {sample.shape}")
            print(f"📊 Output range: [{sample.min():.3f}, {sample.max():.3f}]")
            
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enhanced_ldm_checkpoint()
    test_enhanced_ldm_loading()
    test_enhanced_ldm_sampling()
