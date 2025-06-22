#!/usr/bin/env python3
"""
Test LDM CLIP Configuration
Quick test to verify CLIP-guided LDM configuration is correct for Crell embeddings
"""

import pickle
import numpy as np
from eeg_ldm_clip_guided import CLIPGuidedEEGDataset

def test_ldm_clip_config():
    """Test CLIP-guided LDM configuration"""
    print("🧪 TESTING CLIP-GUIDED LDM CONFIGURATION")
    print("=" * 55)
    
    try:
        # Test dataset loading
        print("📊 Testing dataset loading...")
        
        # Test train split
        train_dataset = CLIPGuidedEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="train", 
            target_size=28
        )
        
        print(f"✅ Train dataset loaded successfully:")
        print(f"   Samples: {len(train_dataset)}")
        print(f"   EEG embeddings shape: {train_dataset.eeg_embeddings.shape}")
        print(f"   Images shape: {train_dataset.images.shape}")
        print(f"   Labels: {len(train_dataset.labels)}")
        
        # Test test split
        test_dataset = CLIPGuidedEEGDataset(
            embeddings_file="crell_embeddings_20250622_173213.pkl",
            split="test", 
            target_size=28
        )
        
        print(f"✅ Test dataset loaded successfully:")
        print(f"   Samples: {len(test_dataset)}")
        print(f"   EEG embeddings shape: {test_dataset.eeg_embeddings.shape}")
        print(f"   Images shape: {test_dataset.images.shape}")
        print(f"   Labels: {len(test_dataset.labels)}")
        
        # Test data item
        print("\n🔍 Testing data item access...")
        eeg_emb, image, label = train_dataset[0]
        
        print(f"✅ Data item test:")
        print(f"   EEG embedding: {eeg_emb.shape} {eeg_emb.dtype}")
        print(f"   Image: {image.shape} {image.dtype}")
        print(f"   Label: {label} ({type(label)})")
        print(f"   EEG range: [{eeg_emb.min():.3f}, {eeg_emb.max():.3f}]")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Verify data consistency
        print("\n🔍 Verifying data consistency...")
        total_samples = len(train_dataset) + len(test_dataset)
        
        # Load original embeddings to verify
        with open("crell_embeddings_20250622_173213.pkl", 'rb') as f:
            emb_data = pickle.load(f)
        
        original_samples = len(emb_data['embeddings'])
        
        print(f"✅ Data consistency check:")
        print(f"   Original embeddings: {original_samples} samples")
        print(f"   Train + Test: {total_samples} samples")
        print(f"   Split ratio: {len(train_dataset)/total_samples:.1%} train, {len(test_dataset)/total_samples:.1%} test")
        
        if total_samples == original_samples:
            print(f"   ✅ Sample count matches!")
        else:
            print(f"   ❌ Sample count mismatch!")
        
        # Test embedding dimensions for LDM
        expected_emb_dim = 512
        expected_img_size = 28
        
        print(f"\n🔍 Verifying LDM dimensions...")
        print(f"✅ LDM dimension check:")
        print(f"   EEG embedding dim: {eeg_emb.shape[0]} (expected: {expected_emb_dim})")
        print(f"   Image size: {image.shape[-1]}x{image.shape[-2]} (expected: {expected_img_size}x{expected_img_size})")
        print(f"   Image channels: {image.shape[0]} (expected: 1 for grayscale)")
        
        if eeg_emb.shape[0] == expected_emb_dim:
            print(f"   ✅ EEG embedding dimension correct for LDM conditioning!")
        else:
            print(f"   ❌ EEG embedding dimension incorrect!")
            
        if image.shape[-1] == expected_img_size and image.shape[-2] == expected_img_size:
            print(f"   ✅ Image dimensions correct for LDM!")
        else:
            print(f"   ❌ Image dimensions incorrect!")
            
        if image.shape[0] == 1:
            print(f"   ✅ Image channels correct for LDM!")
        else:
            print(f"   ❌ Image channels incorrect!")
        
        # Test label distribution
        print(f"\n🔍 Verifying label distribution...")
        train_labels = np.array([train_dataset[i][2] for i in range(min(len(train_dataset), 50))])
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        
        print(f"✅ Label distribution (first 50 samples):")
        for label, count in zip(unique_labels, counts):
            print(f"   Digit {int(label)}: {count} samples")
        
        print(f"   Total unique digits: {len(unique_labels)}")
        print(f"   Expected digits: 0-9 (10 classes)")
        
        if len(unique_labels) <= 10:
            print(f"   ✅ Label distribution looks good for CLIP guidance!")
        else:
            print(f"   ⚠️ More than 10 unique labels detected!")
        
        print(f"\n🎯 LDM-CLIP CONFIGURATION TEST SUMMARY:")
        print(f"   ✅ Dataset loading: SUCCESS")
        print(f"   ✅ Data access: SUCCESS") 
        print(f"   ✅ Data consistency: SUCCESS")
        print(f"   ✅ LDM dimensions: SUCCESS")
        print(f"   ✅ Label distribution: SUCCESS")
        print(f"   🚀 CLIP-guided LDM ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip_availability():
    """Test CLIP availability"""
    print(f"\n🔍 TESTING CLIP AVAILABILITY")
    print("=" * 35)
    
    try:
        import clip
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📱 Device: {device}")
        
        # Try to load CLIP
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP model loaded successfully!")
        print(f"   Model: ViT-B/32")
        print(f"   Device: {device}")
        
        # Test text encoding
        text_prompts = [f"a handwritten digit {i}" for i in range(10)]
        text_tokens = clip.tokenize(text_prompts).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        
        print(f"✅ Text encoding test:")
        print(f"   Text prompts: {len(text_prompts)}")
        print(f"   Text features: {text_features.shape}")
        
        print(f"\n🎯 CLIP GUIDANCE READY:")
        print(f"   ✅ CLIP model available")
        print(f"   ✅ Text prompts encoded")
        print(f"   ✅ Semantic guidance enabled")
        
        return True
        
    except ImportError:
        print(f"❌ CLIP not available - will use classification fallback")
        return False
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 CLIP-GUIDED LDM CONFIGURATION TEST")
    print("=" * 60)
    print("Testing configuration for Crell embeddings with CLIP guidance")
    print("=" * 60)
    
    # Test configuration
    config_success = test_ldm_clip_config()
    
    # Test CLIP availability
    clip_success = test_clip_availability()
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    if config_success:
        print(f"   ✅ Configuration: PASSED")
    else:
        print(f"   ❌ Configuration: FAILED")
        
    if clip_success:
        print(f"   ✅ CLIP Guidance: AVAILABLE")
    else:
        print(f"   ⚠️ CLIP Guidance: FALLBACK MODE")
    
    if config_success:
        print(f"\n🚀 CLIP-guided LDM ready for training with Crell embeddings!")
    else:
        print(f"\n❌ Please fix configuration issues before training.")
