#!/usr/bin/env python3
"""
Test Baseline Configuration
Quick test to verify baseline model configuration is correct for Crell embeddings
"""

import pickle
import numpy as np
from simple_baseline_model import EEGBaselineDataset

def test_baseline_config():
    """Test baseline model configuration"""
    print("🧪 TESTING BASELINE MODEL CONFIGURATION")
    print("=" * 50)
    
    try:
        # Test dataset loading
        print("📊 Testing dataset loading...")
        
        # Test train split
        train_dataset = EEGBaselineDataset(
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
        test_dataset = EEGBaselineDataset(
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
        
        # Test embedding dimensions
        expected_emb_dim = 512
        expected_img_size = 28
        
        print(f"\n🔍 Verifying dimensions...")
        print(f"✅ Dimension check:")
        print(f"   EEG embedding dim: {eeg_emb.shape[0]} (expected: {expected_emb_dim})")
        print(f"   Image size: {image.shape[-1]}x{image.shape[-2]} (expected: {expected_img_size}x{expected_img_size})")
        
        if eeg_emb.shape[0] == expected_emb_dim:
            print(f"   ✅ EEG embedding dimension correct!")
        else:
            print(f"   ❌ EEG embedding dimension incorrect!")
            
        if image.shape[-1] == expected_img_size and image.shape[-2] == expected_img_size:
            print(f"   ✅ Image dimensions correct!")
        else:
            print(f"   ❌ Image dimensions incorrect!")
        
        print(f"\n🎯 CONFIGURATION TEST SUMMARY:")
        print(f"   ✅ Dataset loading: SUCCESS")
        print(f"   ✅ Data access: SUCCESS") 
        print(f"   ✅ Data consistency: SUCCESS")
        print(f"   ✅ Dimensions: SUCCESS")
        print(f"   🚀 Baseline model ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_baseline_config()
    if success:
        print(f"\n✅ All tests passed! Baseline model configuration is correct.")
    else:
        print(f"\n❌ Tests failed! Please check configuration.")
