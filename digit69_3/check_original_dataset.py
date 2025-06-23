#!/usr/bin/env python3
"""
Check Original Digit69 Dataset for Real Labels
"""

import scipy.io as sio
import numpy as np

def check_original_dataset():
    """Check the original .mat file for labels"""
    print("🔍 Checking original digit69_28x28.mat dataset...")
    
    try:
        # Load original dataset
        mat_data = sio.loadmat('../dataset/digit69_28x28.mat')
        
        print(f"✅ Original dataset loaded!")
        print(f"📋 Keys in .mat file: {list(mat_data.keys())}")
        
        # Check each key
        for key, value in mat_data.items():
            if not key.startswith('__'):
                print(f"\n🔑 Key: '{key}'")
                print(f"   Type: {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    
                    # Check for label-like data
                    if 'label' in key.lower() or 'class' in key.lower() or 'digit' in key.lower():
                        print(f"   🎯 POTENTIAL LABELS FOUND!")
                        print(f"   Unique values: {np.unique(value)}")
                        print(f"   Value counts: {np.bincount(value.flatten().astype(int)) if value.dtype in [np.int32, np.int64] else 'N/A'}")
                    
                    # Check if it's a small integer array (could be labels)
                    if value.dtype in [np.int32, np.int64] and value.size < 1000:
                        unique_vals = np.unique(value)
                        if len(unique_vals) <= 10 and unique_vals.max() <= 9:
                            print(f"   🎯 POSSIBLE DIGIT LABELS!")
                            print(f"   Unique values: {unique_vals}")
                            print(f"   First 20 values: {value.flatten()[:20]}")
        
        return mat_data
        
    except Exception as e:
        print(f"❌ Error loading original dataset: {e}")
        return None

if __name__ == "__main__":
    data = check_original_dataset()
