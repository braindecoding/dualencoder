#!/usr/bin/env python3
"""
Check Digit69 Data Structure
"""

import pickle
import numpy as np

def check_data_structure():
    """Check the structure of digit69_embeddings.pkl"""
    print("🔍 Checking Digit69 data structure...")
    
    try:
        with open('digit69_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📋 Dictionary keys: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n🔑 Key: '{key}'")
                print(f"   Type: {type(value)}")
                
                if isinstance(value, np.ndarray):
                    print(f"   Shape: {value.shape}")
                    print(f"   Dtype: {value.dtype}")
                    if len(value) > 0:
                        print(f"   First item shape: {value[0].shape if hasattr(value[0], 'shape') else 'No shape'}")
                        print(f"   First item type: {type(value[0])}")
                elif isinstance(value, list):
                    print(f"   Length: {len(value)}")
                    if len(value) > 0:
                        print(f"   First item type: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"   First item shape: {value[0].shape}")
                elif isinstance(value, dict):
                    print(f"   Sub-keys: {list(value.keys())}")
                    # Check sub-values too
                    for sub_key, sub_value in value.items():
                        print(f"     {sub_key}: {type(sub_value)}")
                        if isinstance(sub_value, np.ndarray):
                            print(f"       Shape: {sub_value.shape}, Dtype: {sub_value.dtype}")
                            if len(sub_value) > 0:
                                print(f"       First item: {type(sub_value[0])}, Shape: {sub_value[0].shape if hasattr(sub_value[0], 'shape') else 'No shape'}")
                else:
                    print(f"   Value: {value}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

if __name__ == "__main__":
    data = check_data_structure()
