#!/usr/bin/env python3
"""
Check data structure in embeddings file
"""

import pickle
import numpy as np

def check_data_structure():
    """Check the actual structure of embeddings data"""
    print("🔍 CHECKING DATA STRUCTURE")
    print("=" * 40)
    
    with open("digit69_embeddings.pkl", 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Top level keys: {list(data.keys())}")
    
    for key in data.keys():
        print(f"\n📁 {key}:")
        if isinstance(data[key], dict):
            print(f"   Type: dict")
            print(f"   Keys: {list(data[key].keys())}")
            
            for subkey in data[key].keys():
                item = data[key][subkey]
                if isinstance(item, np.ndarray):
                    print(f"   {subkey}: {item.shape} {item.dtype}")
                else:
                    print(f"   {subkey}: {type(item)}")
        else:
            print(f"   Type: {type(data[key])}")
            if hasattr(data[key], 'shape'):
                print(f"   Shape: {data[key].shape}")

if __name__ == "__main__":
    check_data_structure()
