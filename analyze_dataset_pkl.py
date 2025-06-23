#!/usr/bin/env python3
"""
Analyze dataset .pkl files to understand data structure and dimensions
"""

import pickle
import numpy as np
import os

def analyze_pkl_file(filepath, dataset_name):
    """Analyze a single .pkl file"""
    print(f"\n{'='*50}")
    print(f"=== {dataset_name.upper()} DATASET ANALYSIS ===")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📁 File: {filepath}")
        print(f"🔑 Keys: {list(data.keys())}")
        print(f"📊 Data Structure:")
        
        total_samples = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
                if 'embedding' in key.lower() or 'eeg' in key.lower():
                    total_samples = max(total_samples, value.shape[0])
            elif isinstance(value, (list, tuple)):
                print(f"   {key}: {type(value).__name__} with {len(value)} items")
                if len(value) > 0:
                    print(f"      First item type: {type(value[0])}")
                    if hasattr(value[0], 'shape'):
                        print(f"      First item shape: {value[0].shape}")
            else:
                print(f"   {key}: {type(value)} - {value}")
        
        print(f"📈 Estimated total samples: {total_samples}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")

def main():
    """Main analysis function"""
    print("🧠 BRAIN-TO-IMAGE DATASET ANALYSIS")
    print("=" * 60)
    
    # Dataset files to analyze
    datasets = [
        ("mbd3/eeg_embeddings_enhanced_20250622_123559.pkl", "MBD3 EEG"),
        ("crell3/crell_embeddings_20250622_173213.pkl", "Crell3 EEG"),
        ("digit69_3/digit69_embeddings.pkl", "Digit69_3 EEG"),
        ("miyawaki4/miyawaki4_embeddings.pkl", "Miyawaki4 fMRI"),
        ("mbd3/explicit_eeg_data_splits.pkl", "MBD3 Data Splits"),
        ("crell3/crell_processed_data_correct.pkl", "Crell3 Processed"),
    ]
    
    for filepath, name in datasets:
        analyze_pkl_file(filepath, name)
    
    print(f"\n{'='*60}")
    print("📋 SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    # Try to load and compare key metrics
    try:
        # MBD3
        with open('mbd3/eeg_embeddings_enhanced_20250622_123559.pkl', 'rb') as f:
            mbd3 = pickle.load(f)
        
        # Crell3
        with open('crell3/crell_embeddings_20250622_173213.pkl', 'rb') as f:
            crell3 = pickle.load(f)
        
        # Digit69_3
        with open('digit69_3/digit69_embeddings.pkl', 'rb') as f:
            digit69 = pickle.load(f)
        
        # Miyawaki4
        with open('miyawaki4/miyawaki4_embeddings.pkl', 'rb') as f:
            miyawaki = pickle.load(f)
        
        print("\n📊 DATASET COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Dataset':<15} {'Total Samples':<15} {'Input Dim':<12} {'Signal Type':<12} {'Target'}")
        print("-" * 80)
        
        # Extract info for each dataset
        datasets_info = []
        
        # MBD3
        if 'eeg_embeddings' in mbd3:
            mbd3_samples = mbd3['eeg_embeddings'].shape[0]
            mbd3_dim = mbd3['eeg_embeddings'].shape[1]
            datasets_info.append(("MBD3", mbd3_samples, mbd3_dim, "EEG", "Digits"))
        
        # Crell3
        if 'eeg_embeddings' in crell3:
            crell3_samples = crell3['eeg_embeddings'].shape[0]
            crell3_dim = crell3['eeg_embeddings'].shape[1]
            datasets_info.append(("Crell3", crell3_samples, crell3_dim, "EEG", "Letters"))
        
        # Digit69_3
        if 'eeg_embeddings' in digit69:
            digit69_samples = digit69['eeg_embeddings'].shape[0]
            digit69_dim = digit69['eeg_embeddings'].shape[1]
            datasets_info.append(("Digit69_3", digit69_samples, digit69_dim, "EEG", "Digits"))
        
        # Miyawaki4
        if 'fmri_embeddings' in miyawaki:
            miyawaki_samples = miyawaki['fmri_embeddings'].shape[0]
            miyawaki_dim = miyawaki['fmri_embeddings'].shape[1]
            datasets_info.append(("Miyawaki4", miyawaki_samples, miyawaki_dim, "fMRI", "Stimuli"))
        
        # Print comparison table
        for name, samples, dim, signal, target in datasets_info:
            print(f"{name:<15} {samples:<15} {dim:<12} {signal:<12} {target}")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

if __name__ == "__main__":
    main()
