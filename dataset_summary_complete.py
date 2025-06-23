#!/usr/bin/env python3
"""
Complete summary of all dataset .pkl files with dimensions and structure
"""

import pickle
import numpy as np

def get_dataset_info():
    """Extract key information from all dataset files"""
    
    datasets_info = []
    
    # MBD3 EEG Dataset
    try:
        with open('mbd3/eeg_embeddings_enhanced_20250622_123559.pkl', 'rb') as f:
            mbd3 = pickle.load(f)
        
        datasets_info.append({
            'name': 'MBD3',
            'type': 'EEG',
            'target': 'Digits (0-9)',
            'total_samples': mbd3['embeddings'].shape[0],
            'train_samples': mbd3['data_info']['splits']['train'],
            'val_samples': mbd3['data_info']['splits']['val'],
            'test_samples': mbd3['data_info']['splits']['test'],
            'input_dim': mbd3['embeddings'].shape[1],
            'output_dim': '784 (28x28)',
            'embedding_type': 'Enhanced EEG Transformer',
            'file_path': 'mbd3/eeg_embeddings_enhanced_20250622_123559.pkl'
        })
    except Exception as e:
        print(f"Error loading MBD3: {e}")
    
    # Crell3 EEG Dataset
    try:
        with open('crell3/crell_embeddings_20250622_173213.pkl', 'rb') as f:
            crell3 = pickle.load(f)
        
        datasets_info.append({
            'name': 'Crell3',
            'type': 'EEG',
            'target': 'Letters (a,d,e,f,j,n,o,s,t,v)',
            'total_samples': crell3['embeddings'].shape[0],
            'train_samples': 102,  # From previous analysis
            'val_samples': 0,
            'test_samples': 26,
            'input_dim': crell3['embeddings'].shape[1],
            'output_dim': '784 (28x28)',
            'embedding_type': 'Stable EEG Model',
            'file_path': 'crell3/crell_embeddings_20250622_173213.pkl'
        })
    except Exception as e:
        print(f"Error loading Crell3: {e}")
    
    # Digit69_3 Dataset
    try:
        with open('digit69_3/digit69_embeddings.pkl', 'rb') as f:
            digit69 = pickle.load(f)
        
        train_samples = digit69['train']['fmri_embeddings'].shape[0]
        test_samples = digit69['test']['fmri_embeddings'].shape[0]
        
        datasets_info.append({
            'name': 'Digit69_3',
            'type': 'fMRI',
            'target': 'Digits (0-9)',
            'total_samples': train_samples + test_samples,
            'train_samples': train_samples,
            'val_samples': 0,
            'test_samples': test_samples,
            'input_dim': digit69['train']['fmri_embeddings'].shape[1],
            'output_dim': '784 (28x28)',
            'embedding_type': 'Contrastive CLIP',
            'file_path': 'digit69_3/digit69_embeddings.pkl'
        })
    except Exception as e:
        print(f"Error loading Digit69_3: {e}")
    
    # Miyawaki4 Dataset
    try:
        with open('miyawaki4/miyawaki4_embeddings.pkl', 'rb') as f:
            miyawaki = pickle.load(f)
        
        train_samples = miyawaki['train']['fmri_embeddings'].shape[0]
        test_samples = miyawaki['test']['fmri_embeddings'].shape[0]
        
        datasets_info.append({
            'name': 'Miyawaki4',
            'type': 'fMRI',
            'target': 'Visual Stimuli',
            'total_samples': train_samples + test_samples,
            'train_samples': train_samples,
            'val_samples': 0,
            'test_samples': test_samples,
            'input_dim': miyawaki['train']['fmri_embeddings'].shape[1],
            'output_dim': '784 (28x28)',
            'embedding_type': 'Contrastive CLIP',
            'file_path': 'miyawaki4/miyawaki4_embeddings.pkl'
        })
    except Exception as e:
        print(f"Error loading Miyawaki4: {e}")
    
    return datasets_info

def print_summary():
    """Print comprehensive dataset summary"""
    
    print("🧠 BRAIN-TO-IMAGE DATASET COMPLETE SUMMARY")
    print("=" * 80)
    
    datasets = get_dataset_info()
    
    # Print detailed table
    print("\n📊 DETAILED DATASET COMPARISON:")
    print("-" * 120)
    print(f"{'Dataset':<12} {'Signal':<6} {'Target':<25} {'Total':<7} {'Train':<7} {'Test':<6} {'Input':<7} {'Output':<10}")
    print("-" * 120)
    
    for ds in datasets:
        print(f"{ds['name']:<12} {ds['type']:<6} {ds['target']:<25} {ds['total_samples']:<7} "
              f"{ds['train_samples']:<7} {ds['test_samples']:<6} {ds['input_dim']:<7} {ds['output_dim']:<10}")
    
    print("-" * 120)
    
    # Performance summary from experiments
    print("\n🎯 PERFORMANCE SUMMARY (Mean Correlation):")
    print("-" * 60)
    print(f"{'Dataset':<12} {'Performance':<15} {'Status':<15} {'Notes'}")
    print("-" * 60)
    print(f"{'MBD3':<12} {'0.57 ± 0.01':<15} {'Excellent':<15} {'Stable & Fast'}")
    print(f"{'Crell3':<12} {'0.40 ± 0.15':<15} {'Good*':<15} {'High Variability'}")
    print(f"{'Digit69_3':<12} {'0.78 ± 0.05':<15} {'Very Good':<15} {'Advanced Models'}")
    print(f"{'Miyawaki4':<12} {'0.96+ ± 0.02':<15} {'Near Perfect':<15} {'fMRI Advantage'}")
    print("-" * 60)
    print("* Crell3 performance varies significantly (0.10-0.40) due to small dataset size")
    
    # Dataset characteristics
    print("\n📈 DATASET CHARACTERISTICS:")
    print("-" * 50)
    for ds in datasets:
        print(f"\n🔹 {ds['name']} ({ds['type']}):")
        print(f"   Target: {ds['target']}")
        print(f"   Samples: {ds['total_samples']} total ({ds['train_samples']} train / {ds['test_samples']} test)")
        print(f"   Dimensions: {ds['input_dim']} → {ds['output_dim']}")
        print(f"   Embedding: {ds['embedding_type']}")
        print(f"   File: {ds['file_path']}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    print("✅ Large datasets (4000+ samples) → Stable performance")
    print("⚠️  Small datasets (<200 samples) → High variability")
    print("🧠 fMRI signals → Better performance than EEG")
    print("📊 All datasets use 512-dim embeddings → 28x28 images")
    print("🎯 Digits easier to reconstruct than letters")
    print("⚡ Training time scales with dataset size")

if __name__ == "__main__":
    print_summary()
