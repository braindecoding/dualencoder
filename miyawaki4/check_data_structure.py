#!/usr/bin/env python3
"""
Check Data Structure in Miyawaki4
Understand what data and embeddings are available
"""

import numpy as np
import pickle
import json
from pathlib import Path
import torch
from scipy.io import loadmat

def check_original_dataset():
    """Check original Miyawaki dataset"""
    print("🔍 CHECKING ORIGINAL MIYAWAKI DATASET")
    print("=" * 60)
    
    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")
    
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
        
        # Load dataset
        data = loadmat(dataset_path)
        
        print(f"\n📊 Dataset Contents:")
        for key, value in data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape} {value.dtype}")
                else:
                    print(f"   {key}: {type(value)}")
        
        # Detailed analysis
        if 'fmriTrn' in data:
            fmri_train = data['fmriTrn']
            stim_train = data['stimTrn']
            fmri_test = data['fmriTest']
            stim_test = data['stimTest']
            
            print(f"\n📈 Training Data:")
            print(f"   fMRI: {fmri_train.shape} - {fmri_train.dtype}")
            print(f"   Stimuli: {stim_train.shape} - {stim_train.dtype}")
            print(f"   fMRI range: [{fmri_train.min():.3f}, {fmri_train.max():.3f}]")
            print(f"   Stimuli range: [{stim_train.min():.3f}, {stim_train.max():.3f}]")
            
            print(f"\n📉 Test Data:")
            print(f"   fMRI: {fmri_test.shape} - {fmri_test.dtype}")
            print(f"   Stimuli: {stim_test.shape} - {stim_test.dtype}")
            print(f"   fMRI range: [{fmri_test.min():.3f}, {fmri_test.max():.3f}]")
            print(f"   Stimuli range: [{stim_test.min():.3f}, {stim_test.max():.3f}]")
            
            return {
                'fmri_train': fmri_train,
                'stim_train': stim_train,
                'fmri_test': fmri_test,
                'stim_test': stim_test
            }
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        return None

def check_trained_model():
    """Check trained model"""
    print("\n🤖 CHECKING TRAINED MODEL")
    print("=" * 40)
    
    model_path = Path("miyawaki_contrastive_clip.pth")
    
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"\n📦 Model Contents:")
        for key, value in checkpoint.items():
            if key == 'fmri_encoder_state_dict':
                print(f"   {key}: {len(value)} layers")
                total_params = sum(p.numel() for p in value.values())
                print(f"      Total parameters: {total_params:,}")
            else:
                print(f"   {key}: {type(value)}")
        
        return checkpoint
    else:
        print(f"❌ Model not found: {model_path}")
        return None

def check_embeddings():
    """Check generated embeddings"""
    print("\n🔗 CHECKING GENERATED EMBEDDINGS")
    print("=" * 45)
    
    # Check metadata
    metadata_path = Path("miyawaki4_embeddings_metadata.json")
    if metadata_path.exists():
        print(f"✅ Metadata found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📋 Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Metadata not found: {metadata_path}")
        metadata = None
    
    # Check pickle embeddings
    pkl_path = Path("miyawaki4_embeddings.pkl")
    if pkl_path.exists():
        print(f"\n✅ Pickle embeddings found: {pkl_path}")
        print(f"   Size: {pkl_path.stat().st_size / (1024*1024):.2f} MB")
        
        with open(pkl_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        print(f"\n📊 Pickle Contents:")
        for split in ['train', 'test']:
            if split in embeddings_data:
                print(f"   {split.upper()}:")
                for key, value in embeddings_data[split].items():
                    if hasattr(value, 'shape'):
                        print(f"      {key}: {value.shape} {value.dtype}")
                        if 'embedding' in key:
                            print(f"         Range: [{value.min():.3f}, {value.max():.3f}]")
                            print(f"         Mean: {value.mean():.3f}, Std: {value.std():.3f}")
    else:
        print(f"❌ Pickle embeddings not found: {pkl_path}")
        embeddings_data = None
    
    # Check numpy embeddings
    npz_path = Path("miyawaki4_embeddings.npz")
    if npz_path.exists():
        print(f"\n✅ Numpy embeddings found: {npz_path}")
        print(f"   Size: {npz_path.stat().st_size / (1024*1024):.2f} MB")
        
        npz_data = np.load(npz_path)
        
        print(f"\n📊 Numpy Contents:")
        for key in npz_data.files:
            value = npz_data[key]
            print(f"   {key}: {value.shape} {value.dtype}")
            if 'emb' in key:
                print(f"      Range: [{value.min():.3f}, {value.max():.3f}]")
                print(f"      Mean: {value.mean():.3f}, Std: {value.std():.3f}")
    else:
        print(f"❌ Numpy embeddings not found: {npz_path}")
        npz_data = None
    
    return metadata, embeddings_data, npz_data

def analyze_embedding_quality(embeddings_data):
    """Analyze embedding quality"""
    if embeddings_data is None:
        print("\n❌ No embeddings data to analyze")
        return
    
    print("\n🔍 ANALYZING EMBEDDING QUALITY")
    print("=" * 45)
    
    # Get embeddings
    train_fmri = embeddings_data['train']['fmri_embeddings']
    train_image = embeddings_data['train']['image_embeddings']
    test_fmri = embeddings_data['test']['fmri_embeddings']
    test_image = embeddings_data['test']['image_embeddings']
    
    # Calculate similarities
    train_similarities = np.sum(train_fmri * train_image, axis=1)
    test_similarities = np.sum(test_fmri * test_image, axis=1)
    
    print(f"📊 Embedding Statistics:")
    print(f"   Training samples: {len(train_fmri)}")
    print(f"   Test samples: {len(test_fmri)}")
    print(f"   Embedding dimension: {train_fmri.shape[1]}")
    
    print(f"\n🔗 Similarity Analysis:")
    print(f"   Training similarities:")
    print(f"      Mean: {train_similarities.mean():.4f}")
    print(f"      Std: {train_similarities.std():.4f}")
    print(f"      Range: [{train_similarities.min():.4f}, {train_similarities.max():.4f}]")
    
    print(f"   Test similarities:")
    print(f"      Mean: {test_similarities.mean():.4f}")
    print(f"      Std: {test_similarities.std():.4f}")
    print(f"      Range: [{test_similarities.min():.4f}, {test_similarities.max():.4f}]")
    
    # Cross-modal retrieval test
    print(f"\n🎯 Cross-Modal Retrieval Test:")
    
    # Calculate similarity matrix for test set
    similarity_matrix = np.dot(test_fmri, test_image.T)
    
    # Top-k accuracy
    for k in [1, 3, 5]:
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
        correct = 0
        for i in range(len(test_fmri)):
            if i in top_k_indices[i]:
                correct += 1
        accuracy = correct / len(test_fmri)
        print(f"   Top-{k} Accuracy: {accuracy:.4f} ({correct}/{len(test_fmri)})")

def check_available_files():
    """Check all available files in miyawaki4"""
    print("\n📁 AVAILABLE FILES IN MIYAWAKI4")
    print("=" * 50)
    
    current_dir = Path(".")
    
    # Group files by type
    file_types = {
        'Models': ['.pth'],
        'Embeddings': ['.pkl', '.npz'],
        'Metadata': ['.json'],
        'Images': ['.png', '.jpg'],
        'Scripts': ['.py'],
        'Data': ['.csv', '.md']
    }
    
    for file_type, extensions in file_types.items():
        print(f"\n{file_type}:")
        files_found = []
        for ext in extensions:
            files_found.extend(list(current_dir.glob(f"*{ext}")))
        
        if files_found:
            for file_path in sorted(files_found):
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"   ✅ {file_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ No {file_type.lower()} found")

def main():
    """Main function to check all data structures"""
    print("🔍 MIYAWAKI4 DATA STRUCTURE ANALYSIS")
    print("=" * 70)
    print("Understanding what data and embeddings are available")
    print("=" * 70)
    
    # Check available files
    check_available_files()
    
    # Check original dataset
    original_data = check_original_dataset()
    
    # Check trained model
    model_data = check_trained_model()
    
    # Check embeddings
    metadata, embeddings_data, npz_data = check_embeddings()
    
    # Analyze embedding quality
    analyze_embedding_quality(embeddings_data)
    
    print("\n" + "=" * 70)
    print("✅ DATA STRUCTURE ANALYSIS COMPLETED!")
    print("=" * 70)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   🗂️  Original Dataset: {'✅ Available' if original_data else '❌ Missing'}")
    print(f"   🤖 Trained Model: {'✅ Available' if model_data else '❌ Missing'}")
    print(f"   🔗 Embeddings (PKL): {'✅ Available' if embeddings_data else '❌ Missing'}")
    print(f"   📊 Embeddings (NPZ): {'✅ Available' if npz_data else '❌ Missing'}")
    print(f"   📋 Metadata: {'✅ Available' if metadata else '❌ Missing'}")
    
    if embeddings_data:
        train_samples = len(embeddings_data['train']['fmri_embeddings'])
        test_samples = len(embeddings_data['test']['fmri_embeddings'])
        print(f"\n🎯 READY FOR USE:")
        print(f"   📊 Training samples: {train_samples}")
        print(f"   📊 Test samples: {test_samples}")
        print(f"   🔗 Embedding dimension: 512 (CLIP space)")
        print(f"   🧠 fMRI dimension: 967")
        print(f"   🖼️  Image format: 224x224x3 (CLIP preprocessed)")
    
    return {
        'original_data': original_data,
        'model_data': model_data,
        'embeddings_data': embeddings_data,
        'npz_data': npz_data,
        'metadata': metadata
    }

if __name__ == "__main__":
    results = main()
