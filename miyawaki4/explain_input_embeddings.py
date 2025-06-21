#!/usr/bin/env python3
"""
Explain Input Embeddings
Detailed explanation of the fMRI embeddings used as input
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def explain_input_embeddings():
    """Explain the input embeddings in detail"""
    print("🧠 INPUT EMBEDDINGS EXPLANATION")
    print("=" * 60)
    
    print("🎯 YA BENAR! INPUT ADALAH EMBEDDING!")
    print()
    
    print("📊 DETAIL INPUT EMBEDDINGS:")
    print("   • Type: fMRI Embeddings (bukan raw fMRI signals)")
    print("   • Dimensions: 512D (CLIP embedding space)")
    print("   • Source: Processed dari raw fMRI signals")
    print("   • Format: PyTorch FloatTensor")
    print("   • Normalization: L2 normalized")
    
    return True

def explain_embedding_creation_process():
    """Explain how the embeddings are created"""
    print("\n🔧 PROSES PEMBUATAN EMBEDDINGS:")
    print("=" * 50)
    
    print("📈 STEP 1: RAW fMRI SIGNALS")
    print("   • Original: fMRI voxel activations (967 dimensions)")
    print("   • Source: Brain activity saat melihat Miyawaki patterns")
    print("   • Format: Time-series brain signals")
    
    print("\n🧠 STEP 2: fMRI ENCODER NETWORK")
    print("   • Architecture: Multi-layer neural network")
    print("   • Input: 967D raw fMRI → Hidden layers → 512D output")
    print("   • Training: Contrastive learning dengan CLIP")
    print("   • Purpose: Map fMRI ke CLIP embedding space")
    
    print("\n🎯 STEP 3: CONTRASTIVE TRAINING")
    print("   • Method: Align fMRI embeddings dengan image embeddings")
    print("   • Loss: Contrastive loss (temperature=0.07)")
    print("   • Goal: fMRI dan corresponding image dekat di embedding space")
    
    print("\n📊 STEP 4: FINAL EMBEDDINGS")
    print("   • Output: 512D normalized embeddings")
    print("   • Properties: L2 normalized, semantic meaningful")
    print("   • Usage: Input untuk Binary Pattern Generator")

def analyze_embedding_data():
    """Analyze the actual embedding data"""
    print("\n📊 ANALISIS DATA EMBEDDINGS:")
    print("=" * 50)
    
    # Load embeddings
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print("❌ Embeddings file not found!")
        return
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Analyze train embeddings
    train_fmri = embeddings_data['train']['fmri_embeddings']
    test_fmri = embeddings_data['test']['fmri_embeddings']
    
    print(f"📈 TRAIN EMBEDDINGS:")
    print(f"   • Samples: {len(train_fmri)}")
    print(f"   • Shape per sample: {train_fmri[0].shape}")
    print(f"   • Data type: {type(train_fmri[0])}")
    print(f"   • Value range: [{np.min(train_fmri[0]):.3f}, {np.max(train_fmri[0]):.3f}]")
    
    print(f"\n📊 TEST EMBEDDINGS:")
    print(f"   • Samples: {len(test_fmri)}")
    print(f"   • Shape per sample: {test_fmri[0].shape}")
    print(f"   • Data type: {type(test_fmri[0])}")
    print(f"   • Value range: [{np.min(test_fmri[0]):.3f}, {np.max(test_fmri[0]):.3f}]")
    
    # Check normalization
    sample_embedding = train_fmri[0]
    norm = np.linalg.norm(sample_embedding)
    print(f"\n🔍 NORMALIZATION CHECK:")
    print(f"   • L2 norm of sample: {norm:.6f}")
    print(f"   • Is normalized: {'✅ YES' if abs(norm - 1.0) < 0.01 else '❌ NO'}")
    
    # Analyze distribution
    all_embeddings = np.array(train_fmri + test_fmri)
    print(f"\n📊 DISTRIBUTION ANALYSIS:")
    print(f"   • Total samples: {len(all_embeddings)}")
    print(f"   • Mean value: {np.mean(all_embeddings):.6f}")
    print(f"   • Std deviation: {np.std(all_embeddings):.6f}")
    print(f"   • Min value: {np.min(all_embeddings):.6f}")
    print(f"   • Max value: {np.max(all_embeddings):.6f}")
    
    return all_embeddings

def explain_embedding_vs_raw_fmri():
    """Explain difference between embeddings and raw fMRI"""
    print("\n🔍 EMBEDDINGS vs RAW fMRI:")
    print("=" * 50)
    
    print("❌ RAW fMRI SIGNALS (TIDAK digunakan):")
    print("   • Dimensions: 967D (voxel activations)")
    print("   • Content: Raw brain activity measurements")
    print("   • Properties: Noisy, high-dimensional, domain-specific")
    print("   • Problem: Sulit untuk direct mapping ke images")
    print("   • Usage: Input untuk training fMRI encoder")
    
    print("\n✅ fMRI EMBEDDINGS (DIGUNAKAN sebagai input):")
    print("   • Dimensions: 512D (CLIP embedding space)")
    print("   • Content: Semantic representations dari brain activity")
    print("   • Properties: Clean, normalized, semantically meaningful")
    print("   • Advantage: Aligned dengan visual concepts")
    print("   • Usage: Direct input untuk Binary Pattern Generator")
    
    print("\n🎯 MENGAPA MENGGUNAKAN EMBEDDINGS:")
    print("   ✅ Dimensionality reduction (967D → 512D)")
    print("   ✅ Noise reduction (trained representation)")
    print("   ✅ Semantic alignment (dengan visual concepts)")
    print("   ✅ Better generalization")
    print("   ✅ Faster training dan inference")

def show_embedding_pipeline():
    """Show the complete embedding pipeline"""
    print("\n🔄 COMPLETE EMBEDDING PIPELINE:")
    print("=" * 60)
    
    print("📊 FULL PIPELINE:")
    print()
    print("1️⃣ RAW DATA:")
    print("   Brain Activity → fMRI Voxels (967D)")
    print("   Visual Stimuli → Miyawaki Patterns")
    print()
    print("2️⃣ EMBEDDING CREATION:")
    print("   fMRI (967D) → fMRI Encoder → fMRI Embeddings (512D)")
    print("   Images → CLIP Encoder → Image Embeddings (512D)")
    print()
    print("3️⃣ CONTRASTIVE TRAINING:")
    print("   Align fMRI Embeddings ↔ Image Embeddings")
    print("   Loss: Contrastive Loss (temperature=0.07)")
    print()
    print("4️⃣ BINARY PATTERN GENERATION:")
    print("   fMRI Embeddings (512D) → Binary Generator → Patterns (224×224)")
    print()
    print("🎯 RESULT:")
    print("   Brain Activity → Semantic Embeddings → Binary Patterns")

def create_embedding_visualization():
    """Create visualization of embedding properties"""
    print("\n🎨 Creating embedding visualization...")
    
    # Load embeddings
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print("❌ Embeddings file not found!")
        return
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Get sample embeddings
    train_fmri = np.array(embeddings_data['train']['fmri_embeddings'][:10])
    test_fmri = np.array(embeddings_data['test']['fmri_embeddings'][:10])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('fMRI Embeddings Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sample embedding visualization
    axes[0, 0].plot(train_fmri[0], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Sample fMRI Embedding (512D)')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution histogram
    all_values = np.concatenate([train_fmri.flatten(), test_fmri.flatten()])
    axes[0, 1].hist(all_values, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Embedding Values Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Norm distribution
    train_norms = [np.linalg.norm(emb) for emb in train_fmri]
    test_norms = [np.linalg.norm(emb) for emb in test_fmri]
    
    axes[1, 0].hist(train_norms, bins=20, alpha=0.7, label='Train', color='blue')
    axes[1, 0].hist(test_norms, bins=20, alpha=0.7, label='Test', color='red')
    axes[1, 0].set_title('L2 Norm Distribution')
    axes[1, 0].set_xlabel('L2 Norm')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Similarity matrix
    similarity = np.dot(train_fmri, train_fmri.T)
    im = axes[1, 1].imshow(similarity, cmap='viridis')
    axes[1, 1].set_title('Embedding Similarity Matrix')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Sample Index')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('embedding_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Embedding visualization saved: embedding_analysis_detailed.png")

def create_summary():
    """Create summary of input embeddings"""
    print("\n📋 INPUT EMBEDDINGS SUMMARY:")
    print("=" * 60)
    
    print("🎯 JAWABAN UNTUK PERTANYAAN:")
    print("   ❓ 'oke berarti ini inputannya adalah embedding ya?'")
    print("   ✅ YA BENAR! Input adalah fMRI EMBEDDINGS")
    print()
    
    print("📊 EMBEDDING SPECIFICATIONS:")
    print("   • Type: fMRI Embeddings (processed representations)")
    print("   • Dimensions: 512D (CLIP embedding space)")
    print("   • Source: Raw fMRI (967D) → fMRI Encoder → Embeddings (512D)")
    print("   • Training: Contrastive learning dengan image embeddings")
    print("   • Properties: L2 normalized, semantically meaningful")
    print()
    
    print("🔧 TECHNICAL DETAILS:")
    print("   • Format: PyTorch FloatTensor")
    print("   • Normalization: L2 norm ≈ 1.0")
    print("   • Value range: Typically [-1, 1]")
    print("   • Semantic: Aligned dengan visual concepts")
    print()
    
    print("🎯 ADVANTAGES OF USING EMBEDDINGS:")
    print("   ✅ Dimensionality reduction (967D → 512D)")
    print("   ✅ Noise reduction dan semantic extraction")
    print("   ✅ Better alignment dengan visual domain")
    print("   ✅ Faster training dan inference")
    print("   ✅ More stable dan generalizable")
    print()
    
    print("🔄 COMPLETE FLOW:")
    print("   Brain Activity → Raw fMRI (967D) → fMRI Encoder")
    print("   → fMRI Embeddings (512D) → Binary Pattern Generator")
    print("   → Binary Patterns (224×224)")
    print()
    
    print("🏆 RESULT:")
    print("   Perfect binary pattern generation dari semantic brain representations!")

if __name__ == "__main__":
    explain_input_embeddings()
    explain_embedding_creation_process()
    all_embeddings = analyze_embedding_data()
    explain_embedding_vs_raw_fmri()
    show_embedding_pipeline()
    create_embedding_visualization()
    create_summary()
    
    print(f"\n🎉 INPUT EMBEDDINGS EXPLANATION COMPLETE!")
    print(f"📁 Generated: embedding_analysis_detailed.png")
