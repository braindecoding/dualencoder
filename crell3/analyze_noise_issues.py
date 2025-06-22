#!/usr/bin/env python3
"""
Analyze Noise Issues in EEG-to-Image Generation
Identify key factors causing noise in generated images
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

def analyze_data_quality():
    """Analyze data quality and preprocessing"""
    print("🔍 ANALYZING DATA QUALITY")
    print("=" * 50)
    
    # Load embeddings
    with open('crell_embeddings_20250622_173213.pkl', 'rb') as f:
        emb_data = pickle.load(f)
    
    # Load original data
    with open('crell_processed_data_correct.pkl', 'rb') as f:
        crell_data = pickle.load(f)
    
    embeddings = emb_data['embeddings']
    labels = emb_data['labels']
    images = crell_data['validation']['images']
    
    print(f"📊 Data Statistics:")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Labels: {len(labels)}")
    print(f"   Images: {len(images)}")
    
    # Analyze embedding quality
    print(f"\n📊 Embedding Quality:")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Check for NaN/Inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"   NaN values: {nan_count}")
    print(f"   Inf values: {inf_count}")
    
    # Analyze embedding variance per dimension
    dim_variance = embeddings.var(axis=0)
    print(f"   Dimension variance - Mean: {dim_variance.mean():.6f}")
    print(f"   Dimension variance - Std: {dim_variance.std():.6f}")
    print(f"   Low variance dims (<0.001): {(dim_variance < 0.001).sum()}/512")
    
    # Analyze image quality
    print(f"\n📊 Image Quality:")
    sample_img = np.array(images[0])
    print(f"   Image shape: {sample_img.shape}")
    print(f"   Image dtype: {sample_img.dtype}")
    print(f"   Image range: [{sample_img.min()}, {sample_img.max()}]")
    
    # Check image diversity
    img_arrays = [np.array(img).flatten() for img in images[:10]]
    img_matrix = np.array(img_arrays)
    correlations = np.corrcoef(img_matrix)
    mean_corr = correlations[np.triu_indices_from(correlations, k=1)].mean()
    print(f"   Inter-image correlation: {mean_corr:.4f}")
    
    return embeddings, images, labels

def analyze_model_architecture():
    """Analyze model architecture issues"""
    print(f"\n🏗️ ANALYZING MODEL ARCHITECTURE")
    print("=" * 50)
    
    # Key architecture factors
    factors = {
        "UNet Channels": "[32, 64, 128, 256] - May be too small",
        "Condition Dim": "512 - Good",
        "Image Size": "28x28 - Small resolution",
        "Timesteps": "100 - Standard",
        "Inference Steps": "20 - May be too few",
        "Model Size": "976K params - Relatively small"
    }
    
    for factor, analysis in factors.items():
        print(f"   {factor}: {analysis}")
    
    # Potential issues
    print(f"\n⚠️ Potential Architecture Issues:")
    print(f"   1. Small UNet channels may lack capacity")
    print(f"   2. 28x28 resolution limits detail generation")
    print(f"   3. 20 inference steps may be insufficient")
    print(f"   4. Model size may be too small for complex mapping")

def analyze_training_dynamics():
    """Analyze training dynamics"""
    print(f"\n📈 ANALYZING TRAINING DYNAMICS")
    print("=" * 50)
    
    # Training observations
    print(f"📊 Training Observations:")
    print(f"   Best test loss: 0.0913 (very low)")
    print(f"   Final train loss: 0.2187")
    print(f"   Final test loss: 0.1193")
    print(f"   Training time: 141.17s")
    
    print(f"\n⚠️ Potential Training Issues:")
    print(f"   1. Very low loss may indicate overfitting")
    print(f"   2. Gap between train/test loss suggests overfitting")
    print(f"   3. Fast convergence may miss optimal solution")
    print(f"   4. Small dataset (102 train, 26 test) prone to overfitting")

def analyze_loss_components():
    """Analyze loss component balance"""
    print(f"\n⚖️ ANALYZING LOSS COMPONENTS")
    print("=" * 50)
    
    print(f"📊 Hybrid Loss Weights:")
    print(f"   SSIM Loss: 0.4 (40%)")
    print(f"   Classification Loss: 0.4 (40%)")
    print(f"   CLIP Loss: 0.2 (20%)")
    print(f"   L1 Loss: 0.1 (10%)")
    
    print(f"\n⚠️ Potential Loss Issues:")
    print(f"   1. SSIM may dominate and cause blurring")
    print(f"   2. Classification loss may conflict with generation")
    print(f"   3. CLIP loss weight too low for semantic guidance")
    print(f"   4. Multiple losses may cause optimization conflicts")

def analyze_diffusion_process():
    """Analyze diffusion process issues"""
    print(f"\n🌊 ANALYZING DIFFUSION PROCESS")
    print("=" * 50)
    
    print(f"📊 Diffusion Parameters:")
    print(f"   Beta schedule: Linear (0.0001 to 0.02)")
    print(f"   Timesteps: 100")
    print(f"   Inference steps: 20")
    print(f"   Sampling method: DDIM")
    
    print(f"\n⚠️ Potential Diffusion Issues:")
    print(f"   1. Linear beta schedule may be suboptimal")
    print(f"   2. 20 inference steps may be too few for quality")
    print(f"   3. DDIM sampling may need more steps")
    print(f"   4. Noise schedule may not match data distribution")

def suggest_improvements():
    """Suggest key improvements"""
    print(f"\n🚀 KEY IMPROVEMENT SUGGESTIONS")
    print("=" * 60)
    
    improvements = {
        "1. Architecture Scaling": [
            "Increase UNet channels: [64, 128, 256, 512]",
            "Add more UNet layers for better capacity",
            "Increase model parameters (2M+ params)",
            "Use attention mechanisms in UNet"
        ],
        "2. Training Strategy": [
            "Increase training epochs (200-500)",
            "Use data augmentation to prevent overfitting",
            "Implement progressive training",
            "Add regularization (dropout, weight decay)"
        ],
        "3. Loss Function Optimization": [
            "Rebalance loss weights (reduce SSIM dominance)",
            "Use perceptual loss instead of SSIM",
            "Increase CLIP loss weight to 0.4-0.6",
            "Add adversarial loss for realism"
        ],
        "4. Diffusion Process": [
            "Increase inference steps to 50-100",
            "Use cosine beta schedule",
            "Implement classifier-free guidance",
            "Add noise conditioning"
        ],
        "5. Data Enhancement": [
            "Use larger image resolution (64x64 or 128x128)",
            "Implement data augmentation",
            "Add more diverse training samples",
            "Improve EEG preprocessing"
        ]
    }
    
    for category, suggestions in improvements.items():
        print(f"\n🎯 {category}:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")

def compare_model_performance():
    """Compare performance across models"""
    print(f"\n📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    models = {
        "Baseline": {
            "Training Time": "10.87s",
            "Best Loss": "0.1840",
            "Semantic Accuracy": "N/A",
            "Key Issue": "No semantic guidance"
        },
        "CLIP-Guided": {
            "Training Time": "14.38s", 
            "Best Loss": "0.1011",
            "Semantic Accuracy": "5%",
            "Key Issue": "Single loss component"
        },
        "Hybrid CLIP-SSIM": {
            "Training Time": "141.17s",
            "Best Loss": "0.0913",
            "Semantic Accuracy": "5%",
            "Key Issue": "Loss conflicts, overfitting"
        }
    }
    
    for model, stats in models.items():
        print(f"\n🔍 {model}:")
        for metric, value in stats.items():
            print(f"   {metric}: {value}")

def main():
    """Main analysis function"""
    print("🔍 NOISE ANALYSIS FOR EEG-TO-IMAGE GENERATION")
    print("=" * 70)
    print("Analyzing key factors causing noise in generated images")
    print("=" * 70)
    
    # Run analyses
    embeddings, images, labels = analyze_data_quality()
    analyze_model_architecture()
    analyze_training_dynamics()
    analyze_loss_components()
    analyze_diffusion_process()
    compare_model_performance()
    suggest_improvements()
    
    print(f"\n🎯 TOP PRIORITY FIXES:")
    print(f"=" * 40)
    print(f"1. 🏗️ SCALE UP ARCHITECTURE")
    print(f"   • Increase UNet channels: [64, 128, 256, 512]")
    print(f"   • Add attention mechanisms")
    print(f"   • Target 2M+ parameters")
    
    print(f"\n2. ⚖️ REBALANCE LOSS FUNCTIONS")
    print(f"   • Reduce SSIM weight: 0.2 (from 0.4)")
    print(f"   • Increase CLIP weight: 0.5 (from 0.2)")
    print(f"   • Add perceptual loss")
    
    print(f"\n3. 🌊 IMPROVE DIFFUSION PROCESS")
    print(f"   • Increase inference steps: 50-100")
    print(f"   • Use cosine beta schedule")
    print(f"   • Implement classifier-free guidance")
    
    print(f"\n4. 📊 ENHANCE TRAINING")
    print(f"   • Increase epochs: 200-500")
    print(f"   • Add data augmentation")
    print(f"   • Implement progressive training")
    
    print(f"\n5. 🎯 INCREASE RESOLUTION")
    print(f"   • Target 64x64 or 128x128 images")
    print(f"   • Better detail generation capacity")
    
    print(f"\n🚀 EXPECTED IMPROVEMENTS:")
    print(f"   • Better semantic accuracy (>20%)")
    print(f"   • Higher visual quality (less noise)")
    print(f"   • More recognizable letter shapes")
    print(f"   • Better EEG-image alignment")

if __name__ == "__main__":
    main()
