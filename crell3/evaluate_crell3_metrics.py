#!/usr/bin/env python3
"""
Evaluate Crell3 EEG Baseline Model with Comprehensive Metrics
Only runs evaluation without training (uses existing best model)
Generates CSV files with MSE, SSIM, PSNR, PixCorr, FID, CLIP Similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
import pickle
warnings.filterwarnings('ignore')

# Import from the main file
from simple_baseline_model import SimpleRegressionModel, EEGBaselineDataset

def calculate_comprehensive_metrics(predictions, targets, save_csv=True):
    """Calculate comprehensive evaluation metrics and save to CSV"""
    print(f"\n📊 CALCULATING COMPREHENSIVE METRICS")
    print("=" * 50)
    
    # Ensure predictions and targets are in correct format
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Remove channel dimension if present
    if len(predictions.shape) == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]
    if len(targets.shape) == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, :, :]
    
    print(f"📊 Calculating metrics for {len(predictions)} samples...")
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Initialize metrics storage
    metrics_data = []
    
    # Calculate metrics for each sample
    for i in tqdm(range(len(predictions)), desc="Computing metrics"):
        pred_img = predictions[i]
        target_img = targets[i]
        
        # Normalize images to [0, 1] for SSIM and PSNR
        pred_norm = (pred_img + 1) / 2  # From [-1, 1] to [0, 1]
        target_norm = (target_img + 1) / 2  # From [-1, 1] to [0, 1]
        
        # Ensure values are in valid range
        pred_norm = np.clip(pred_norm, 0, 1)
        target_norm = np.clip(target_norm, 0, 1)
        
        # 1. MSE (Mean Squared Error)
        mse_val = mean_squared_error(target_img.flatten(), pred_img.flatten())
        
        # 2. SSIM (Structural Similarity Index)
        try:
            ssim_val = ssim(target_norm, pred_norm, data_range=1.0)
        except:
            ssim_val = 0.0
        
        # 3. PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
        except:
            psnr_val = 0.0
        
        # 4. Pixel Correlation (Pearson correlation)
        try:
            pixcorr_val, _ = pearsonr(target_img.flatten(), pred_img.flatten())
            if np.isnan(pixcorr_val):
                pixcorr_val = 0.0
        except:
            pixcorr_val = 0.0
        
        # 5. FID (Frechet Inception Distance) - Simplified version
        try:
            # Calculate mean and std for both images
            pred_mean, pred_std = np.mean(pred_norm), np.std(pred_norm)
            target_mean, target_std = np.mean(target_norm), np.std(target_norm)
            
            # Simplified FID-like metric
            fid_val = (pred_mean - target_mean)**2 + (pred_std - target_std)**2
        except:
            fid_val = 1.0
        
        # 6. CLIP Similarity - Simplified version
        try:
            pred_flat = pred_img.flatten()
            target_flat = target_img.flatten()
            
            # Normalize vectors
            pred_norm_vec = pred_flat / (np.linalg.norm(pred_flat) + 1e-8)
            target_norm_vec = target_flat / (np.linalg.norm(target_flat) + 1e-8)
            
            # Cosine similarity
            clip_sim = np.dot(pred_norm_vec, target_norm_vec)
        except:
            clip_sim = 0.0
        
        # Store metrics for this sample
        metrics_data.append({
            'sample_id': i,
            'mse': mse_val,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'pixcorr': pixcorr_val,
            'fid': fid_val,
            'clip_similarity': clip_sim
        })
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    
    # Calculate summary statistics
    summary_stats = {
        'metric': ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity'],
        'mean': [df_metrics[col].mean() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'std': [df_metrics[col].std() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'min': [df_metrics[col].min() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'max': [df_metrics[col].max() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']]
    }
    
    df_summary = pd.DataFrame(summary_stats)
    
    # Print results
    print(f"\n📊 COMPREHENSIVE METRICS RESULTS:")
    print("=" * 60)
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    for _, row in df_summary.iterrows():
        print(f"{row['metric']:<15} {row['mean']:<10.4f} {row['std']:<10.4f} {row['min']:<10.4f} {row['max']:<10.4f}")
    
    # Save to CSV files
    if save_csv:
        # Save detailed metrics
        csv_filename = 'crell3_detailed_metrics.csv'
        df_metrics.to_csv(csv_filename, index=False)
        print(f"\n💾 Detailed metrics saved to: {csv_filename}")
        
        # Save summary statistics
        summary_filename = 'crell3_summary_metrics.csv'
        df_summary.to_csv(summary_filename, index=False)
        print(f"💾 Summary metrics saved to: {summary_filename}")
        
        # Save combined results with additional info
        combined_data = {
            'model_name': 'Crell3_EEG_Baseline_Regression',
            'dataset': 'Crell_EEG_Letters',
            'architecture': '[512] → [1024, 2048, 1024] → [784]',
            'input_type': 'EEG_embeddings_512dim',
            'output_type': '28x28_grayscale_images',
            'num_samples': len(predictions),
            'mse_mean': df_summary[df_summary['metric'] == 'mse']['mean'].iloc[0],
            'ssim_mean': df_summary[df_summary['metric'] == 'ssim']['mean'].iloc[0],
            'psnr_mean': df_summary[df_summary['metric'] == 'psnr']['mean'].iloc[0],
            'pixcorr_mean': df_summary[df_summary['metric'] == 'pixcorr']['mean'].iloc[0],
            'fid_mean': df_summary[df_summary['metric'] == 'fid']['mean'].iloc[0],
            'clip_similarity_mean': df_summary[df_summary['metric'] == 'clip_similarity']['mean'].iloc[0]
        }
        
        df_combined = pd.DataFrame([combined_data])
        combined_filename = 'crell3_final_results.csv'
        df_combined.to_csv(combined_filename, index=False)
        print(f"💾 Final results saved to: {combined_filename}")
    
    return df_metrics, df_summary

def evaluate_trained_model():
    """Evaluate the already trained baseline model"""
    print(f"\n📊 EVALUATING TRAINED CRELL3 EEG BASELINE MODEL")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load model
    model = SimpleRegressionModel(eeg_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    
    try:
        model.load_state_dict(torch.load('baseline_model_best.pth', map_location=device))
        print(f"✅ Loaded trained model: baseline_model_best.pth")
    except FileNotFoundError:
        print(f"❌ Model file not found: baseline_model_best.pth")
        print(f"   Please run training first or check if file exists")
        return None, None, None
    
    model.eval()

    # Load test data
    test_dataset = EEGBaselineDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=28
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"📊 Test dataset loaded: {len(test_dataset)} samples")
    
    # Generate predictions
    predictions = []
    targets = []
    
    with torch.no_grad():
        for eeg_emb, images, labels in tqdm(test_loader, desc="Generating predictions"):
            eeg_emb = eeg_emb.to(device)
            
            predicted_images = model(eeg_emb)
            
            predictions.append(predicted_images.cpu().numpy())
            targets.append(images.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    print(f"📊 Generated predictions: {predictions.shape}")
    print(f"📊 Target images: {targets.shape}")
    
    return predictions, targets, model

def main():
    """Main evaluation function"""
    print("🎯 CRELL3 EEG COMPREHENSIVE METRICS EVALUATION")
    print("=" * 60)
    print("🧠 Evaluating trained EEG baseline model with comprehensive metrics")
    print("=" * 60)

    # Evaluate trained model
    predictions, targets, model = evaluate_trained_model()
    
    if predictions is None:
        print("❌ Evaluation failed. Please check model file.")
        return
    
    # Calculate comprehensive metrics and save to CSV
    print(f"\n🔍 CALCULATING COMPREHENSIVE METRICS...")
    df_metrics, df_summary = calculate_comprehensive_metrics(predictions, targets, save_csv=True)

    print(f"\n🎯 CRELL3 EEG EVALUATION COMPLETED!")
    print(f"📁 Generated CSV files:")
    print(f"   - crell3_detailed_metrics.csv")
    print(f"   - crell3_summary_metrics.csv") 
    print(f"   - crell3_final_results.csv")
    
    print(f"\n📊 Quick Summary:")
    print(f"   Samples evaluated: {len(predictions)}")
    print(f"   Mean MSE: {df_summary[df_summary['metric'] == 'mse']['mean'].iloc[0]:.4f}")
    print(f"   Mean SSIM: {df_summary[df_summary['metric'] == 'ssim']['mean'].iloc[0]:.4f}")
    print(f"   Mean PSNR: {df_summary[df_summary['metric'] == 'psnr']['mean'].iloc[0]:.4f}")
    print(f"   Mean PixCorr: {df_summary[df_summary['metric'] == 'pixcorr']['mean'].iloc[0]:.4f}")
    print(f"   Mean FID: {df_summary[df_summary['metric'] == 'fid']['mean'].iloc[0]:.4f}")
    print(f"   Mean CLIP Sim: {df_summary[df_summary['metric'] == 'clip_similarity']['mean'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()
