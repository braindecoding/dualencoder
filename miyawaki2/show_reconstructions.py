#!/usr/bin/env python3
"""
Show Reconstruction Results from Modular Training
Display visual comparison of original vs reconstructed images
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_show_results():
    """Load results and show reconstructions"""
    print("🎨 Loading Reconstruction Results")
    print("=" * 50)
    
    # Load results
    results_path = 'modular_training_results.pth'
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("💡 Please run modular_training.py first")
        return
    
    try:
        results = torch.load(results_path, weights_only=False)
        predictions = results['results']['predictions']
        
        targets = predictions['targets']
        diffusion_preds = predictions['diffusion']
        gan_preds = predictions['gan']
        
        print(f"✅ Loaded results:")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Diffusion predictions: {diffusion_preds.shape}")
        print(f"  GAN predictions: {gan_preds.shape}")
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Show reconstructions
    show_reconstruction_grid(targets, diffusion_preds, gan_preds)
    show_detailed_comparison(targets, diffusion_preds, gan_preds)
    show_metrics_summary(results)

def show_reconstruction_grid(targets, diffusion_preds, gan_preds, num_samples=12):
    """Show grid of reconstructions"""
    print(f"\n🖼️ Creating Reconstruction Grid ({num_samples} samples)")
    
    # Use all available samples (max 12)
    num_samples = min(num_samples, targets.shape[0])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
    fig.suptitle('Miyawaki2 Modular Training - Reconstruction Results', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(targets[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Diffusion
        axes[1, i].imshow(diffusion_preds[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Diffusion', fontsize=10)
        axes[1, i].axis('off')
        
        # GAN
        axes[2, i].imshow(gan_preds[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'GAN', fontsize=10)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', rotation=90, fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Diffusion', rotation=90, fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('GAN', rotation=90, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('miyawaki2_reconstruction_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Grid saved as 'miyawaki2_reconstruction_grid.png'")

def show_detailed_comparison(targets, diffusion_preds, gan_preds, num_samples=4):
    """Show detailed side-by-side comparison"""
    print(f"\n🔍 Creating Detailed Comparison ({num_samples} samples)")
    
    num_samples = min(num_samples, targets.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    fig.suptitle('Detailed Reconstruction Comparison', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(targets[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        # Diffusion
        axes[i, 1].imshow(diffusion_preds[i, 0], cmap='gray', vmin=0, vmax=1)
        diff_mse = torch.mean((diffusion_preds[i] - targets[i])**2).item()
        axes[i, 1].set_title(f'Diffusion\nMSE: {diff_mse:.4f}')
        axes[i, 1].axis('off')
        
        # GAN
        axes[i, 2].imshow(gan_preds[i, 0], cmap='gray', vmin=0, vmax=1)
        gan_mse = torch.mean((gan_preds[i] - targets[i])**2).item()
        axes[i, 2].set_title(f'GAN\nMSE: {gan_mse:.4f}')
        axes[i, 2].axis('off')
        
        # Difference map (GAN vs Original)
        diff_map = torch.abs(gan_preds[i, 0] - targets[i, 0])
        im = axes[i, 3].imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 3].set_title(f'GAN Error Map\nMax: {diff_map.max():.3f}')
        axes[i, 3].axis('off')
        
        # Add colorbar for difference map
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('miyawaki2_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Detailed comparison saved as 'miyawaki2_detailed_comparison.png'")

def show_metrics_summary(results):
    """Show metrics summary with bar chart"""
    print(f"\n📊 Creating Metrics Summary")
    
    diff_metrics = results['results']['diffusion_metrics']
    gan_metrics = results['results']['gan_metrics']
    
    # Prepare data for plotting
    metrics = ['MSE', 'SSIM', 'PSNR', 'LPIPS', 'Pixel Corr', 'Cosine Sim']
    diffusion_values = [
        diff_metrics['mse'],
        diff_metrics['ssim'],
        diff_metrics['psnr'],
        diff_metrics['lpips'],
        diff_metrics['pixel_correlation'],
        diff_metrics['cosine_similarity']
    ]
    gan_values = [
        gan_metrics['mse'],
        gan_metrics['ssim'],
        gan_metrics['psnr'],
        gan_metrics['lpips'],
        gan_metrics['pixel_correlation'],
        gan_metrics['cosine_similarity']
    ]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, diffusion_values, width, label='Diffusion', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, gan_values, width, label='GAN', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Miyawaki2 Modular: Diffusion vs GAN Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Training progress
    training_history = results['training_history']
    
    if 'Phase 1: Encoder Training' in training_history:
        ax2.plot(training_history['Phase 1: Encoder Training'], label='Phase 1: Encoders', linewidth=2)
    if 'Phase 2: Decoder Training' in training_history:
        ax2.plot(training_history['Phase 2: Decoder Training'], label='Phase 2: Decoders', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('miyawaki2_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Metrics summary saved as 'miyawaki2_metrics_summary.png'")

def print_numerical_summary(results):
    """Print numerical summary"""
    print(f"\n📋 NUMERICAL SUMMARY")
    print("=" * 50)
    
    diff_metrics = results['results']['diffusion_metrics']
    gan_metrics = results['results']['gan_metrics']
    retrieval_acc = results['results']['retrieval_accuracy']
    
    print(f"🎯 Cross-Modal Retrieval: {retrieval_acc:.1f}%")
    print(f"\n🔍 Reconstruction Quality:")
    
    metrics_info = [
        ('MSE (Lower Better)', 'mse'),
        ('SSIM (Higher Better)', 'ssim'),
        ('PSNR (Higher Better)', 'psnr'),
        ('LPIPS (Lower Better)', 'lpips'),
        ('Pixel Correlation (Higher Better)', 'pixel_correlation'),
        ('Cosine Similarity (Higher Better)', 'cosine_similarity')
    ]
    
    for name, key in metrics_info:
        diff_val = diff_metrics[key]
        gan_val = gan_metrics[key]
        
        # Determine winner based on metric type
        if 'Lower Better' in name:
            winner = 'Diffusion' if diff_val < gan_val else 'GAN'
            improvement = abs(diff_val - gan_val) / max(diff_val, gan_val) * 100
        else:
            winner = 'Diffusion' if diff_val > gan_val else 'GAN'
            improvement = abs(diff_val - gan_val) / max(diff_val, gan_val) * 100
        
        print(f"  {name}:")
        print(f"    Diffusion: {diff_val:.4f}")
        print(f"    GAN:       {gan_val:.4f}")
        print(f"    Winner:    {winner} ({improvement:.1f}% better)")
        print()

def main():
    """Main function"""
    print("🎨 Miyawaki2 Reconstruction Viewer")
    print("=" * 60)
    
    # Load and show results
    load_and_show_results()
    
    # Load results for numerical summary
    results_path = 'modular_training_results.pth'
    if Path(results_path).exists():
        results = torch.load(results_path, weights_only=False)
        print_numerical_summary(results)
    
    print("\n✅ Reconstruction visualization completed!")
    print("📁 Generated files:")
    print("  - miyawaki2_reconstruction_grid.png")
    print("  - miyawaki2_detailed_comparison.png") 
    print("  - miyawaki2_metrics_summary.png")

if __name__ == "__main__":
    main()
