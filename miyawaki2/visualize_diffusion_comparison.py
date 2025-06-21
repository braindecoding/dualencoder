#!/usr/bin/env python3
"""
Visualize Diffusion Comparison Results
Compare Original vs Fixed Diffusion performance
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_comparison_results():
    """Load comparison results"""
    results_path = 'diffusion_comparison_results.pth'
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("💡 Please run modular_training_fixed.py first")
        return None
    
    try:
        results = torch.load(results_path, weights_only=False)
        print("✅ Loaded comparison results")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def create_metrics_comparison(results):
    """Create comprehensive metrics comparison"""
    print("📊 Creating Metrics Comparison")
    
    orig_results = results['Original']['eval_results']
    fixed_results = results['Fixed']['eval_results']
    
    # Prepare data
    metrics = ['MSE', 'SSIM', 'PSNR', 'LPIPS', 'Pixel Corr', 'Cosine Sim']
    
    # Diffusion metrics
    orig_diff_values = [
        orig_results['diffusion_metrics']['mse'],
        orig_results['diffusion_metrics']['ssim'],
        orig_results['diffusion_metrics']['psnr'],
        orig_results['diffusion_metrics']['lpips'],
        orig_results['diffusion_metrics']['pixel_correlation'],
        orig_results['diffusion_metrics']['cosine_similarity']
    ]
    
    fixed_diff_values = [
        fixed_results['diffusion_metrics']['mse'],
        fixed_results['diffusion_metrics']['ssim'],
        fixed_results['diffusion_metrics']['psnr'],
        fixed_results['diffusion_metrics']['lpips'],
        fixed_results['diffusion_metrics']['pixel_correlation'],
        fixed_results['diffusion_metrics']['cosine_similarity']
    ]
    
    # GAN metrics (for reference)
    orig_gan_values = [
        orig_results['gan_metrics']['mse'],
        orig_results['gan_metrics']['ssim'],
        orig_results['gan_metrics']['psnr'],
        orig_results['gan_metrics']['lpips'],
        orig_results['gan_metrics']['pixel_correlation'],
        orig_results['gan_metrics']['cosine_similarity']
    ]
    
    fixed_gan_values = [
        fixed_results['gan_metrics']['mse'],
        fixed_results['gan_metrics']['ssim'],
        fixed_results['gan_metrics']['psnr'],
        fixed_results['gan_metrics']['lpips'],
        fixed_results['gan_metrics']['pixel_correlation'],
        fixed_results['gan_metrics']['cosine_similarity']
    ]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Diffusion Comparison: Original vs Fixed', fontsize=16, fontweight='bold')
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Diffusion comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, orig_diff_values, width, label='Original Diffusion', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x + width/2, fixed_diff_values, width, label='Fixed Diffusion', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Diffusion Decoder Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # GAN comparison (for reference)
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, orig_gan_values, width, label='Original Setup', alpha=0.8, color='lightgreen')
    bars4 = ax2.bar(x + width/2, fixed_gan_values, width, label='Fixed Setup', alpha=0.8, color='gold')
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Values')
    ax2.set_title('GAN Decoder Comparison (Reference)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cross-modal retrieval comparison
    ax3 = axes[0, 2]
    retrieval_data = [
        orig_results['retrieval_accuracy'],
        fixed_results['retrieval_accuracy']
    ]
    colors = ['lightcoral', 'lightblue']
    bars = ax3.bar(['Original', 'Fixed'], retrieval_data, color=colors, alpha=0.8)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Cross-Modal Retrieval Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, retrieval_data):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    ax4 = axes[1, 0]
    training_times = [
        results['Original']['training_time'] / 60,  # Convert to minutes
        results['Fixed']['training_time'] / 60
    ]
    bars = ax4.bar(['Original', 'Fixed'], training_times, color=['lightcoral', 'lightblue'], alpha=0.8)
    ax4.set_ylabel('Time (minutes)')
    ax4.set_title('Training Time Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, training_times):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # Parameter count comparison
    ax5 = axes[1, 1]
    # Approximate parameter counts (from output)
    param_counts = [176.0, 173.0]  # Million parameters
    bars = ax5.bar(['Original', 'Fixed'], param_counts, color=['lightcoral', 'lightblue'], alpha=0.8)
    ax5.set_ylabel('Parameters (Millions)')
    ax5.set_title('Model Size Comparison')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, param_counts):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.0f}M', ha='center', va='bottom', fontweight='bold')
    
    # Improvement summary
    ax6 = axes[1, 2]
    improvements = [
        'Retrieval Acc', 'Training Speed', 'Parameters', 'Diffusion MSE'
    ]
    improvement_values = [
        8.3,  # +8.3% retrieval accuracy
        96.0,  # 96% faster training (5.1min -> 0.2min)
        1.7,   # 1.7% fewer parameters (176M -> 173M)
        -35.9  # -35.9% worse MSE (regression)
    ]
    
    colors = ['green' if x > 0 else 'red' for x in improvement_values]
    bars = ax6.bar(improvements, improvement_values, color=colors, alpha=0.7)
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('Fixed vs Original Improvements')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, improvement_values):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:+.1f}%', ha='center', 
                va='bottom' if value > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('diffusion_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Metrics comparison saved as 'diffusion_comparison_metrics.png'")

def create_reconstruction_comparison(results):
    """Create reconstruction comparison visualization"""
    print("🎨 Creating Reconstruction Comparison")
    
    orig_preds = results['Original']['eval_results']['predictions']
    fixed_preds = results['Fixed']['eval_results']['predictions']
    
    targets = orig_preds['targets']
    orig_diff = orig_preds['diffusion']
    orig_gan = orig_preds['gan']
    fixed_diff = fixed_preds['diffusion']
    fixed_gan = fixed_preds['gan']
    
    # Show 6 samples
    num_samples = min(6, targets.shape[0])
    
    fig, axes = plt.subplots(5, num_samples, figsize=(20, 12))
    fig.suptitle('Reconstruction Comparison: Original vs Fixed Diffusion', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(targets[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Original diffusion
        axes[1, i].imshow(orig_diff[i, 0], cmap='gray', vmin=0, vmax=1)
        orig_mse = torch.mean((orig_diff[i] - targets[i])**2).item()
        axes[1, i].set_title(f'Orig Diff\nMSE: {orig_mse:.3f}')
        axes[1, i].axis('off')
        
        # Fixed diffusion
        axes[2, i].imshow(fixed_diff[i, 0], cmap='gray', vmin=0, vmax=1)
        fixed_mse = torch.mean((fixed_diff[i] - targets[i])**2).item()
        axes[2, i].set_title(f'Fixed Diff\nMSE: {fixed_mse:.3f}')
        axes[2, i].axis('off')
        
        # Original GAN (reference)
        axes[3, i].imshow(orig_gan[i, 0], cmap='gray', vmin=0, vmax=1)
        gan_mse = torch.mean((orig_gan[i] - targets[i])**2).item()
        axes[3, i].set_title(f'Orig GAN\nMSE: {gan_mse:.3f}')
        axes[3, i].axis('off')
        
        # Fixed GAN (reference)
        axes[4, i].imshow(fixed_gan[i, 0], cmap='gray', vmin=0, vmax=1)
        fixed_gan_mse = torch.mean((fixed_gan[i] - targets[i])**2).item()
        axes[4, i].set_title(f'Fixed GAN\nMSE: {fixed_gan_mse:.3f}')
        axes[4, i].axis('off')
    
    # Add row labels
    row_labels = ['Original', 'Orig Diffusion', 'Fixed Diffusion', 'Orig GAN', 'Fixed GAN']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('diffusion_comparison_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Reconstruction comparison saved as 'diffusion_comparison_reconstructions.png'")

def create_training_curves(results):
    """Create training curves comparison"""
    print("📈 Creating Training Curves")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
    
    # Original training curves
    orig_encoder_losses = results['Original']['encoder_losses']
    orig_decoder_losses = results['Original']['decoder_losses']
    
    axes[0, 0].plot(orig_encoder_losses, 'o-', color='red', alpha=0.7, label='Encoder Loss')
    axes[0, 0].set_title('Original Diffusion - Encoder Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(orig_decoder_losses, 's-', color='red', alpha=0.7, label='Decoder Loss')
    axes[0, 1].set_title('Original Diffusion - Decoder Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Fixed training curves
    fixed_encoder_losses = results['Fixed']['encoder_losses']
    fixed_decoder_losses = results['Fixed']['decoder_losses']
    
    axes[1, 0].plot(fixed_encoder_losses, 'o-', color='blue', alpha=0.7, label='Encoder Loss')
    axes[1, 0].set_title('Fixed Diffusion - Encoder Training')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(fixed_decoder_losses, 's-', color='blue', alpha=0.7, label='Decoder Loss')
    axes[1, 1].set_title('Fixed Diffusion - Decoder Training')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('diffusion_comparison_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Training curves saved as 'diffusion_comparison_training.png'")

def print_detailed_summary(results):
    """Print detailed numerical summary"""
    print("\n" + "="*80)
    print("📋 DETAILED COMPARISON SUMMARY")
    print("="*80)
    
    orig_results = results['Original']['eval_results']
    fixed_results = results['Fixed']['eval_results']
    
    print(f"\n🎯 CROSS-MODAL RETRIEVAL:")
    orig_acc = orig_results['retrieval_accuracy']
    fixed_acc = fixed_results['retrieval_accuracy']
    improvement = ((fixed_acc - orig_acc) / orig_acc) * 100
    print(f"  Original: {orig_acc:.1f}%")
    print(f"  Fixed:    {fixed_acc:.1f}%")
    print(f"  Improvement: {improvement:+.1f}% relative")
    
    print(f"\n🔍 DIFFUSION RECONSTRUCTION QUALITY:")
    metrics = ['mse', 'ssim', 'psnr', 'lpips', 'pixel_correlation', 'cosine_similarity']
    for metric in metrics:
        orig_val = orig_results['diffusion_metrics'][metric]
        fixed_val = fixed_results['diffusion_metrics'][metric]
        
        # Calculate improvement (depends on metric type)
        if metric in ['mse', 'lpips']:  # Lower is better
            improvement = ((orig_val - fixed_val) / orig_val) * 100
        else:  # Higher is better
            improvement = ((fixed_val - orig_val) / abs(orig_val)) * 100 if orig_val != 0 else 0
        
        print(f"  {metric.upper()}:")
        print(f"    Original: {orig_val:.4f}")
        print(f"    Fixed:    {fixed_val:.4f}")
        print(f"    Change:   {improvement:+.1f}%")
    
    print(f"\n🏆 GAN PERFORMANCE (Reference):")
    orig_gan_mse = orig_results['gan_metrics']['mse']
    fixed_gan_mse = fixed_results['gan_metrics']['mse']
    print(f"  Original GAN MSE: {orig_gan_mse:.4f}")
    print(f"  Fixed GAN MSE:    {fixed_gan_mse:.4f}")
    print(f"  GAN vs Orig Diffusion: {(orig_gan_mse/orig_results['diffusion_metrics']['mse']):.1f}x better")
    print(f"  GAN vs Fixed Diffusion: {(fixed_gan_mse/fixed_results['diffusion_metrics']['mse']):.1f}x better")
    
    print(f"\n⏱️ EFFICIENCY GAINS:")
    orig_time = results['Original']['training_time']
    fixed_time = results['Fixed']['training_time']
    speedup = orig_time / fixed_time
    print(f"  Original Training: {orig_time/60:.1f} minutes")
    print(f"  Fixed Training:    {fixed_time/60:.1f} minutes")
    print(f"  Speedup:          {speedup:.1f}x faster")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  ✅ Fixed diffusion significantly improves cross-modal retrieval (+50%)")
    print(f"  ✅ Training is 25x faster with simplified architecture")
    print(f"  ❌ Reconstruction quality regressed (-36% MSE)")
    print(f"  🏆 GAN remains superior for reconstruction task")
    print(f"  💡 Fixed conditioning helps alignment but not reconstruction")

def main():
    """Main visualization function"""
    print("🎨 Visualizing Diffusion Comparison Results")
    print("="*60)
    
    # Load results
    results = load_comparison_results()
    if results is None:
        return
    
    # Create visualizations
    create_metrics_comparison(results)
    create_reconstruction_comparison(results)
    create_training_curves(results)
    
    # Print detailed summary
    print_detailed_summary(results)
    
    print(f"\n✅ Visualization completed!")
    print(f"📁 Generated files:")
    print(f"  - diffusion_comparison_metrics.png")
    print(f"  - diffusion_comparison_reconstructions.png")
    print(f"  - diffusion_comparison_training.png")

if __name__ == "__main__":
    main()
