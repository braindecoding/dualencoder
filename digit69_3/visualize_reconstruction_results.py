#!/usr/bin/env python3
"""
Visualize Reconstruction Results from All Models
Compare Enhanced LDM, Baseline, and Improved CLIP v2.0 results
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def load_evaluation_results():
    """Load evaluation results from pickle file"""
    print("📂 Loading evaluation results...")
    
    with open('improved_clip_v2_evaluation_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print(f"✅ Loaded results for {len(results)} models")
    return results

def create_comprehensive_visualization():
    """Create comprehensive visualization of reconstruction results"""
    print("🎨 Creating comprehensive reconstruction visualization...")
    
    # Load results
    results = load_evaluation_results()
    
    # Extract data
    models = list(results.keys())
    n_models = len(models)
    n_samples = 10  # We have 10 test samples (digits 0-9)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle('Brain-to-Image Reconstruction: Model Comparison\n'
                'fMRI → Digit Images (Test Set: Digits 0-9)', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Get target images (same for all models)
    target_images = results[models[0]]['target_images'].numpy()  # (10, 1, 28, 28)
    
    # Create grid: rows = samples, columns = target + models
    n_cols = 1 + n_models  # target + all models
    n_rows = n_samples
    
    # Plot each sample
    for sample_idx in range(n_samples):
        # Plot target image
        plt.subplot(n_rows, n_cols, sample_idx * n_cols + 1)
        target_img = target_images[sample_idx, 0]  # Remove channel dimension
        target_img = (target_img + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        
        plt.imshow(target_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Target\nDigit {sample_idx}', fontweight='bold', fontsize=10)
        plt.axis('off')
        
        # Add sample label on the left
        if sample_idx == 0:
            plt.ylabel('Digit 0', fontsize=12, fontweight='bold', rotation=0, labelpad=40)
        elif sample_idx < n_samples:
            plt.ylabel(f'Digit {sample_idx}', fontsize=12, fontweight='bold', rotation=0, labelpad=40)
        
        # Plot reconstructions from each model
        for model_idx, model_name in enumerate(models):
            plt.subplot(n_rows, n_cols, sample_idx * n_cols + 2 + model_idx)
            
            generated_images = results[model_name]['generated_images'].numpy()
            generated_img = generated_images[sample_idx, 0]  # Remove channel dimension
            generated_img = (generated_img + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            
            plt.imshow(generated_img, cmap='gray', vmin=0, vmax=1)
            
            # Add model name as title for first row
            if sample_idx == 0:
                # Get metrics for this model
                mse = results[model_name]['mse']
                ssim_mean = results[model_name]['ssim_mean']
                clip_score = results[model_name]['clip_score_mean']
                
                title = f'{model_name}\n'
                title += f'MSE: {mse:.3f}\n'
                title += f'SSIM: {ssim_mean:.3f}\n'
                title += f'CLIP: {clip_score:.3f}'
                
                plt.title(title, fontweight='bold', fontsize=9)
            
            plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.1, wspace=0.05)
    plt.savefig('reconstruction_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis():
    """Create detailed analysis with metrics and sample comparisons"""
    print("📊 Creating detailed analysis...")
    
    results = load_evaluation_results()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Reconstruction Analysis', fontsize=16, fontweight='bold')
    
    # Extract metrics
    models = list(results.keys())
    mse_scores = [results[model]['mse'] for model in models]
    ssim_scores = [results[model]['ssim_mean'] for model in models]
    corr_scores = [results[model]['correlation_mean'] for model in models]
    clip_scores = [results[model]['clip_score_mean'] for model in models]
    param_counts = [results[model]['total_params'] / 1e6 for model in models]
    
    # Colors for each model
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, mse_scores, color=colors)
    ax1.set_title('Mean Squared Error\n(Lower = Better)', fontweight='bold')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. SSIM Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, ssim_scores, color=colors)
    ax2.set_title('Structural Similarity\n(Higher = Better)', fontweight='bold')
    ax2.set_ylabel('SSIM')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars2, ssim_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. CLIP Score Comparison
    ax3 = axes[0, 2]
    bars3 = ax3.bar(models, clip_scores, color=colors)
    ax3.set_title('CLIP Semantic Score\n(Higher = Better)', fontweight='bold')
    ax3.set_ylabel('CLIP Score')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars3, clip_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Sample Comparison (Digit 3)
    ax4 = axes[1, 0]
    sample_idx = 3  # Digit 3
    target_images = results[models[0]]['target_images'].numpy()
    target_img = target_images[sample_idx, 0]
    target_img = (target_img + 1.0) / 2.0
    
    # Create comparison for digit 3
    comparison_imgs = [target_img]
    comparison_labels = ['Target']
    
    for model in models:
        generated_images = results[model]['generated_images'].numpy()
        generated_img = generated_images[sample_idx, 0]
        generated_img = (generated_img + 1.0) / 2.0
        comparison_imgs.append(generated_img)
        comparison_labels.append(model.replace(' ', '\n'))
    
    # Concatenate images horizontally
    combined_img = np.concatenate(comparison_imgs, axis=1)
    ax4.imshow(combined_img, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Sample Comparison: Digit 3', fontweight='bold')
    ax4.axis('off')
    
    # Add labels below each image
    img_width = 28
    for i, label in enumerate(comparison_labels):
        x_pos = i * img_width + img_width // 2
        ax4.text(x_pos, 32, label, ha='center', va='top', fontsize=8, fontweight='bold')
    
    # 5. Parameter Efficiency
    ax5 = axes[1, 1]
    bars5 = ax5.bar(models, param_counts, color=colors)
    ax5.set_title('Model Size\n(Parameters)', fontweight='bold')
    ax5.set_ylabel('Parameters (Millions)')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars5, param_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 6. Quality vs Efficiency Scatter
    ax6 = axes[1, 2]
    
    # Use inverse MSE as quality metric (higher = better)
    quality_scores = [1/mse for mse in mse_scores]
    
    scatter = ax6.scatter(param_counts, quality_scores, c=colors, s=200, alpha=0.7)
    
    for i, model in enumerate(models):
        ax6.annotate(model.replace(' ', '\n'), 
                    (param_counts[i], quality_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    ax6.set_xlabel('Parameters (Millions)')
    ax6.set_ylabel('Quality (1/MSE)')
    ax6.set_title('Quality vs Model Size\n(Top-left = Best)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_reconstruction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_sample_analysis():
    """Create detailed analysis for individual samples"""
    print("🔍 Creating individual sample analysis...")
    
    results = load_evaluation_results()
    models = list(results.keys())
    
    # Focus on a few interesting samples
    interesting_samples = [0, 3, 7, 9]  # Digits that might show clear differences
    
    fig, axes = plt.subplots(len(interesting_samples), len(models) + 1, 
                            figsize=(16, 12))
    fig.suptitle('Individual Sample Analysis: Selected Digits', 
                fontsize=16, fontweight='bold')
    
    target_images = results[models[0]]['target_images'].numpy()
    
    for row, sample_idx in enumerate(interesting_samples):
        # Plot target
        ax = axes[row, 0]
        target_img = target_images[sample_idx, 0]
        target_img = (target_img + 1.0) / 2.0
        
        ax.imshow(target_img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Target\nDigit {sample_idx}', fontweight='bold')
        ax.axis('off')
        
        # Plot reconstructions
        for col, model in enumerate(models):
            ax = axes[row, col + 1]
            
            generated_images = results[model]['generated_images'].numpy()
            generated_img = generated_images[sample_idx, 0]
            generated_img = (generated_img + 1.0) / 2.0
            
            ax.imshow(generated_img, cmap='gray', vmin=0, vmax=1)
            
            # Compute individual metrics for this sample
            target_flat = target_images[sample_idx, 0].flatten()
            generated_flat = generated_images[sample_idx, 0].flatten()
            
            # MSE for this sample
            sample_mse = mean_squared_error(target_flat, generated_flat)
            
            # SSIM for this sample
            target_ssim = (target_images[sample_idx, 0] + 1.0) / 2.0
            generated_ssim = (generated_images[sample_idx, 0] + 1.0) / 2.0
            sample_ssim = ssim(target_ssim, generated_ssim, data_range=1.0)
            
            # Correlation for this sample
            sample_corr = np.corrcoef(target_flat, generated_flat)[0, 1]
            if np.isnan(sample_corr):
                sample_corr = 0.0
            
            title = f'{model}\n'
            title += f'MSE: {sample_mse:.3f}\n'
            title += f'SSIM: {sample_ssim:.3f}\n'
            title += f'Corr: {sample_corr:.3f}'
            
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('individual_sample_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_metrics():
    """Print detailed metrics comparison"""
    print("\n📊 DETAILED METRICS COMPARISON")
    print("=" * 80)
    
    results = load_evaluation_results()
    
    # Print header
    print(f"{'Model':<25} {'MSE':<8} {'SSIM':<12} {'Correlation':<12} {'CLIP Score':<12} {'Params':<10}")
    print("-" * 80)
    
    # Sort by MSE (best first)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mse'])
    
    for model_name, result in sorted_models:
        print(f"{model_name:<25} "
              f"{result['mse']:<8.4f} "
              f"{result['ssim_mean']:<12.4f} "
              f"{result['correlation_mean']:<12.4f} "
              f"{result['clip_score_mean']:<12.4f} "
              f"{result['total_params']/1e6:<10.1f}M")
    
    print("\n🏆 RANKING ANALYSIS:")
    print("=" * 50)
    
    # Best in each category
    best_mse = min(results.items(), key=lambda x: x[1]['mse'])
    best_ssim = max(results.items(), key=lambda x: x[1]['ssim_mean'])
    best_corr = max(results.items(), key=lambda x: x[1]['correlation_mean'])
    best_clip = max(results.items(), key=lambda x: x[1]['clip_score_mean'])
    most_efficient = min(results.items(), key=lambda x: x[1]['total_params'])
    
    print(f"🥇 Best MSE (Pixel Accuracy): {best_mse[0]} ({best_mse[1]['mse']:.4f})")
    print(f"🥇 Best SSIM (Structure): {best_ssim[0]} ({best_ssim[1]['ssim_mean']:.4f})")
    print(f"🥇 Best Correlation: {best_corr[0]} ({best_corr[1]['correlation_mean']:.4f})")
    print(f"🥇 Best CLIP Score (Semantics): {best_clip[0]} ({best_clip[1]['clip_score_mean']:.4f})")
    print(f"⚡ Most Efficient: {most_efficient[0]} ({most_efficient[1]['total_params']/1e6:.1f}M params)")
    
    # CLIP guidance effectiveness
    if 'Improved CLIP v2.0 (Pure)' in results and 'Improved CLIP v2.0 (CLIP)' in results:
        pure_result = results['Improved CLIP v2.0 (Pure)']
        clip_result = results['Improved CLIP v2.0 (CLIP)']
        
        print(f"\n🎯 CLIP GUIDANCE EFFECTIVENESS:")
        print("=" * 50)
        
        mse_improvement = ((pure_result['mse'] - clip_result['mse']) / pure_result['mse']) * 100
        ssim_improvement = ((clip_result['ssim_mean'] - pure_result['ssim_mean']) / pure_result['ssim_mean']) * 100
        corr_improvement = ((clip_result['correlation_mean'] - pure_result['correlation_mean']) / pure_result['correlation_mean']) * 100
        clip_improvement = ((clip_result['clip_score_mean'] - pure_result['clip_score_mean']) / pure_result['clip_score_mean']) * 100
        
        print(f"MSE Improvement: {mse_improvement:+.1f}% ({'better' if mse_improvement > 0 else 'worse'})")
        print(f"SSIM Improvement: {ssim_improvement:+.1f}% ({'better' if ssim_improvement > 0 else 'worse'})")
        print(f"Correlation Improvement: {corr_improvement:+.1f}% ({'better' if corr_improvement > 0 else 'worse'})")
        print(f"CLIP Score Improvement: {clip_improvement:+.1f}% ({'better' if clip_improvement > 0 else 'worse'})")

def main():
    """Main visualization function"""
    print("🎨 STARTING RECONSTRUCTION VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create comprehensive grid visualization
        create_comprehensive_visualization()
        
        # Create detailed analysis
        create_detailed_analysis()
        
        # Create individual sample analysis
        create_individual_sample_analysis()
        
        # Print detailed metrics
        print_detailed_metrics()
        
        print("\n🎉 VISUALIZATION COMPLETED!")
        print("📁 Generated files:")
        print("   - reconstruction_comparison_grid.png")
        print("   - detailed_reconstruction_analysis.png") 
        print("   - individual_sample_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
