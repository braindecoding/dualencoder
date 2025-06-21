#!/usr/bin/env python3
"""
Create comprehensive evaluation visualization and analysis
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_comprehensive_evaluation_report():
    """Create comprehensive evaluation report with visualizations"""
    
    # Load results
    with open('comprehensive_improved_clip_evaluation.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print("🎯 COMPREHENSIVE EVALUATION ANALYSIS")
    print("=" * 60)
    
    # Extract metrics
    models = []
    mse_scores = []
    ssim_scores = []
    corr_scores = []
    clip_scores = []
    param_counts = []
    
    for model_name, result in results.items():
        models.append(model_name)
        mse_scores.append(result['mse'])
        ssim_scores.append(result['ssim_mean'])
        corr_scores.append(result['correlation_mean'])
        clip_scores.append(result['clip_score_mean'])
        param_counts.append(result['total_params'] / 1e6)  # Convert to millions
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Evaluation: Improved CLIP vs Baseline', fontsize=16, fontweight='bold')
    
    # 1. MSE Comparison (lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, mse_scores, color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Mean Squared Error (Lower = Better)', fontweight='bold')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. SSIM Comparison (higher is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, ssim_scores, color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Structural Similarity (Higher = Better)', fontweight='bold')
    ax2.set_ylabel('SSIM')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars2, ssim_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Correlation Comparison (higher is better)
    ax3 = axes[0, 2]
    bars3 = ax3.bar(models, corr_scores, color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('Pixel Correlation (Higher = Better)', fontweight='bold')
    ax3.set_ylabel('Correlation')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars3, corr_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. CLIP Score Comparison (higher is better)
    ax4 = axes[1, 0]
    bars4 = ax4.bar(models, clip_scores, color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('CLIP Semantic Score (Higher = Better)', fontweight='bold')
    ax4.set_ylabel('CLIP Score')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars4, clip_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Parameter Count Comparison
    ax5 = axes[1, 1]
    bars5 = ax5.bar(models, param_counts, color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax5.set_title('Model Size (Parameters)', fontweight='bold')
    ax5.set_ylabel('Parameters (Millions)')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars5, param_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 6. Overall Performance Radar Chart
    ax6 = axes[1, 2]
    
    # Normalize metrics for radar chart (0-1 scale)
    # For MSE: lower is better, so invert
    norm_mse = [(1 - (score - min(mse_scores)) / (max(mse_scores) - min(mse_scores))) for score in mse_scores]
    # For others: higher is better
    norm_ssim = [(score - min(ssim_scores)) / (max(ssim_scores) - min(ssim_scores)) if max(ssim_scores) > min(ssim_scores) else 0.5 for score in ssim_scores]
    norm_corr = [(score - min(corr_scores)) / (max(corr_scores) - min(corr_scores)) for score in corr_scores]
    norm_clip = [(score - min(clip_scores)) / (max(clip_scores) - min(clip_scores)) for score in clip_scores]
    
    # Create radar chart data
    categories = ['MSE\n(inverted)', 'SSIM', 'Correlation', 'CLIP Score']
    
    # Plot each model
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, model in enumerate(models):
        values = [norm_mse[i], norm_ssim[i], norm_corr[i], norm_clip[i]]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax6.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('Overall Performance\n(Normalized 0-1)', fontweight='bold')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('comprehensive_improved_clip_evaluation_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n📊 DETAILED PERFORMANCE ANALYSIS:")
    print("=" * 60)
    
    # Find best and worst performers
    best_mse_idx = np.argmin(mse_scores)
    best_ssim_idx = np.argmax(ssim_scores)
    best_corr_idx = np.argmax(corr_scores)
    best_clip_idx = np.argmax(clip_scores)
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"   MSE (Lower=Better): {models[best_mse_idx]} ({mse_scores[best_mse_idx]:.4f})")
    print(f"   SSIM (Higher=Better): {models[best_ssim_idx]} ({ssim_scores[best_ssim_idx]:.4f})")
    print(f"   Correlation (Higher=Better): {models[best_corr_idx]} ({corr_scores[best_corr_idx]:.4f})")
    print(f"   CLIP Score (Higher=Better): {models[best_clip_idx]} ({clip_scores[best_clip_idx]:.4f})")
    
    # Performance vs Baseline
    baseline_idx = models.index('Baseline')
    print(f"\n📈 PERFORMANCE vs BASELINE:")
    for i, model in enumerate(models):
        if model != 'Baseline':
            mse_change = ((mse_scores[i] - mse_scores[baseline_idx]) / mse_scores[baseline_idx]) * 100
            ssim_change = ((ssim_scores[i] - ssim_scores[baseline_idx]) / ssim_scores[baseline_idx]) * 100
            corr_change = ((corr_scores[i] - corr_scores[baseline_idx]) / corr_scores[baseline_idx]) * 100
            clip_change = ((clip_scores[i] - clip_scores[baseline_idx]) / clip_scores[baseline_idx]) * 100
            
            print(f"\n   {model}:")
            print(f"     MSE: {mse_change:+.1f}% ({'worse' if mse_change > 0 else 'better'})")
            print(f"     SSIM: {ssim_change:+.1f}% ({'better' if ssim_change > 0 else 'worse'})")
            print(f"     Correlation: {corr_change:+.1f}% ({'better' if corr_change > 0 else 'worse'})")
            print(f"     CLIP Score: {clip_change:+.1f}% ({'better' if clip_change > 0 else 'worse'})")
    
    # Model efficiency analysis
    print(f"\n⚡ MODEL EFFICIENCY ANALYSIS:")
    for i, model in enumerate(models):
        efficiency_score = (ssim_scores[i] + corr_scores[i] + clip_scores[i]) / (param_counts[i] / 10)  # Normalize by 10M params
        print(f"   {model}: Efficiency Score = {efficiency_score:.4f}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   1. Baseline model achieves best traditional metrics (SSIM, Correlation)")
    print(f"   2. Improved CLIP models show progressive improvement with higher weights")
    print(f"   3. All CLIP models significantly larger (157M vs 5.5M parameters)")
    print(f"   4. CLIP guidance did not achieve expected semantic improvements")
    print(f"   5. Training stability achieved but quality metrics concerning")
    
    return results

if __name__ == "__main__":
    results = create_comprehensive_evaluation_report()
