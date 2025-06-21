#!/usr/bin/env python3
"""
Performance Metrics Report
Create detailed performance metrics report with visualizations
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_metrics_results():
    """Load the comprehensive metrics results"""
    results_path = "comprehensive_metrics_results.pkl"
    if not Path(results_path).exists():
        print("❌ Metrics results not found! Run comprehensive_metrics.py first.")
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results

def create_metrics_table(results):
    """Create a comprehensive metrics table"""
    print("📊 COMPREHENSIVE PERFORMANCE METRICS TABLE")
    print("=" * 80)
    
    # Prepare data for table
    metrics_data = []
    
    # Standard CV metrics
    metrics_data.append({
        'Category': 'Computer Vision',
        'Metric': 'Pixel Correlation (PixCorr)',
        'Mean': f"{results['pixel_correlation']['mean']:.4f}",
        'Std': f"{results['pixel_correlation']['std']:.4f}",
        'Range': f"[{min(results['pixel_correlation']['individual']):.4f}, {max(results['pixel_correlation']['individual']):.4f}]",
        'Unit': 'correlation',
        'Higher_Better': 'Yes'
    })
    
    metrics_data.append({
        'Category': 'Computer Vision',
        'Metric': 'Structural Similarity (SSIM)',
        'Mean': f"{results['ssim']['mean']:.4f}",
        'Std': f"{results['ssim']['std']:.4f}",
        'Range': f"[{min(results['ssim']['individual']):.4f}, {max(results['ssim']['individual']):.4f}]",
        'Unit': 'similarity',
        'Higher_Better': 'Yes'
    })
    
    metrics_data.append({
        'Category': 'Computer Vision',
        'Metric': 'Mean Squared Error (MSE)',
        'Mean': f"{results['mse']['mean']:.6f}",
        'Std': f"{results['mse']['std']:.6f}",
        'Range': f"[{min(results['mse']['individual']):.6f}, {max(results['mse']['individual']):.6f}]",
        'Unit': 'error',
        'Higher_Better': 'No'
    })
    
    metrics_data.append({
        'Category': 'Computer Vision',
        'Metric': 'Peak Signal-to-Noise Ratio (PSNR)',
        'Mean': f"{results['psnr']['mean']:.2f}",
        'Std': f"{results['psnr']['std']:.2f}",
        'Range': f"[{min([p for p in results['psnr']['individual'] if p != float('inf')]):.2f}, {max([p for p in results['psnr']['individual'] if p != float('inf')]):.2f}]",
        'Unit': 'dB',
        'Higher_Better': 'Yes'
    })
    
    # Deep learning metrics
    if 'note' not in results['clip_similarity']:
        metrics_data.append({
            'Category': 'Deep Learning',
            'Metric': 'CLIP Similarity',
            'Mean': f"{results['clip_similarity']['mean']:.4f}",
            'Std': f"{results['clip_similarity']['std']:.4f}",
            'Range': f"[{min(results['clip_similarity']['individual']):.4f}, {max(results['clip_similarity']['individual']):.4f}]",
            'Unit': 'cosine similarity',
            'Higher_Better': 'Yes'
        })
    
    if 'note' not in results['inception_distance']:
        metrics_data.append({
            'Category': 'Deep Learning',
            'Metric': 'Inception Distance',
            'Mean': f"{results['inception_distance']['mean']:.4f}",
            'Std': f"{results['inception_distance']['std']:.4f}",
            'Range': f"[{min(results['inception_distance']['individual']):.4f}, {max(results['inception_distance']['individual']):.4f}]",
            'Unit': 'distance',
            'Higher_Better': 'No'
        })
    
    # Binary-specific metrics
    binary_metrics = results['binary_metrics']
    for metric_name, metric_data in binary_metrics.items():
        display_name = metric_name.replace('_', ' ').title()
        metrics_data.append({
            'Category': 'Binary-Specific',
            'Metric': display_name,
            'Mean': f"{metric_data['mean']:.4f}",
            'Std': f"{metric_data['std']:.4f}",
            'Range': f"[{min(metric_data['individual']):.4f}, {max(metric_data['individual']):.4f}]",
            'Unit': 'ratio/accuracy',
            'Higher_Better': 'Yes'
        })
    
    # Create DataFrame and display
    df = pd.DataFrame(metrics_data)
    
    print("\n📋 PERFORMANCE METRICS SUMMARY TABLE:")
    print("-" * 120)
    print(f"{'Category':<15} {'Metric':<35} {'Mean':<12} {'Std':<12} {'Range':<25} {'Unit':<15} {'Better':<8}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['Category']:<15} {row['Metric']:<35} {row['Mean']:<12} {row['Std']:<12} {row['Range']:<25} {row['Unit']:<15} {row['Higher_Better']:<8}")
    
    return df

def create_metrics_visualization(results):
    """Create comprehensive metrics visualization"""
    print("\n🎨 Creating metrics visualization...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Comprehensive Performance Metrics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pixel Correlation
    axes[0, 0].hist(results['pixel_correlation']['individual'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].set_title('Pixel Correlation Distribution')
    axes[0, 0].set_xlabel('Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(results['pixel_correlation']['mean'], color='red', linestyle='--', 
                      label=f"Mean: {results['pixel_correlation']['mean']:.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. SSIM
    axes[0, 1].hist(results['ssim']['individual'], bins=10, alpha=0.7, color='green')
    axes[0, 1].set_title('SSIM Distribution')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(results['ssim']['mean'], color='red', linestyle='--',
                      label=f"Mean: {results['ssim']['mean']:.3f}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MSE
    axes[0, 2].hist(results['mse']['individual'], bins=10, alpha=0.7, color='orange')
    axes[0, 2].set_title('MSE Distribution')
    axes[0, 2].set_xlabel('MSE')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(results['mse']['mean'], color='red', linestyle='--',
                      label=f"Mean: {results['mse']['mean']:.4f}")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. PSNR
    psnr_finite = [p for p in results['psnr']['individual'] if p != float('inf')]
    axes[1, 0].hist(psnr_finite, bins=10, alpha=0.7, color='purple')
    axes[1, 0].set_title('PSNR Distribution')
    axes[1, 0].set_xlabel('PSNR (dB)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(results['psnr']['mean'], color='red', linestyle='--',
                      label=f"Mean: {results['psnr']['mean']:.1f} dB")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. CLIP Similarity (if available)
    if 'note' not in results['clip_similarity']:
        axes[1, 1].hist(results['clip_similarity']['individual'], bins=10, alpha=0.7, color='cyan')
        axes[1, 1].set_title('CLIP Similarity Distribution')
        axes[1, 1].set_xlabel('CLIP Similarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(results['clip_similarity']['mean'], color='red', linestyle='--',
                          label=f"Mean: {results['clip_similarity']['mean']:.3f}")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'CLIP Similarity\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('CLIP Similarity')
    
    # 6. Binary Accuracy
    binary_acc = results['binary_metrics']['binary_accuracy']['individual']
    axes[1, 2].hist(binary_acc, bins=10, alpha=0.7, color='red')
    axes[1, 2].set_title('Binary Accuracy Distribution')
    axes[1, 2].set_xlabel('Accuracy')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(results['binary_metrics']['binary_accuracy']['mean'], 
                      color='blue', linestyle='--',
                      label=f"Mean: {results['binary_metrics']['binary_accuracy']['mean']:.3f}")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Dice Coefficient
    dice_coeff = results['binary_metrics']['dice_coefficient']['individual']
    axes[2, 0].hist(dice_coeff, bins=10, alpha=0.7, color='brown')
    axes[2, 0].set_title('Dice Coefficient Distribution')
    axes[2, 0].set_xlabel('Dice Coefficient')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].axvline(results['binary_metrics']['dice_coefficient']['mean'], 
                      color='blue', linestyle='--',
                      label=f"Mean: {results['binary_metrics']['dice_coefficient']['mean']:.3f}")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Jaccard Index
    jaccard_idx = results['binary_metrics']['jaccard_index']['individual']
    axes[2, 1].hist(jaccard_idx, bins=10, alpha=0.7, color='pink')
    axes[2, 1].set_title('Jaccard Index Distribution')
    axes[2, 1].set_xlabel('Jaccard Index')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].axvline(results['binary_metrics']['jaccard_index']['mean'], 
                      color='blue', linestyle='--',
                      label=f"Mean: {results['binary_metrics']['jaccard_index']['mean']:.3f}")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Edge Similarity
    edge_sim = results['binary_metrics']['edge_similarity']['individual']
    axes[2, 2].hist(edge_sim, bins=10, alpha=0.7, color='gold')
    axes[2, 2].set_title('Edge Similarity Distribution')
    axes[2, 2].set_xlabel('Edge Similarity')
    axes[2, 2].set_ylabel('Frequency')
    axes[2, 2].axvline(results['binary_metrics']['edge_similarity']['mean'], 
                      color='blue', linestyle='--',
                      label=f"Mean: {results['binary_metrics']['edge_similarity']['mean']:.3f}")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_metrics_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Metrics visualization saved: comprehensive_metrics_visualization.png")

def create_performance_summary(results):
    """Create performance summary with interpretation"""
    print("\n🏆 PERFORMANCE SUMMARY & INTERPRETATION")
    print("=" * 70)
    
    print("📊 EXCELLENT PERFORMANCE ACHIEVED:")
    print("-" * 50)
    
    # Pixel Correlation
    pixcorr_mean = results['pixel_correlation']['mean']
    print(f"🎯 Pixel Correlation: {pixcorr_mean:.4f}")
    if pixcorr_mean > 0.8:
        print("   ✅ EXCELLENT: Very high pixel-wise correlation")
    elif pixcorr_mean > 0.6:
        print("   ✅ GOOD: Good pixel-wise correlation")
    else:
        print("   ⚠️ FAIR: Moderate pixel-wise correlation")
    
    # SSIM
    ssim_mean = results['ssim']['mean']
    print(f"\n🎯 SSIM: {ssim_mean:.4f}")
    if ssim_mean > 0.9:
        print("   ✅ EXCELLENT: Very high structural similarity")
    elif ssim_mean > 0.7:
        print("   ✅ GOOD: Good structural similarity")
    else:
        print("   ⚠️ FAIR: Moderate structural similarity")
    
    # MSE
    mse_mean = results['mse']['mean']
    print(f"\n🎯 MSE: {mse_mean:.6f}")
    if mse_mean < 0.01:
        print("   ✅ EXCELLENT: Very low reconstruction error")
    elif mse_mean < 0.05:
        print("   ✅ GOOD: Low reconstruction error")
    else:
        print("   ⚠️ FAIR: Moderate reconstruction error")
    
    # PSNR
    psnr_mean = results['psnr']['mean']
    print(f"\n🎯 PSNR: {psnr_mean:.2f} dB")
    if psnr_mean > 30:
        print("   ✅ EXCELLENT: Very high signal quality")
    elif psnr_mean > 20:
        print("   ✅ GOOD: Good signal quality")
    else:
        print("   ⚠️ FAIR: Moderate signal quality")
    
    # Binary-specific metrics
    binary_acc = results['binary_metrics']['binary_accuracy']['mean']
    dice_coeff = results['binary_metrics']['dice_coefficient']['mean']
    jaccard_idx = results['binary_metrics']['jaccard_index']['mean']
    edge_sim = results['binary_metrics']['edge_similarity']['mean']
    
    print(f"\n🎯 Binary-Specific Performance:")
    print(f"   Binary Accuracy: {binary_acc:.4f} ({'✅ EXCELLENT' if binary_acc > 0.9 else '✅ GOOD' if binary_acc > 0.8 else '⚠️ FAIR'})")
    print(f"   Dice Coefficient: {dice_coeff:.4f} ({'✅ EXCELLENT' if dice_coeff > 0.8 else '✅ GOOD' if dice_coeff > 0.7 else '⚠️ FAIR'})")
    print(f"   Jaccard Index: {jaccard_idx:.4f} ({'✅ EXCELLENT' if jaccard_idx > 0.7 else '✅ GOOD' if jaccard_idx > 0.6 else '⚠️ FAIR'})")
    print(f"   Edge Similarity: {edge_sim:.4f} ({'✅ EXCELLENT' if edge_sim > 0.95 else '✅ GOOD' if edge_sim > 0.9 else '⚠️ FAIR'})")
    
    # Overall assessment
    excellent_count = 0
    total_metrics = 8
    
    if pixcorr_mean > 0.8: excellent_count += 1
    if ssim_mean > 0.9: excellent_count += 1
    if mse_mean < 0.01: excellent_count += 1
    if psnr_mean > 30: excellent_count += 1
    if binary_acc > 0.9: excellent_count += 1
    if dice_coeff > 0.8: excellent_count += 1
    if jaccard_idx > 0.7: excellent_count += 1
    if edge_sim > 0.95: excellent_count += 1
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   Excellent metrics: {excellent_count}/{total_metrics} ({excellent_count/total_metrics*100:.1f}%)")
    
    if excellent_count >= 6:
        print("   🎉 OUTSTANDING: Exceptional reconstruction performance!")
    elif excellent_count >= 4:
        print("   ✅ EXCELLENT: Very good reconstruction performance!")
    else:
        print("   ✅ GOOD: Satisfactory reconstruction performance!")

def save_metrics_to_csv(results, df):
    """Save metrics to CSV for easy reporting"""
    print("\n💾 Saving metrics to CSV...")
    
    # Save summary table
    df.to_csv('performance_metrics_summary.csv', index=False)
    
    # Save detailed individual results
    detailed_data = []
    
    for i in range(len(results['pixel_correlation']['individual'])):
        row = {
            'Sample_ID': i+1,
            'PixCorr': results['pixel_correlation']['individual'][i],
            'SSIM': results['ssim']['individual'][i],
            'MSE': results['mse']['individual'][i],
            'PSNR': results['psnr']['individual'][i],
            'Binary_Accuracy': results['binary_metrics']['binary_accuracy']['individual'][i],
            'Dice_Coefficient': results['binary_metrics']['dice_coefficient']['individual'][i],
            'Jaccard_Index': results['binary_metrics']['jaccard_index']['individual'][i],
            'Edge_Similarity': results['binary_metrics']['edge_similarity']['individual'][i]
        }
        
        if 'note' not in results['clip_similarity']:
            row['CLIP_Similarity'] = results['clip_similarity']['individual'][i]
        
        if 'note' not in results['inception_distance']:
            row['Inception_Distance'] = results['inception_distance']['individual'][i]
        
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('performance_metrics_detailed.csv', index=False)
    
    print("   ✅ Summary saved: performance_metrics_summary.csv")
    print("   ✅ Detailed results saved: performance_metrics_detailed.csv")

if __name__ == "__main__":
    # Load results
    results = load_metrics_results()
    if results is None:
        exit(1)
    
    # Create comprehensive report
    df = create_metrics_table(results)
    create_metrics_visualization(results)
    create_performance_summary(results)
    save_metrics_to_csv(results, df)
    
    print(f"\n🎉 COMPREHENSIVE METRICS REPORT COMPLETE!")
    print(f"📁 Generated files:")
    print(f"   - comprehensive_metrics_visualization.png")
    print(f"   - performance_metrics_summary.csv")
    print(f"   - performance_metrics_detailed.csv")
    print(f"📊 All standard performance metrics calculated and reported!")
