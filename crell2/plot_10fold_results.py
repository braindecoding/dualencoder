#!/usr/bin/env python3
"""
Plot 10-Fold CV Results from Saved Data
Generate comprehensive plots from existing 10-fold CV results
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import os

def load_latest_10fold_results():
    """
    Load the latest 10-fold CV results file
    """
    # Find all 10-fold CV results files
    pattern = "advanced_cv_results_10fold_*.pkl"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No 10-fold CV results files found!")
        print("   Expected pattern: advanced_cv_results_10fold_YYYYMMDD_HHMMSS.pkl")
        return None
    
    # Get the latest file
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 Loading latest 10-fold results: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        cv_results = pickle.load(f)
    
    return cv_results, latest_file

def plot_10fold_cv_results(cv_results):
    """
    Plot comprehensive 10-fold CV results
    """
    print("📊 Generating comprehensive 10-fold CV plots...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('10-Fold Cross-Validation Results - Advanced EEG Training', fontsize=16)
    
    # Extract data
    fold_results = cv_results['fold_results']
    stats = cv_results['cv_statistics']
    
    best_accuracies = [result['best_val_accuracy'] for result in fold_results]
    final_accuracies = [result['final_val_accuracy'] for result in fold_results]
    training_times = [result['training_time']/60 for result in fold_results]  # Convert to minutes
    epochs_trained = [result['epochs_trained'] for result in fold_results]
    
    print(f"📊 Data extracted:")
    print(f"   Best accuracies: {[f'{acc:.3f}' for acc in best_accuracies]}")
    print(f"   Final accuracies: {[f'{acc:.3f}' for acc in final_accuracies]}")
    print(f"   Training times: {[f'{time:.1f}min' for time in training_times]}")
    
    # Plot 1: Best Validation Accuracy by Fold
    fold_numbers = [f"Fold {i+1}" for i in range(len(best_accuracies))]
    bars1 = axes[0, 0].bar(fold_numbers, best_accuracies, alpha=0.7, color='skyblue')
    axes[0, 0].axhline(y=stats['mean_val_accuracy'], color='red', linestyle='--', 
                      label=f"Mean: {stats['mean_val_accuracy']:.3f}")
    axes[0, 0].axhline(y=stats['mean_val_accuracy'] + stats['std_val_accuracy'], 
                      color='red', linestyle=':', alpha=0.5, label=f"±Std: {stats['std_val_accuracy']:.3f}")
    axes[0, 0].axhline(y=stats['mean_val_accuracy'] - stats['std_val_accuracy'], 
                      color='red', linestyle=':', alpha=0.5)
    axes[0, 0].set_ylabel('Best Validation Accuracy')
    axes[0, 0].set_title('Best Validation Accuracy by Fold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, best_accuracies):
        height = bar.get_height()
        axes[0, 0].annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Final Validation Accuracy by Fold
    bars2 = axes[0, 1].bar(fold_numbers, final_accuracies, alpha=0.7, color='lightgreen')
    axes[0, 1].axhline(y=stats['mean_final_accuracy'], color='blue', linestyle='--',
                      label=f"Mean: {stats['mean_final_accuracy']:.3f}")
    axes[0, 1].set_ylabel('Final Validation Accuracy')
    axes[0, 1].set_title('Final Validation Accuracy by Fold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, acc in zip(bars2, final_accuracies):
        height = bar.get_height()
        axes[0, 1].annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Training Time by Fold
    bars3 = axes[0, 2].bar(fold_numbers, training_times, alpha=0.7, color='orange')
    axes[0, 2].set_ylabel('Training Time (minutes)')
    axes[0, 2].set_title('Training Time by Fold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars3, training_times):
        height = bar.get_height()
        axes[0, 2].annotate(f'{time:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Accuracy Distribution
    axes[1, 0].hist(best_accuracies, bins=5, alpha=0.7, color='lightblue', edgecolor='black', label='Best Accuracy')
    axes[1, 0].axvline(x=stats['mean_val_accuracy'], color='red', linestyle='--', 
                      label=f"Mean: {stats['mean_val_accuracy']:.3f}")
    axes[1, 0].axvline(x=0.1, color='gray', linestyle='--', label='Random: 0.1')
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Accuracy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Best vs Final Accuracy Scatter
    axes[1, 1].scatter(best_accuracies, final_accuracies, s=100, alpha=0.7, c=range(len(best_accuracies)), cmap='viridis')
    for i, (best, final) in enumerate(zip(best_accuracies, final_accuracies)):
        axes[1, 1].annotate(f'F{i+1}', (best, final), xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line
    min_acc = min(min(best_accuracies), min(final_accuracies))
    max_acc = max(max(best_accuracies), max(final_accuracies))
    axes[1, 1].plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5, label='Perfect Agreement')
    
    axes[1, 1].set_xlabel('Best Validation Accuracy')
    axes[1, 1].set_ylabel('Final Validation Accuracy')
    axes[1, 1].set_title('Best vs Final Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Epochs Trained by Fold
    bars6 = axes[1, 2].bar(fold_numbers, epochs_trained, alpha=0.7, color='purple')
    axes[1, 2].set_ylabel('Epochs Trained')
    axes[1, 2].set_title('Training Epochs by Fold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, epochs in zip(bars6, epochs_trained):
        height = bar.get_height()
        axes[1, 2].annotate(f'{epochs}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 7: Statistics Summary
    stats_text = f"""10-Fold Cross-Validation Summary:

Best Validation Accuracy:
  Mean: {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}
  Range: {min(best_accuracies):.4f} - {max(best_accuracies):.4f}

Final Validation Accuracy:
  Mean: {stats['mean_final_accuracy']:.4f} ± {stats['std_final_accuracy']:.4f}
  Range: {min(final_accuracies):.4f} - {max(final_accuracies):.4f}

Test Set Accuracy: {stats['test_accuracy']:.4f}
Best Fold: {stats['best_fold']}

Training Statistics:
  Avg Training Time: {np.mean(training_times):.1f} ± {np.std(training_times):.1f} min
  Avg Epochs: {np.mean(epochs_trained):.1f} ± {np.std(epochs_trained):.1f}
  Total Training Time: {sum(training_times):.1f} min

Performance vs Random (10%):
  Improvement: {stats['mean_val_accuracy']/0.1:.1f}x
  Percentage: {stats['mean_val_accuracy']*100:.1f}%"""
    
    axes[2, 0].text(0.05, 0.95, stats_text, transform=axes[2, 0].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[2, 0].set_xlim(0, 1)
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].axis('off')
    axes[2, 0].set_title('CV Statistics Summary')
    
    # Plot 8: Performance Consistency
    cv_coefficient = stats['std_val_accuracy'] / stats['mean_val_accuracy']
    consistency_metrics = {
        'Mean Accuracy': stats['mean_val_accuracy'],
        'Std Accuracy': stats['std_val_accuracy'],
        'CV Coefficient': cv_coefficient,
        'Min Accuracy': min(best_accuracies),
        'Max Accuracy': max(best_accuracies),
        'Range': max(best_accuracies) - min(best_accuracies)
    }
    
    metrics_names = list(consistency_metrics.keys())
    metrics_values = list(consistency_metrics.values())
    
    bars8 = axes[2, 1].bar(metrics_names, metrics_values, alpha=0.7, color='lightcoral')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].set_title('Performance Consistency Metrics')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars8, metrics_values):
        height = bar.get_height()
        axes[2, 1].annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # Plot 9: Fold Ranking
    fold_ranking = sorted(enumerate(best_accuracies), key=lambda x: x[1], reverse=True)
    rank_folds = [f"Fold {fold+1}" for fold, _ in fold_ranking]
    rank_accuracies = [acc for _, acc in fold_ranking]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rank_accuracies)))
    bars9 = axes[2, 2].bar(range(len(rank_folds)), rank_accuracies, alpha=0.7, color=colors)
    axes[2, 2].set_xticks(range(len(rank_folds)))
    axes[2, 2].set_xticklabels(rank_folds, rotation=45)
    axes[2, 2].set_ylabel('Best Validation Accuracy')
    axes[2, 2].set_title('Fold Performance Ranking')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars9, rank_accuracies):
        height = bar.get_height()
        axes[2, 2].annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'advanced_10fold_cv_comprehensive_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 10-fold CV comprehensive plot saved: {plot_filename}")
    
    return plot_filename

def main():
    """
    Main function to plot 10-fold CV results
    """
    print("📊 PLOTTING 10-FOLD CV RESULTS - ADVANCED EEG TRAINING")
    print("=" * 70)
    
    # Load latest 10-fold CV results
    result = load_latest_10fold_results()
    if result is None:
        return
    
    cv_results, results_file = result
    
    # Generate comprehensive plots
    plot_filename = plot_10fold_cv_results(cv_results)
    
    print(f"\n✅ PLOTTING COMPLETED!")
    print(f"   Source: {results_file}")
    print(f"   Plot: {plot_filename}")
    
    # Print summary
    stats = cv_results['cv_statistics']
    print(f"\n📊 SUMMARY:")
    print(f"   Mean Validation Accuracy: {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}")
    print(f"   Test Set Accuracy: {stats['test_accuracy']:.4f}")
    print(f"   Best Fold: {stats['best_fold']}")
    print(f"   Improvement over Random: {stats['mean_val_accuracy']/0.1:.1f}x")

if __name__ == "__main__":
    main()
