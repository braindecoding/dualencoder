#!/usr/bin/env python3
"""
Run 10-Fold Cross-Validation for Advanced EEG Training
Enhanced EEG Transformer with Contrastive Learning
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Import the main functions
from advanced_training import run_10fold_cross_validation

def main():
    """
    Main function to run 10-fold cross-validation for advanced EEG training
    """
    print("🚀 STARTING 10-FOLD CROSS-VALIDATION - ADVANCED EEG TRAINING")
    print("=" * 80)
    print("Configuration: Enhanced EEG Transformer")
    print("Architecture: Spatial Projection → Patch Embedding → Transformer → Contrastive Learning")
    print("Dataset: Crell (10 letters: a,d,e,f,j,n,o,s,t,v)")
    print("Loss: Adaptive Temperature Contrastive Loss")
    print("Optimizer: AdamW with Differential Learning Rates")
    print("Scheduler: Warmup + Cosine Annealing")
    print("Reproducibility: Stratified 10-fold with random seeds")
    print("=" * 80)
    
    start_time = datetime.now()
    print(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run 10-fold cross-validation
        print("\n🔄 Running 10-fold cross-validation...")
        cv_results, histories = run_10fold_cross_validation()
        
        # Print final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🏆 10-FOLD CV COMPLETED SUCCESSFULLY!")
        print(f"=" * 60)
        print(f"🕐 Total time: {duration}")
        print(f"📊 Results:")
        
        stats = cv_results['cv_statistics']
        print(f"   Mean Validation Accuracy: {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}")
        print(f"   Mean Final Accuracy: {stats['mean_final_accuracy']:.4f} ± {stats['std_final_accuracy']:.4f}")
        print(f"   Test Set Accuracy: {stats['test_accuracy']:.4f}")
        print(f"   Best Fold: {stats['best_fold']}")
        
        print(f"\n📁 Files saved:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"   Results: advanced_cv_results_10fold_{timestamp}.pkl")
        print(f"   Models: advanced_eeg_fold_[1-10]_best.pth")
        print(f"   Plot: advanced_training_results_{timestamp}.png")
        
        # Performance analysis
        print(f"\n🔍 PERFORMANCE ANALYSIS:")
        fold_results = cv_results['fold_results']
        val_accuracies = [r['best_val_accuracy'] for r in fold_results]
        final_accuracies = [r['final_val_accuracy'] for r in fold_results]
        
        print(f"   Individual fold validation accuracies:")
        for i, acc in enumerate(val_accuracies):
            print(f"     Fold {i+1}: {acc:.4f} ({acc*100:.1f}%)")
        
        print(f"   Individual fold final accuracies:")
        for i, acc in enumerate(final_accuracies):
            print(f"     Fold {i+1}: {acc:.4f} ({acc*100:.1f}%)")
        
        print(f"   Best fold validation accuracy: {max(val_accuracies):.4f} ({max(val_accuracies)*100:.1f}%)")
        print(f"   Worst fold validation accuracy: {min(val_accuracies):.4f} ({min(val_accuracies)*100:.1f}%)")
        print(f"   Validation accuracy range: {max(val_accuracies) - min(val_accuracies):.4f}")
        
        # Performance interpretation
        print(f"\n📈 PERFORMANCE INTERPRETATION:")
        mean_acc = stats['mean_val_accuracy']
        std_acc = stats['std_val_accuracy']
        
        if mean_acc > 0.8:
            print(f"   🎉 EXCELLENT: Mean accuracy > 80%")
            print(f"      Model shows outstanding performance")
        elif mean_acc > 0.6:
            print(f"   ✅ VERY GOOD: Mean accuracy > 60%")
            print(f"      Model shows strong performance")
        elif mean_acc > 0.4:
            print(f"   👍 GOOD: Mean accuracy > 40%")
            print(f"      Model shows reasonable performance")
        elif mean_acc > 0.2:
            print(f"   ⚠️  MODERATE: Mean accuracy > 20%")
            print(f"      Model shows some learning capability")
        else:
            print(f"   ❌ POOR: Mean accuracy ≤ 20%")
            print(f"      Model needs significant improvement")
        
        # Consistency analysis
        cv_coefficient = std_acc / mean_acc if mean_acc > 0 else float('inf')
        print(f"\n📊 CONSISTENCY ANALYSIS:")
        print(f"   Coefficient of Variation: {cv_coefficient:.3f}")
        
        if cv_coefficient < 0.1:
            print(f"   🎯 VERY CONSISTENT: CV < 0.1")
            print(f"      Model performance is highly stable across folds")
        elif cv_coefficient < 0.2:
            print(f"   ✅ CONSISTENT: CV < 0.2")
            print(f"      Model performance is reasonably stable")
        elif cv_coefficient < 0.3:
            print(f"   ⚠️  MODERATE VARIANCE: CV < 0.3")
            print(f"      Some variability in performance across folds")
        else:
            print(f"   ❌ HIGH VARIANCE: CV ≥ 0.3")
            print(f"      Significant variability - may need more robust training")
        
        # Comparison with baselines
        print(f"\n🎯 BASELINE COMPARISON:")
        print(f"   Random baseline (10 classes): ~10%")
        print(f"   Current model: {mean_acc*100:.1f}%")
        
        improvement_factor = mean_acc / 0.1  # vs random
        print(f"   Improvement over random: {improvement_factor:.1f}x")
        
        if improvement_factor > 8:
            print(f"   🚀 OUTSTANDING: >8x improvement over random")
        elif improvement_factor > 6:
            print(f"   🎉 EXCELLENT: >6x improvement over random")
        elif improvement_factor > 4:
            print(f"   ✅ VERY GOOD: >4x improvement over random")
        elif improvement_factor > 2:
            print(f"   👍 GOOD: >2x improvement over random")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: <2x improvement over random")
        
        print(f"\n🎯 NEXT STEPS:")
        if mean_acc > 0.7:
            print(f"   🚀 Model performing excellently! Consider:")
            print(f"     - Fine-tuning hyperparameters for even better performance")
            print(f"     - Exploring model interpretability and attention visualization")
            print(f"     - Testing on additional datasets for generalization")
            print(f"     - Deploying for real-world applications")
        elif mean_acc > 0.5:
            print(f"   🔧 Model shows strong promise! Consider:")
            print(f"     - Increasing model capacity (more layers/dimensions)")
            print(f"     - Longer training with more epochs")
            print(f"     - Advanced data augmentation techniques")
            print(f"     - Ensemble methods combining multiple folds")
        elif mean_acc > 0.3:
            print(f"   ⚠️  Model needs improvement! Consider:")
            print(f"     - Reviewing data preprocessing pipeline")
            print(f"     - Trying different loss functions")
            print(f"     - Adjusting learning rate schedules")
            print(f"     - Adding regularization techniques")
        else:
            print(f"   🔧 Significant improvements needed! Consider:")
            print(f"     - Checking data quality and preprocessing")
            print(f"     - Simplifying model architecture")
            print(f"     - Debugging training pipeline")
            print(f"     - Reviewing feature engineering")
        
        return cv_results, histories
        
    except Exception as e:
        print(f"\n❌ ERROR during 10-fold CV:")
        print(f"   {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check if all required files exist")
        print(f"   2. Verify CUDA/GPU availability")
        print(f"   3. Check memory usage (may need smaller batch size)")
        print(f"   4. Ensure all dependencies are installed")
        print(f"   5. Check data file paths and formats")
        
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        
        return None, None

if __name__ == "__main__":
    # Run the 10-fold cross-validation
    results = main()
    
    if results[0] is not None:
        print(f"\n✅ 10-fold cross-validation completed successfully!")
        print(f"   Advanced EEG transformer evaluation complete")
        print(f"   Ready for deployment or further optimization")
    else:
        print(f"\n❌ 10-fold cross-validation failed!")
        print(f"   Check error messages above for debugging")
