#!/usr/bin/env python3
"""
CLIP vs Non-CLIP Comparison
High Priority Task 4: Compare CLIP vs non-CLIP results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_ind
import seaborn as sns

class CLIPvsNonCLIPComparator:
    """Comprehensive comparison between CLIP and non-CLIP approaches"""
    
    def __init__(self):
        self.results = {}
        
        print(f"🔍 CLIP vs NON-CLIP COMPARATOR")
        print("=" * 40)
    
    def load_results(self):
        """Load results from different evaluations"""
        print(f"\n📊 LOADING EVALUATION RESULTS")
        
        # Load comprehensive evaluation (non-CLIP)
        try:
            with open('comprehensive_evaluation_report.pkl', 'rb') as f:
                nonclip_results = pickle.load(f)
            self.results['non_clip'] = nonclip_results['model_results']
            print(f"   ✅ Non-CLIP results loaded")
        except FileNotFoundError:
            print(f"   ⚠️ Non-CLIP results not found")
            self.results['non_clip'] = {}
        
        # Load CLIP evaluation results
        try:
            with open('clip_comprehensive_evaluation_report.pkl', 'rb') as f:
                clip_results = pickle.load(f)
            self.results['clip'] = clip_results['model_results']
            print(f"   ✅ CLIP results loaded")
        except FileNotFoundError:
            print(f"   ⚠️ CLIP results not found")
            self.results['clip'] = {}
        
        # Load CLIP weights comparison
        try:
            with open('clip_weights_comparison.pkl', 'rb') as f:
                weights_results = pickle.load(f)
            self.results['clip_weights'] = weights_results
            print(f"   ✅ CLIP weights comparison loaded")
        except FileNotFoundError:
            print(f"   ⚠️ CLIP weights comparison not found")
            self.results['clip_weights'] = {}
    
    def compare_baseline_models(self):
        """Compare baseline model dengan dan tanpa CLIP"""
        print(f"\n🎯 COMPARING BASELINE MODELS")
        print("=" * 40)
        
        # Non-CLIP baseline
        if 'baseline' in self.results['non_clip']:
            nonclip_baseline = self.results['non_clip']['baseline']['metrics']
            print(f"📊 Non-CLIP Baseline:")
            print(f"   MSE: {nonclip_baseline['mse']:.4f}")
            print(f"   SSIM: {nonclip_baseline['ssim_mean']:.4f}")
            print(f"   Correlation: {nonclip_baseline['correlation_mean']:.4f}")
        
        # CLIP baseline
        if 'baseline' in self.results['clip']:
            clip_baseline = self.results['clip']['baseline']['traditional_metrics']
            clip_scores = self.results['clip']['baseline']['clip_metrics']
            print(f"📊 CLIP-Evaluated Baseline:")
            print(f"   MSE: {clip_baseline['mse']:.4f}")
            print(f"   SSIM: {clip_baseline['ssim_mean']:.4f}")
            print(f"   Correlation: {clip_baseline['correlation_mean']:.4f}")
            print(f"   CLIP Score: {clip_scores['mean']:.4f}")
    
    def compare_enhanced_ldm_models(self):
        """Compare Enhanced LDM dengan dan tanpa CLIP"""
        print(f"\n🚀 COMPARING ENHANCED LDM MODELS")
        print("=" * 40)
        
        # Non-CLIP Enhanced LDM
        if 'enhanced_ldm' in self.results['non_clip']:
            nonclip_ldm = self.results['non_clip']['enhanced_ldm']['metrics']
            print(f"📊 Non-CLIP Enhanced LDM:")
            print(f"   MSE: {nonclip_ldm['mse']:.4f}")
            print(f"   SSIM: {nonclip_ldm['ssim_mean']:.4f}")
            print(f"   Correlation: {nonclip_ldm['correlation_mean']:.4f}")
        
        # CLIP Enhanced LDM
        if 'enhanced_ldm' in self.results['clip']:
            clip_ldm = self.results['clip']['enhanced_ldm']['traditional_metrics']
            clip_scores = self.results['clip']['enhanced_ldm']['clip_metrics']
            print(f"📊 CLIP-Evaluated Enhanced LDM:")
            print(f"   MSE: {clip_ldm['mse']:.4f}")
            print(f"   SSIM: {clip_ldm['ssim_mean']:.4f}")
            print(f"   Correlation: {clip_ldm['correlation_mean']:.4f}")
            print(f"   CLIP Score: {clip_scores['mean']:.4f}")
    
    def analyze_clip_weights_impact(self):
        """Analyze impact of different CLIP weights"""
        print(f"\n🎨 ANALYZING CLIP WEIGHTS IMPACT")
        print("=" * 40)
        
        if not self.results['clip_weights']:
            print(f"   ⚠️ No CLIP weights data available")
            return
        
        weights = []
        clip_scores = []
        losses = []
        
        for weight, results in self.results['clip_weights'].items():
            weights.append(weight)
            clip_scores.append(results['best_clip_score'])
            losses.append(results['best_loss'])
            
            print(f"📊 CLIP Weight {weight}:")
            print(f"   Best CLIP Score: {results['best_clip_score']:.4f}")
            print(f"   Best Loss: {results['best_loss']:.4f}")
        
        # Find optimal weight
        best_weight_idx = np.argmax(clip_scores)
        best_weight = weights[best_weight_idx]
        best_score = clip_scores[best_weight_idx]
        
        print(f"\n🏆 OPTIMAL CLIP WEIGHT:")
        print(f"   Weight: {best_weight}")
        print(f"   CLIP Score: {best_score:.4f}")
        
        return weights, clip_scores, losses, best_weight
    
    def statistical_significance_test(self):
        """Perform statistical significance tests"""
        print(f"\n📊 STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 40)
        
        # Compare correlations between CLIP and non-CLIP
        if ('baseline' in self.results['non_clip'] and 
            'baseline' in self.results['clip']):
            
            # Get correlation arrays
            nonclip_corrs = self.results['non_clip']['baseline']['metrics']['correlations']
            clip_corrs = self.results['clip']['baseline']['traditional_metrics']['correlations']
            
            # T-test
            t_stat, p_value = ttest_ind(nonclip_corrs, clip_corrs)
            
            print(f"📊 Baseline Correlation Comparison:")
            print(f"   Non-CLIP mean: {np.mean(nonclip_corrs):.4f}")
            print(f"   CLIP mean: {np.mean(clip_corrs):.4f}")
            print(f"   T-statistic: {t_stat:.4f}")
            print(f"   P-value: {p_value:.4f}")
            print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    def create_comprehensive_comparison_plot(self):
        """Create comprehensive comparison visualization"""
        print(f"\n🎨 CREATING COMPREHENSIVE COMPARISON PLOT")
        print("=" * 45)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CLIP vs Non-CLIP Comprehensive Comparison', fontsize=16)
        
        # 1. Baseline Comparison
        if ('baseline' in self.results['non_clip'] and 
            'baseline' in self.results['clip']):
            
            metrics = ['MSE', 'SSIM', 'Correlation']
            nonclip_vals = [
                self.results['non_clip']['baseline']['metrics']['mse'],
                self.results['non_clip']['baseline']['metrics']['ssim_mean'],
                self.results['non_clip']['baseline']['metrics']['correlation_mean']
            ]
            clip_vals = [
                self.results['clip']['baseline']['traditional_metrics']['mse'],
                self.results['clip']['baseline']['traditional_metrics']['ssim_mean'],
                self.results['clip']['baseline']['traditional_metrics']['correlation_mean']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, nonclip_vals, width, label='Non-CLIP', alpha=0.7)
            axes[0, 0].bar(x + width/2, clip_vals, width, label='CLIP Evaluated', alpha=0.7)
            axes[0, 0].set_title('Baseline Model Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(metrics)
            axes[0, 0].legend()
        
        # 2. Enhanced LDM Comparison
        if ('enhanced_ldm' in self.results['non_clip'] and 
            'enhanced_ldm' in self.results['clip']):
            
            nonclip_vals = [
                self.results['non_clip']['enhanced_ldm']['metrics']['mse'],
                self.results['non_clip']['enhanced_ldm']['metrics']['ssim_mean'],
                self.results['non_clip']['enhanced_ldm']['metrics']['correlation_mean']
            ]
            clip_vals = [
                self.results['clip']['enhanced_ldm']['traditional_metrics']['mse'],
                self.results['clip']['enhanced_ldm']['traditional_metrics']['ssim_mean'],
                self.results['clip']['enhanced_ldm']['traditional_metrics']['correlation_mean']
            ]
            
            axes[0, 1].bar(x - width/2, nonclip_vals, width, label='Non-CLIP', alpha=0.7)
            axes[0, 1].bar(x + width/2, clip_vals, width, label='CLIP Evaluated', alpha=0.7)
            axes[0, 1].set_title('Enhanced LDM Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(metrics)
            axes[0, 1].legend()
        
        # 3. CLIP Scores (only for CLIP evaluated models)
        if self.results['clip']:
            models = []
            clip_scores = []
            
            for model_name, results in self.results['clip'].items():
                if 'clip_metrics' in results:
                    models.append(model_name.replace('_', ' ').title())
                    clip_scores.append(results['clip_metrics']['mean'])
            
            if models:
                axes[0, 2].bar(models, clip_scores, alpha=0.7, color='red')
                axes[0, 2].set_title('CLIP Scores by Model')
                axes[0, 2].set_ylabel('CLIP Score')
                axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. CLIP Weights Impact
        if self.results['clip_weights']:
            weights, clip_scores, losses, best_weight = self.analyze_clip_weights_impact()
            
            axes[1, 0].plot(weights, clip_scores, 'o-', label='CLIP Score', color='red')
            axes[1, 0].axvline(x=best_weight, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('CLIP Weight Impact on Score')
            axes[1, 0].set_xlabel('CLIP Weight')
            axes[1, 0].set_ylabel('CLIP Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            ax_twin = axes[1, 0].twinx()
            ax_twin.plot(weights, losses, 's-', label='Loss', color='blue')
            ax_twin.set_ylabel('Loss', color='blue')
            ax_twin.legend(loc='upper right')
        
        # 5. Correlation Distribution Comparison
        if ('baseline' in self.results['non_clip'] and 
            'baseline' in self.results['clip']):
            
            nonclip_corrs = self.results['non_clip']['baseline']['metrics']['correlations']
            clip_corrs = self.results['clip']['baseline']['traditional_metrics']['correlations']
            
            axes[1, 1].hist(nonclip_corrs, bins=10, alpha=0.7, label='Non-CLIP', density=True)
            axes[1, 1].hist(clip_corrs, bins=10, alpha=0.7, label='CLIP Evaluated', density=True)
            axes[1, 1].set_title('Correlation Distribution Comparison')
            axes[1, 1].set_xlabel('Correlation')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
        
        # 6. Performance Summary
        summary_data = []
        summary_labels = []
        
        # Collect all available metrics
        if 'baseline' in self.results['non_clip']:
            summary_data.append([
                self.results['non_clip']['baseline']['metrics']['mse'],
                self.results['non_clip']['baseline']['metrics']['ssim_mean'],
                self.results['non_clip']['baseline']['metrics']['correlation_mean']
            ])
            summary_labels.append('Baseline\n(Non-CLIP)')
        
        if 'enhanced_ldm' in self.results['non_clip']:
            summary_data.append([
                self.results['non_clip']['enhanced_ldm']['metrics']['mse'],
                self.results['non_clip']['enhanced_ldm']['metrics']['ssim_mean'],
                self.results['non_clip']['enhanced_ldm']['metrics']['correlation_mean']
            ])
            summary_labels.append('Enhanced LDM\n(Non-CLIP)')
        
        if 'baseline' in self.results['clip']:
            summary_data.append([
                self.results['clip']['baseline']['traditional_metrics']['mse'],
                self.results['clip']['baseline']['traditional_metrics']['ssim_mean'],
                self.results['clip']['baseline']['traditional_metrics']['correlation_mean']
            ])
            summary_labels.append('Baseline\n(CLIP)')
        
        if 'enhanced_ldm' in self.results['clip']:
            summary_data.append([
                self.results['clip']['enhanced_ldm']['traditional_metrics']['mse'],
                self.results['clip']['enhanced_ldm']['traditional_metrics']['ssim_mean'],
                self.results['clip']['enhanced_ldm']['traditional_metrics']['correlation_mean']
            ])
            summary_labels.append('Enhanced LDM\n(CLIP)')
        
        if summary_data:
            summary_data = np.array(summary_data).T  # Transpose for heatmap
            
            im = axes[1, 2].imshow(summary_data, cmap='RdYlBu_r', aspect='auto')
            axes[1, 2].set_title('Performance Heatmap')
            axes[1, 2].set_xticks(range(len(summary_labels)))
            axes[1, 2].set_xticklabels(summary_labels, rotation=45)
            axes[1, 2].set_yticks(range(3))
            axes[1, 2].set_yticklabels(['MSE', 'SSIM', 'Correlation'])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 2])
            
            # Add text annotations
            for i in range(3):
                for j in range(len(summary_labels)):
                    text = axes[1, 2].text(j, i, f'{summary_data[i, j]:.3f}',
                                         ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig('clip_vs_nonclip_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self):
        """Generate final comparison report"""
        print(f"\n📋 GENERATING COMPARISON REPORT")
        print("=" * 40)
        
        report = {
            'comparison_summary': {
                'non_clip_models': list(self.results['non_clip'].keys()) if self.results['non_clip'] else [],
                'clip_models': list(self.results['clip'].keys()) if self.results['clip'] else [],
                'clip_weights_tested': list(self.results['clip_weights'].keys()) if self.results['clip_weights'] else []
            },
            'results': self.results
        }
        
        # Save report
        with open('clip_vs_nonclip_comparison_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        print(f"\n🎯 COMPARISON SUMMARY")
        print("=" * 30)
        
        print(f"📊 Models Compared:")
        print(f"   Non-CLIP: {len(self.results['non_clip'])} models")
        print(f"   CLIP: {len(self.results['clip'])} models")
        print(f"   CLIP Weights: {len(self.results['clip_weights'])} configurations")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        if self.results['clip_weights']:
            weights, clip_scores, losses, best_weight = self.analyze_clip_weights_impact()
            print(f"   🏆 Best CLIP Weight: {best_weight}")
            print(f"   📈 CLIP Score Range: {min(clip_scores):.4f} - {max(clip_scores):.4f}")
        
        print(f"\n📁 Generated files:")
        print(f"   - clip_vs_nonclip_comprehensive_comparison.png")
        print(f"   - clip_vs_nonclip_comparison_report.pkl")
        
        return report

def main():
    """Main comparison function"""
    print("🎯 HIGH PRIORITY: CLIP vs NON-CLIP COMPARISON")
    print("=" * 60)
    
    comparator = CLIPvsNonCLIPComparator()
    
    # Load all results
    comparator.load_results()
    
    # Perform comparisons
    comparator.compare_baseline_models()
    comparator.compare_enhanced_ldm_models()
    comparator.analyze_clip_weights_impact()
    comparator.statistical_significance_test()
    
    # Create visualizations
    comparator.create_comprehensive_comparison_plot()
    
    # Generate final report
    report = comparator.generate_comparison_report()
    
    print(f"\n✅ CLIP vs NON-CLIP COMPARISON COMPLETED!")

if __name__ == "__main__":
    main()
