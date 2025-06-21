#!/usr/bin/env python3
"""
Comprehensive Evaluation of All Approaches
Compare Original LDM vs Enhanced LDM vs Baseline Regression
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import time

# Import models
from simple_baseline_model import SimpleRegressionModel, Digit69BaselineDataset
from improved_unet import ImprovedUNet
from enhanced_training_pipeline import EnhancedDiffusionModel

class ComprehensiveEvaluator:
    """Comprehensive evaluation of all approaches"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        print("🔍 COMPREHENSIVE EVALUATION OF ALL APPROACHES")
        print("=" * 60)
        print(f"📱 Device: {device}")
    
    def load_test_data(self):
        """Load test dataset"""
        print("\n📊 LOADING TEST DATA")
        print("=" * 30)
        
        test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
        
        # Get all test data
        fmri_embeddings = []
        target_images = []
        
        for i in range(len(test_dataset)):
            fmri_emb, img = test_dataset[i]
            fmri_embeddings.append(fmri_emb.numpy())
            target_images.append(img.numpy())
        
        self.fmri_embeddings = np.array(fmri_embeddings)
        self.target_images = np.array(target_images)
        
        print(f"✅ Test data loaded:")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Target images: {self.target_images.shape}")
        
        return self.fmri_embeddings, self.target_images
    
    def evaluate_baseline_regression(self):
        """Evaluate baseline regression model"""
        print(f"\n🎯 EVALUATING BASELINE REGRESSION MODEL")
        print("=" * 50)
        
        # Load model
        model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(self.device)
        model.load_state_dict(torch.load('baseline_model_best.pth', map_location=self.device))
        model.eval()
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(len(self.fmri_embeddings)), desc="Generating"):
                fmri_emb = torch.FloatTensor(self.fmri_embeddings[i:i+1]).to(self.device)
                pred = model(fmri_emb)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, self.target_images, "Baseline Regression")
        
        self.results['baseline'] = {
            'predictions': predictions,
            'metrics': metrics,
            'model_size': sum(p.numel() for p in model.parameters()),
            'inference_time': self.measure_inference_time(model, 'baseline')
        }
        
        return metrics
    
    def evaluate_enhanced_ldm(self):
        """Evaluate enhanced LDM"""
        print(f"\n🚀 EVALUATING ENHANCED LDM")
        print("=" * 40)
        
        # Load model
        unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=64,
            num_res_blocks=2
        )
        
        model = EnhancedDiffusionModel(unet, num_timesteps=1000).to(self.device)
        
        # Load best weights
        checkpoint = torch.load('enhanced_ldm_best.pth', map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Move noise schedule tensors to device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(model, attr_name)
            setattr(model, attr_name, attr_value.to(self.device))
        
        # Generate predictions
        predictions = []
        
        print("🔄 Generating samples (this may take a while)...")
        
        with torch.no_grad():
            for i in tqdm(range(len(self.fmri_embeddings)), desc="Sampling"):
                fmri_emb = torch.FloatTensor(self.fmri_embeddings[i:i+1]).to(self.device)
                
                # Generate sample
                pred = model.sample(fmri_emb, image_size=28)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, self.target_images, "Enhanced LDM")
        
        self.results['enhanced_ldm'] = {
            'predictions': predictions,
            'metrics': metrics,
            'model_size': sum(p.numel() for p in model.parameters()),
            'inference_time': self.measure_inference_time(model, 'enhanced_ldm')
        }
        
        return metrics
    
    def calculate_metrics(self, predictions, targets, model_name):
        """Calculate comprehensive metrics"""
        print(f"\n📊 CALCULATING METRICS FOR {model_name}")
        print("=" * 50)
        
        # Flatten for some metrics
        pred_flat = predictions.reshape(len(predictions), -1)
        target_flat = targets.reshape(len(targets), -1)
        
        # MSE
        mse = mean_squared_error(target_flat, pred_flat)
        
        # MAE
        mae = np.mean(np.abs(target_flat - pred_flat))
        
        # PSNR
        psnr_values = []
        for i in range(len(predictions)):
            mse_img = np.mean((predictions[i] - targets[i]) ** 2)
            if mse_img > 0:
                psnr = 20 * np.log10(2.0 / np.sqrt(mse_img))  # Range is [-1, 1]
                psnr_values.append(psnr)
        
        psnr_mean = np.mean(psnr_values) if psnr_values else 0
        psnr_std = np.std(psnr_values) if psnr_values else 0
        
        # SSIM
        ssim_values = []
        for i in range(len(predictions)):
            pred_img = predictions[i, 0]  # Remove channel dimension
            target_img = targets[i, 0]
            
            # SSIM expects values in [0, 1] or [-1, 1]
            ssim_val = ssim(target_img, pred_img, data_range=2.0)  # data_range = max - min = 1 - (-1) = 2
            ssim_values.append(ssim_val)
        
        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values)
        
        # Pixel correlation
        correlations = []
        for i in range(len(predictions)):
            pred_flat_img = predictions[i].flatten()
            target_flat_img = targets[i].flatten()
            
            corr, _ = pearsonr(pred_flat_img, target_flat_img)
            if not np.isnan(corr):
                correlations.append(corr)
        
        corr_mean = np.mean(correlations) if correlations else 0
        corr_std = np.std(correlations) if correlations else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'psnr_mean': psnr_mean,
            'psnr_std': psnr_std,
            'ssim_mean': ssim_mean,
            'ssim_std': ssim_std,
            'correlation_mean': corr_mean,
            'correlation_std': corr_std,
            'correlations': correlations,
            'ssim_values': ssim_values,
            'psnr_values': psnr_values
        }
        
        print(f"📊 {model_name} Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
        print(f"   SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
        print(f"   Correlation: {corr_mean:.4f} ± {corr_std:.4f}")
        
        return metrics
    
    def measure_inference_time(self, model, model_type):
        """Measure inference time"""
        print(f"\n⏱️ MEASURING INFERENCE TIME")
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for i in range(5):  # 5 runs for average
                fmri_emb = torch.FloatTensor(self.fmri_embeddings[0:1]).to(self.device)
                
                start_time = time.time()
                
                if model_type == 'baseline':
                    _ = model(fmri_emb)
                elif model_type == 'enhanced_ldm':
                    _ = model.sample(fmri_emb, image_size=28)
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   Average inference time: {avg_time:.3f} ± {std_time:.3f} seconds")
        
        return {'mean': avg_time, 'std': std_time}
    
    def create_comparison_visualization(self):
        """Create comprehensive comparison visualization"""
        print(f"\n🎨 CREATING COMPARISON VISUALIZATION")
        print("=" * 45)
        
        # Create figure
        fig, axes = plt.subplots(4, 6, figsize=(20, 16))
        fig.suptitle('Comprehensive Model Comparison', fontsize=20)
        
        # Show 4 test samples
        for i in range(4):
            # Original
            orig_img = self.target_images[i, 0]
            axes[i, 0].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')
            
            # Baseline prediction
            baseline_img = self.results['baseline']['predictions'][i, 0]
            baseline_corr = self.results['baseline']['metrics']['correlations'][i]
            axes[i, 1].imshow(baseline_img, cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'Baseline\nCorr: {baseline_corr:.3f}')
            axes[i, 1].axis('off')
            
            # Enhanced LDM prediction
            ldm_img = self.results['enhanced_ldm']['predictions'][i, 0]
            ldm_corr = self.results['enhanced_ldm']['metrics']['correlations'][i]
            axes[i, 2].imshow(ldm_img, cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Enhanced LDM\nCorr: {ldm_corr:.3f}')
            axes[i, 2].axis('off')
            
            # Baseline difference
            baseline_diff = np.abs(orig_img - baseline_img)
            axes[i, 3].imshow(baseline_diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Baseline |Diff|')
            axes[i, 3].axis('off')
            
            # LDM difference
            ldm_diff = np.abs(orig_img - ldm_img)
            axes[i, 4].imshow(ldm_diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 4].set_title(f'LDM |Diff|')
            axes[i, 4].axis('off')
            
            # SSIM comparison
            baseline_ssim = self.results['baseline']['metrics']['ssim_values'][i]
            ldm_ssim = self.results['enhanced_ldm']['metrics']['ssim_values'][i]
            
            axes[i, 5].bar(['Baseline', 'Enhanced LDM'], [baseline_ssim, ldm_ssim], 
                          color=['blue', 'red'], alpha=0.7)
            axes[i, 5].set_title(f'SSIM Comparison')
            axes[i, 5].set_ylabel('SSIM')
            axes[i, 5].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_metrics_comparison(self):
        """Create metrics comparison chart"""
        print(f"\n📊 CREATING METRICS COMPARISON")
        print("=" * 40)
        
        # Prepare data
        models = ['Baseline Regression', 'Enhanced LDM']
        
        mse_values = [
            self.results['baseline']['metrics']['mse'],
            self.results['enhanced_ldm']['metrics']['mse']
        ]
        
        ssim_values = [
            self.results['baseline']['metrics']['ssim_mean'],
            self.results['enhanced_ldm']['metrics']['ssim_mean']
        ]
        
        corr_values = [
            self.results['baseline']['metrics']['correlation_mean'],
            self.results['enhanced_ldm']['metrics']['correlation_mean']
        ]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantitative Metrics Comparison', fontsize=16)
        
        # MSE (lower is better)
        axes[0, 0].bar(models, mse_values, color=['blue', 'red'], alpha=0.7)
        axes[0, 0].set_title('Mean Squared Error (Lower = Better)')
        axes[0, 0].set_ylabel('MSE')
        for i, v in enumerate(mse_values):
            axes[0, 0].text(i, v + max(mse_values)*0.01, f'{v:.4f}', ha='center')
        
        # SSIM (higher is better)
        axes[0, 1].bar(models, ssim_values, color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_title('Structural Similarity (Higher = Better)')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(ssim_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # Correlation (higher is better)
        axes[0, 2].bar(models, corr_values, color=['blue', 'red'], alpha=0.7)
        axes[0, 2].set_title('Pixel Correlation (Higher = Better)')
        axes[0, 2].set_ylabel('Correlation')
        axes[0, 2].set_ylim(0, 1)
        for i, v in enumerate(corr_values):
            axes[0, 2].text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        # Model size comparison
        model_sizes = [
            self.results['baseline']['model_size'] / 1e6,  # Convert to millions
            self.results['enhanced_ldm']['model_size'] / 1e6
        ]
        
        axes[1, 0].bar(models, model_sizes, color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_title('Model Size (Parameters)')
        axes[1, 0].set_ylabel('Parameters (Millions)')
        for i, v in enumerate(model_sizes):
            axes[1, 0].text(i, v + max(model_sizes)*0.01, f'{v:.1f}M', ha='center')
        
        # Inference time comparison
        inference_times = [
            self.results['baseline']['inference_time']['mean'],
            self.results['enhanced_ldm']['inference_time']['mean']
        ]
        
        axes[1, 1].bar(models, inference_times, color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_title('Inference Time (Seconds)')
        axes[1, 1].set_ylabel('Time (s)')
        for i, v in enumerate(inference_times):
            axes[1, 1].text(i, v + max(inference_times)*0.01, f'{v:.3f}s', ha='center')
        
        # Distribution comparison
        baseline_corrs = self.results['baseline']['metrics']['correlations']
        ldm_corrs = self.results['enhanced_ldm']['metrics']['correlations']
        
        axes[1, 2].hist([baseline_corrs, ldm_corrs], bins=10, alpha=0.7, 
                       label=['Baseline', 'Enhanced LDM'], color=['blue', 'red'])
        axes[1, 2].set_title('Correlation Distribution')
        axes[1, 2].set_xlabel('Correlation')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print(f"\n📋 GENERATING FINAL REPORT")
        print("=" * 40)
        
        report = {
            'evaluation_summary': {
                'test_samples': len(self.fmri_embeddings),
                'image_size': self.target_images.shape[-2:],
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_results': self.results
        }
        
        # Save report
        with open('comprehensive_evaluation_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        # Print summary
        print(f"\n🎯 FINAL EVALUATION SUMMARY")
        print("=" * 50)
        
        print(f"\n📊 BASELINE REGRESSION:")
        baseline = self.results['baseline']['metrics']
        print(f"   MSE: {baseline['mse']:.4f}")
        print(f"   SSIM: {baseline['ssim_mean']:.4f} ± {baseline['ssim_std']:.4f}")
        print(f"   Correlation: {baseline['correlation_mean']:.4f} ± {baseline['correlation_std']:.4f}")
        print(f"   Model size: {self.results['baseline']['model_size']/1e6:.1f}M parameters")
        print(f"   Inference time: {self.results['baseline']['inference_time']['mean']:.3f}s")
        
        print(f"\n🚀 ENHANCED LDM:")
        ldm = self.results['enhanced_ldm']['metrics']
        print(f"   MSE: {ldm['mse']:.4f}")
        print(f"   SSIM: {ldm['ssim_mean']:.4f} ± {ldm['ssim_std']:.4f}")
        print(f"   Correlation: {ldm['correlation_mean']:.4f} ± {ldm['correlation_std']:.4f}")
        print(f"   Model size: {self.results['enhanced_ldm']['model_size']/1e6:.1f}M parameters")
        print(f"   Inference time: {self.results['enhanced_ldm']['inference_time']['mean']:.3f}s")
        
        # Winner analysis
        print(f"\n🏆 WINNER ANALYSIS:")
        
        if baseline['correlation_mean'] > ldm['correlation_mean']:
            print(f"   🥇 CORRELATION: Baseline ({baseline['correlation_mean']:.4f} vs {ldm['correlation_mean']:.4f})")
        else:
            print(f"   🥇 CORRELATION: Enhanced LDM ({ldm['correlation_mean']:.4f} vs {baseline['correlation_mean']:.4f})")
        
        if baseline['ssim_mean'] > ldm['ssim_mean']:
            print(f"   🥇 SSIM: Baseline ({baseline['ssim_mean']:.4f} vs {ldm['ssim_mean']:.4f})")
        else:
            print(f"   🥇 SSIM: Enhanced LDM ({ldm['ssim_mean']:.4f} vs {baseline['ssim_mean']:.4f})")
        
        if baseline['mse'] < ldm['mse']:
            print(f"   🥇 MSE: Baseline ({baseline['mse']:.4f} vs {ldm['mse']:.4f})")
        else:
            print(f"   🥇 MSE: Enhanced LDM ({ldm['mse']:.4f} vs {baseline['mse']:.4f})")
        
        print(f"\n📁 Generated files:")
        print(f"   - comprehensive_model_comparison.png")
        print(f"   - metrics_comparison.png")
        print(f"   - comprehensive_evaluation_report.pkl")
        
        return report

def main():
    """Main evaluation function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluator = ComprehensiveEvaluator(device)
    
    # Load test data
    evaluator.load_test_data()
    
    # Evaluate all models
    evaluator.evaluate_baseline_regression()
    evaluator.evaluate_enhanced_ldm()
    
    # Create visualizations
    evaluator.create_comparison_visualization()
    evaluator.create_metrics_comparison()
    
    # Generate final report
    report = evaluator.generate_final_report()
    
    print(f"\n✅ COMPREHENSIVE EVALUATION COMPLETED!")

if __name__ == "__main__":
    main()
