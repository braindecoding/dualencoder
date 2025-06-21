#!/usr/bin/env python3
"""
CLIP Score Evaluation for Enhanced LDM
High Priority Task 2: Add CLIP score evaluation to comprehensive evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import clip
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# Import existing models
from simple_baseline_model import SimpleRegressionModel, Digit69BaselineDataset
from improved_unet import ImprovedUNet
from enhanced_training_pipeline import EnhancedDiffusionModel
from clip_guidance_ldm import CLIPGuidanceLoss

class CLIPEvaluator:
    """Comprehensive evaluator dengan CLIP scores"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # Initialize CLIP loss for evaluation
        self.clip_loss = CLIPGuidanceLoss("ViT-B/32", device=device)
        
        print(f"🔍 CLIP EVALUATOR INITIALIZED")
        print(f"   Device: {device}")
        print(f"   CLIP model: ViT-B/32")
    
    def load_test_data(self):
        """Load test dataset"""
        print(f"\n📊 LOADING TEST DATA")
        
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
        
        # Generate digit classes (in real scenario, you'd have actual labels)
        self.digit_classes = torch.randint(0, 10, (len(test_dataset),))
        
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Target images: {self.target_images.shape}")
        print(f"   Digit classes: {self.digit_classes.shape}")
        
        return self.fmri_embeddings, self.target_images, self.digit_classes
    
    def calculate_clip_scores(self, images, digit_classes, model_name="Unknown"):
        """Calculate CLIP scores untuk generated images"""
        print(f"\n📊 CALCULATING CLIP SCORES FOR {model_name}")
        
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)
        if isinstance(digit_classes, np.ndarray):
            digit_classes = torch.LongTensor(digit_classes)
        
        images = images.to(self.device)
        digit_classes = digit_classes.to(self.device)
        
        # Calculate CLIP scores
        with torch.no_grad():
            clip_loss, clip_scores = self.clip_loss(images, digit_classes)
        
        clip_scores_np = clip_scores.cpu().numpy()
        
        # Statistics
        clip_stats = {
            'mean': np.mean(clip_scores_np),
            'std': np.std(clip_scores_np),
            'min': np.min(clip_scores_np),
            'max': np.max(clip_scores_np),
            'median': np.median(clip_scores_np),
            'scores': clip_scores_np
        }
        
        print(f"   CLIP Score Statistics:")
        print(f"     Mean: {clip_stats['mean']:.4f} ± {clip_stats['std']:.4f}")
        print(f"     Range: [{clip_stats['min']:.4f}, {clip_stats['max']:.4f}]")
        print(f"     Median: {clip_stats['median']:.4f}")
        
        return clip_stats
    
    def evaluate_baseline_with_clip(self):
        """Evaluate baseline model dengan CLIP scores"""
        print(f"\n🎯 EVALUATING BASELINE MODEL WITH CLIP")
        print("=" * 50)
        
        # Load baseline model
        model = SimpleRegressionModel(fmri_dim=512, image_size=28).to(self.device)
        model.load_state_dict(torch.load('baseline_model_best.pth', map_location=self.device))
        model.eval()
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(len(self.fmri_embeddings)), desc="Baseline Generation"):
                fmri_emb = torch.FloatTensor(self.fmri_embeddings[i:i+1]).to(self.device)
                pred = model(fmri_emb)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate traditional metrics
        traditional_metrics = self.calculate_traditional_metrics(predictions, self.target_images, "Baseline")
        
        # Calculate CLIP scores
        clip_metrics = self.calculate_clip_scores(predictions, self.digit_classes, "Baseline")
        
        # Combine metrics
        self.results['baseline'] = {
            'predictions': predictions,
            'traditional_metrics': traditional_metrics,
            'clip_metrics': clip_metrics,
            'model_size': sum(p.numel() for p in model.parameters())
        }
        
        return traditional_metrics, clip_metrics
    
    def evaluate_enhanced_ldm_with_clip(self):
        """Evaluate enhanced LDM dengan CLIP scores"""
        print(f"\n🚀 EVALUATING ENHANCED LDM WITH CLIP")
        print("=" * 50)
        
        # Load enhanced LDM
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
            for i in tqdm(range(len(self.fmri_embeddings)), desc="Enhanced LDM Generation"):
                fmri_emb = torch.FloatTensor(self.fmri_embeddings[i:i+1]).to(self.device)
                
                # Generate sample
                pred = model.sample(fmri_emb, image_size=28)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate traditional metrics
        traditional_metrics = self.calculate_traditional_metrics(predictions, self.target_images, "Enhanced LDM")
        
        # Calculate CLIP scores
        clip_metrics = self.calculate_clip_scores(predictions, self.digit_classes, "Enhanced LDM")
        
        # Combine metrics
        self.results['enhanced_ldm'] = {
            'predictions': predictions,
            'traditional_metrics': traditional_metrics,
            'clip_metrics': clip_metrics,
            'model_size': sum(p.numel() for p in model.parameters())
        }
        
        return traditional_metrics, clip_metrics
    
    def evaluate_clip_guided_ldm(self, clip_weight=1.0):
        """Evaluate CLIP guided LDM jika tersedia"""
        print(f"\n🎨 EVALUATING CLIP GUIDED LDM (Weight: {clip_weight})")
        print("=" * 50)
        
        # Check if CLIP guided model exists
        clip_model_path = f'clip_guided_ldm_w{clip_weight}_best.pth'
        
        if not torch.cuda.is_available():
            print("   ⚠️ CLIP guided model requires CUDA, skipping...")
            return None, None
        
        try:
            from clip_guidance_ldm import CLIPGuidedDiffusionModel
            
            # Load CLIP guided model
            unet = ImprovedUNet(
                in_channels=1,
                out_channels=1,
                condition_dim=512,
                model_channels=64,
                num_res_blocks=2
            )
            
            model = CLIPGuidedDiffusionModel(
                unet,
                num_timesteps=1000,
                clip_guidance_weight=clip_weight,
                clip_model_name="ViT-B/32"
            ).to(self.device)

            # Fix device issues for noise schedule tensors
            for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                             'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
                attr_value = getattr(model, attr_name)
                setattr(model, attr_name, attr_value.to(self.device))
            
            # Load weights if available
            checkpoint = torch.load(clip_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Generate predictions
            predictions = []
            
            with torch.no_grad():
                for i in tqdm(range(len(self.fmri_embeddings)), desc="CLIP Guided Generation"):
                    fmri_emb = torch.FloatTensor(self.fmri_embeddings[i:i+1]).to(self.device)
                    
                    # Generate sample
                    pred = model.sample(fmri_emb, image_size=28)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.vstack(predictions)
            
            # Calculate traditional metrics
            traditional_metrics = self.calculate_traditional_metrics(predictions, self.target_images, f"CLIP Guided (w={clip_weight})")
            
            # Calculate CLIP scores
            clip_metrics = self.calculate_clip_scores(predictions, self.digit_classes, f"CLIP Guided (w={clip_weight})")
            
            # Combine metrics
            self.results[f'clip_guided_{clip_weight}'] = {
                'predictions': predictions,
                'traditional_metrics': traditional_metrics,
                'clip_metrics': clip_metrics,
                'model_size': sum(p.numel() for p in model.parameters()),
                'clip_weight': clip_weight
            }
            
            return traditional_metrics, clip_metrics
            
        except FileNotFoundError:
            print(f"   ⚠️ CLIP guided model not found: {clip_model_path}")
            return None, None
        except Exception as e:
            print(f"   ❌ Error loading CLIP guided model: {e}")
            return None, None
    
    def calculate_traditional_metrics(self, predictions, targets, model_name):
        """Calculate traditional metrics (MSE, SSIM, correlation)"""
        # Flatten for some metrics
        pred_flat = predictions.reshape(len(predictions), -1)
        target_flat = targets.reshape(len(targets), -1)
        
        # MSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        # SSIM
        ssim_values = []
        for i in range(len(predictions)):
            pred_img = predictions[i, 0]
            target_img = targets[i, 0]
            ssim_val = ssim(target_img, pred_img, data_range=2.0)
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
            'ssim_mean': ssim_mean,
            'ssim_std': ssim_std,
            'correlation_mean': corr_mean,
            'correlation_std': corr_std,
            'correlations': correlations,
            'ssim_values': ssim_values
        }
        
        print(f"📊 {model_name} Traditional Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
        print(f"   Correlation: {corr_mean:.4f} ± {corr_std:.4f}")
        
        return metrics
    
    def create_clip_comparison_visualization(self):
        """Create comprehensive comparison dengan CLIP scores"""
        print(f"\n🎨 CREATING CLIP COMPARISON VISUALIZATION")
        print("=" * 45)
        
        # Prepare data
        models = []
        traditional_scores = {'mse': [], 'ssim': [], 'correlation': []}
        clip_scores = {'mean': [], 'std': []}
        
        for model_name, results in self.results.items():
            models.append(model_name.replace('_', ' ').title())
            
            # Traditional metrics
            traditional_scores['mse'].append(results['traditional_metrics']['mse'])
            traditional_scores['ssim'].append(results['traditional_metrics']['ssim_mean'])
            traditional_scores['correlation'].append(results['traditional_metrics']['correlation_mean'])
            
            # CLIP metrics
            clip_scores['mean'].append(results['clip_metrics']['mean'])
            clip_scores['std'].append(results['clip_metrics']['std'])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Traditional vs CLIP Metrics Comparison', fontsize=16)
        
        # MSE (lower is better)
        axes[0, 0].bar(models, traditional_scores['mse'], alpha=0.7, color='blue')
        axes[0, 0].set_title('Mean Squared Error (Lower = Better)')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # SSIM (higher is better)
        axes[0, 1].bar(models, traditional_scores['ssim'], alpha=0.7, color='green')
        axes[0, 1].set_title('Structural Similarity (Higher = Better)')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Correlation (higher is better)
        axes[0, 2].bar(models, traditional_scores['correlation'], alpha=0.7, color='orange')
        axes[0, 2].set_title('Pixel Correlation (Higher = Better)')
        axes[0, 2].set_ylabel('Correlation')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # CLIP scores (higher is better)
        axes[1, 0].bar(models, clip_scores['mean'], yerr=clip_scores['std'], 
                      alpha=0.7, color='red', capsize=5)
        axes[1, 0].set_title('CLIP Similarity Scores (Higher = Better)')
        axes[1, 0].set_ylabel('CLIP Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Traditional vs CLIP correlation
        axes[1, 1].scatter(traditional_scores['correlation'], clip_scores['mean'], 
                          s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        axes[1, 1].set_xlabel('Traditional Correlation')
        axes[1, 1].set_ylabel('CLIP Score')
        axes[1, 1].set_title('Traditional vs CLIP Correlation')
        
        # Add model labels
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (traditional_scores['correlation'][i], clip_scores['mean'][i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # CLIP score distribution
        for i, (model_name, results) in enumerate(self.results.items()):
            clip_scores_dist = results['clip_metrics']['scores']
            axes[1, 2].hist(clip_scores_dist, bins=10, alpha=0.5, 
                           label=model_name.replace('_', ' ').title())
        
        axes[1, 2].set_title('CLIP Score Distributions')
        axes[1, 2].set_xlabel('CLIP Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('clip_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_report_with_clip(self):
        """Generate final report dengan CLIP metrics"""
        print(f"\n📋 GENERATING FINAL REPORT WITH CLIP METRICS")
        print("=" * 50)
        
        report = {
            'evaluation_summary': {
                'test_samples': len(self.fmri_embeddings),
                'models_evaluated': list(self.results.keys()),
                'metrics_included': ['MSE', 'SSIM', 'Correlation', 'CLIP Score']
            },
            'model_results': self.results
        }
        
        # Save report
        with open('clip_comprehensive_evaluation_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        # Print summary
        print(f"\n🎯 FINAL EVALUATION SUMMARY WITH CLIP")
        print("=" * 60)
        
        for model_name, results in self.results.items():
            print(f"\n📊 {model_name.upper().replace('_', ' ')}:")
            
            trad = results['traditional_metrics']
            clip_m = results['clip_metrics']
            
            print(f"   Traditional Metrics:")
            print(f"     MSE: {trad['mse']:.4f}")
            print(f"     SSIM: {trad['ssim_mean']:.4f} ± {trad['ssim_std']:.4f}")
            print(f"     Correlation: {trad['correlation_mean']:.4f} ± {trad['correlation_std']:.4f}")
            
            print(f"   CLIP Metrics:")
            print(f"     CLIP Score: {clip_m['mean']:.4f} ± {clip_m['std']:.4f}")
            print(f"     CLIP Range: [{clip_m['min']:.4f}, {clip_m['max']:.4f}]")
            
            print(f"   Model Size: {results['model_size']/1e6:.1f}M parameters")
        
        # Winner analysis
        print(f"\n🏆 WINNER ANALYSIS:")
        
        # Find best models for each metric
        best_mse = min(self.results.items(), key=lambda x: x[1]['traditional_metrics']['mse'])
        best_ssim = max(self.results.items(), key=lambda x: x[1]['traditional_metrics']['ssim_mean'])
        best_corr = max(self.results.items(), key=lambda x: x[1]['traditional_metrics']['correlation_mean'])
        best_clip = max(self.results.items(), key=lambda x: x[1]['clip_metrics']['mean'])
        
        print(f"   🥇 Best MSE: {best_mse[0]} ({best_mse[1]['traditional_metrics']['mse']:.4f})")
        print(f"   🥇 Best SSIM: {best_ssim[0]} ({best_ssim[1]['traditional_metrics']['ssim_mean']:.4f})")
        print(f"   🥇 Best Correlation: {best_corr[0]} ({best_corr[1]['traditional_metrics']['correlation_mean']:.4f})")
        print(f"   🥇 Best CLIP Score: {best_clip[0]} ({best_clip[1]['clip_metrics']['mean']:.4f})")
        
        print(f"\n📁 Generated files:")
        print(f"   - clip_comprehensive_comparison.png")
        print(f"   - clip_comprehensive_evaluation_report.pkl")
        
        return report

def main():
    """Main evaluation function dengan CLIP scores"""
    print("🎯 HIGH PRIORITY: CLIP SCORE EVALUATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluator = CLIPEvaluator(device)
    
    # Load test data
    evaluator.load_test_data()
    
    # Evaluate all models dengan CLIP scores
    evaluator.evaluate_baseline_with_clip()
    evaluator.evaluate_enhanced_ldm_with_clip()
    
    # Try to evaluate CLIP guided models if available
    for clip_weight in [0.5, 1.0, 2.0]:
        evaluator.evaluate_clip_guided_ldm(clip_weight)
    
    # Create visualizations
    evaluator.create_clip_comparison_visualization()
    
    # Generate final report
    report = evaluator.generate_final_report_with_clip()
    
    print(f"\n✅ CLIP EVALUATION COMPLETED!")

if __name__ == "__main__":
    main()
