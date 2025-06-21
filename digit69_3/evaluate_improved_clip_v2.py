#!/usr/bin/env python3
"""
Comprehensive Evaluation: Improved CLIP v2.0 vs Enhanced LDM vs Baseline
Test both pure diffusion and CLIP-guided sampling
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import clip
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from simple_baseline_model import SimpleRegressionModel
from improved_unet import ImprovedUNet
from enhanced_training_pipeline import EnhancedDiffusionModel
from improved_clip_v2 import InferenceTimeCLIPGuidedModel, create_improved_clip_v2_model

class ImprovedCLIPv2Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Load test data
        self.load_test_data()
        
    def load_test_data(self):
        """Load test data for evaluation"""
        print("📂 Loading test data...")
        
        with open('digit69_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Use test split
        test_data = data['test']
        self.fmri_embeddings = torch.FloatTensor(test_data['fmri_embeddings']).to(self.device)
        
        # Convert RGB to grayscale and resize
        original_images = torch.FloatTensor(test_data['original_images'])
        images_gray = original_images.mean(dim=1, keepdim=True)  # (10, 1, 224, 224)
        self.target_images = F.interpolate(images_gray, size=(28, 28), mode='bilinear', align_corners=False)
        self.target_images = (self.target_images - 0.5) * 2.0  # Normalize to [-1, 1]
        self.target_images = self.target_images.to(self.device)
        
        # Create labels and text prompts
        self.labels = torch.arange(10).to(self.device)
        self.text_prompts = [f"a handwritten digit {i}" for i in range(10)]
        
        print(f"   Test fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Test images: {self.target_images.shape}")
        print(f"   Test labels: {self.labels.shape}")
    
    def load_baseline_model(self):
        """Load baseline model"""
        print("🔄 Loading Baseline Model...")
        
        model = SimpleRegressionModel(
            fmri_dim=512,
            image_size=28,
            hidden_dims=[1024, 2048, 1024]
        ).to(self.device)
        
        model.load_state_dict(torch.load('baseline_model_best.pth', map_location=self.device))
        model.eval()
        
        return model
    
    def load_enhanced_ldm(self):
        """Load Enhanced LDM model"""
        print("🔄 Loading Enhanced LDM...")
        
        unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=64,
            num_res_blocks=2
        )
        
        model = EnhancedDiffusionModel(
            unet=unet,
            num_timesteps=1000
        ).to(self.device)
        
        # Load checkpoint and extract model state
        checkpoint = torch.load('enhanced_ldm_best.pth', map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Extract only UNet parameters and remove prefix
        unet_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                new_key = key[5:]  # Remove 'unet.' prefix
                unet_state_dict[new_key] = value
        
        model.unet.load_state_dict(unet_state_dict)
        
        # Fix device issues for noise schedule tensors
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(model, attr_name)
            setattr(model, attr_name, attr_value.to(self.device))
        
        model.eval()
        return model
    
    def load_improved_clip_v2(self):
        """Load Improved CLIP v2.0 model"""
        print("🔄 Loading Improved CLIP v2.0...")
        
        model = create_improved_clip_v2_model().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load('improved_clip_v2_best.pth', map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def compute_clip_score(self, generated_images, labels):
        """Compute CLIP score between generated images and text descriptions"""
        # Prepare images for CLIP
        images = (generated_images + 1.0) / 2.0  # Convert to [0, 1]
        images = images.repeat(1, 3, 1, 1)  # Convert to RGB
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # Get CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
        
        # Get text features
        text_descriptions = [f"a handwritten digit {label.item()}" for label in labels]
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        similarity = torch.sum(image_features * text_features, dim=-1)
        return similarity.mean().item(), similarity.std().item()
    
    def evaluate_model(self, model, model_name, model_type='baseline', test_clip_guidance=False):
        """Evaluate a single model"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        generated_images = []
        
        with torch.no_grad():
            for i in range(len(self.fmri_embeddings)):
                fmri_emb = self.fmri_embeddings[i:i+1]
                
                if model_type == 'baseline':
                    # Baseline model
                    output = model(fmri_emb)
                    generated = output.view(1, 1, 28, 28)
                    generated = torch.tanh(generated)  # Ensure [-1, 1] range
                    
                elif model_type == 'enhanced_ldm':
                    # Enhanced LDM
                    generated = model.sample(fmri_emb, image_size=28)
                    
                elif model_type == 'improved_clip_v2':
                    # Improved CLIP v2.0
                    if test_clip_guidance:
                        # Test with CLIP guidance
                        text_prompt = [self.text_prompts[i]]
                        generated = model.sample(
                            fmri_emb, 
                            text_prompts=text_prompt,
                            num_samples=1, 
                            num_timesteps=200,  # Faster sampling for evaluation
                            clip_guidance_scale=0.01,
                            clip_start_step=100
                        )
                    else:
                        # Test pure diffusion (no CLIP guidance)
                        generated = model.sample(fmri_emb, num_samples=1, num_timesteps=200)
                
                generated_images.append(generated)
        
        # Concatenate all generated images
        generated_images = torch.cat(generated_images, dim=0)
        
        # Compute metrics
        target_np = self.target_images.cpu().numpy()
        generated_np = generated_images.cpu().numpy()
        
        # MSE
        mse = mean_squared_error(target_np.flatten(), generated_np.flatten())
        
        # SSIM (per image, then average)
        ssim_scores = []
        for i in range(len(target_np)):
            target_img = target_np[i, 0]  # Remove channel dimension
            generated_img = generated_np[i, 0]
            
            # Convert to [0, 1] for SSIM
            target_img = (target_img + 1.0) / 2.0
            generated_img = (generated_img + 1.0) / 2.0
            
            ssim_score = ssim(target_img, generated_img, data_range=1.0)
            ssim_scores.append(ssim_score)
        
        ssim_mean = np.mean(ssim_scores)
        ssim_std = np.std(ssim_scores)
        
        # Correlation
        correlations = []
        for i in range(len(target_np)):
            target_flat = target_np[i].flatten()
            generated_flat = generated_np[i].flatten()
            corr = np.corrcoef(target_flat, generated_flat)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        corr_mean = np.mean(correlations)
        corr_std = np.std(correlations)
        
        # CLIP Score
        clip_mean, clip_std = self.compute_clip_score(generated_images, self.labels)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'ssim_mean': ssim_mean,
            'ssim_std': ssim_std,
            'correlation_mean': corr_mean,
            'correlation_std': corr_std,
            'clip_score_mean': clip_mean,
            'clip_score_std': clip_std,
            'total_params': total_params,
            'generated_images': generated_images.cpu(),
            'target_images': self.target_images.cpu()
        }
        
        print(f"   MSE: {mse:.4f}")
        print(f"   SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
        print(f"   Correlation: {corr_mean:.4f} ± {corr_std:.4f}")
        print(f"   CLIP Score: {clip_mean:.4f} ± {clip_std:.4f}")
        print(f"   Parameters: {total_params:,}")
        
        return results

def run_improved_clip_v2_evaluation():
    """Run comprehensive evaluation of Improved CLIP v2.0"""
    print("🚀 IMPROVED CLIP v2.0 COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    evaluator = ImprovedCLIPv2Evaluator()
    all_results = {}
    
    # 1. Evaluate Baseline Model
    try:
        baseline_model = evaluator.load_baseline_model()
        results = evaluator.evaluate_model(baseline_model, "Baseline", "baseline")
        all_results['Baseline'] = results
        del baseline_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error evaluating Baseline: {e}")
    
    # 2. Evaluate Enhanced LDM
    try:
        enhanced_model = evaluator.load_enhanced_ldm()
        results = evaluator.evaluate_model(enhanced_model, "Enhanced LDM", "enhanced_ldm")
        all_results['Enhanced LDM'] = results
        del enhanced_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error evaluating Enhanced LDM: {e}")
    
    # 3. Evaluate Improved CLIP v2.0 (Pure Diffusion)
    try:
        clip_v2_model = evaluator.load_improved_clip_v2()
        results = evaluator.evaluate_model(clip_v2_model, "Improved CLIP v2.0 (Pure)", "improved_clip_v2", test_clip_guidance=False)
        all_results['Improved CLIP v2.0 (Pure)'] = results
    except Exception as e:
        print(f"❌ Error evaluating Improved CLIP v2.0 (Pure): {e}")
    
    # 4. Evaluate Improved CLIP v2.0 (With CLIP Guidance)
    try:
        results = evaluator.evaluate_model(clip_v2_model, "Improved CLIP v2.0 (CLIP)", "improved_clip_v2", test_clip_guidance=True)
        all_results['Improved CLIP v2.0 (CLIP)'] = results
        del clip_v2_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error evaluating Improved CLIP v2.0 (CLIP): {e}")
    
    # Save results
    with open('improved_clip_v2_evaluation_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n📁 Results saved to: improved_clip_v2_evaluation_results.pkl")
    return all_results

if __name__ == "__main__":
    results = run_improved_clip_v2_evaluation()
    
    print("\n📊 FINAL COMPARISON SUMMARY:")
    print("=" * 90)
    print(f"{'Model':<30} {'MSE':<8} {'SSIM':<12} {'Correlation':<12} {'CLIP Score':<12} {'Params':<10}")
    print("-" * 90)
    
    for model_name, result in results.items():
        print(f"{model_name:<30} "
              f"{result['mse']:<8.4f} "
              f"{result['ssim_mean']:<12.4f} "
              f"{result['correlation_mean']:<12.4f} "
              f"{result['clip_score_mean']:<12.4f} "
              f"{result['total_params']/1e6:<10.1f}M")
