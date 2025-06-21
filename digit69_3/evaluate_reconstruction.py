#!/usr/bin/env python3
"""
Evaluate Digit69 LDM Reconstruction Quality
Comprehensive evaluation of fMRI-to-digit reconstruction
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import pearsonr
import seaborn as sns

# Import our models - copy definitions to avoid import issues
import sys
sys.path.append('.')

# Copy SimpleDiffusion class definition
class SimpleDiffusionModel(torch.nn.Module):
    """Simple diffusion model for proof of concept"""

    def __init__(self, condition_dim=512):
        super().__init__()

        # Simple encoder-decoder with conditioning
        self.condition_proj = torch.nn.Linear(condition_dim, 64)

        # Encoder
        self.enc1 = torch.nn.Conv2d(1, 32, 3, padding=1)  # 28x28
        self.enc2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 14x14
        self.enc3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 7x7

        # Middle with conditioning
        self.mid1 = torch.nn.Conv2d(128 + 64, 128, 3, padding=1)  # +64 from condition
        self.mid2 = torch.nn.Conv2d(128, 128, 3, padding=1)

        # Decoder
        self.dec3 = torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 14x14
        self.dec2 = torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 28x28
        self.dec1 = torch.nn.Conv2d(32, 1, 3, padding=1)  # 28x28

        # Time embedding (simplified)
        self.time_embed = torch.nn.Embedding(1000, 64)

    def forward(self, x, t, condition):
        # Time embedding
        time_emb = self.time_embed(t)  # (B, 64)

        # Condition projection
        cond_emb = self.condition_proj(condition)  # (B, 64)

        # Combine time and condition
        combined_emb = time_emb + cond_emb  # (B, 64)

        # Encoder
        h1 = F.relu(self.enc1(x))      # (B, 32, 28, 28)
        h2 = F.relu(self.enc2(h1))     # (B, 64, 14, 14)
        h3 = F.relu(self.enc3(h2))     # (B, 128, 7, 7)

        # Add conditioning to middle layer
        # Broadcast condition to spatial dimensions
        cond_spatial = combined_emb[:, :, None, None].expand(-1, -1, 7, 7)  # (B, 64, 7, 7)
        h3_cond = torch.cat([h3, cond_spatial], dim=1)  # (B, 192, 7, 7)

        # Middle
        h = F.relu(self.mid1(h3_cond))
        h = F.relu(self.mid2(h))

        # Decoder
        h = F.relu(self.dec3(h))       # (B, 64, 14, 14)
        h = F.relu(self.dec2(h))       # (B, 32, 28, 28)
        h = self.dec1(h)               # (B, 1, 28, 28)

        return h

# Use the same structure as training - inherit from SimpleDiffusionModel directly
class SimpleDiffusion(SimpleDiffusionModel):
    """Simple diffusion model for digit generation - same as training"""

    def __init__(self, image_size=28, condition_dim=512, num_timesteps=1000):
        super().__init__(condition_dim=condition_dim)

        self.image_size = image_size
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps

        # Noise schedule (linear) - these will be loaded from checkpoint
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def sample(self, condition, num_steps=50):
        """Generate images from condition"""
        device = condition.device
        batch_size = condition.shape[0]

        # Move noise schedule to device
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)

        # Start from noise
        x = torch.randn(batch_size, 1, self.image_size, self.image_size, device=device)

        # Sampling steps
        step_size = self.num_timesteps // num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), self.num_timesteps - 1 - i * step_size, device=device, dtype=torch.long)
            t = torch.clamp(t, 0, self.num_timesteps - 1)

            # Predict noise - use self directly since we inherit from SimpleDiffusionModel
            predicted_noise = self.forward(x, t, condition)

            # Remove noise (simplified DDPM step)
            alpha_t = alphas[t[0]]
            alpha_cumprod_t = alphas_cumprod[t[0]]
            beta_t = betas[t[0]]

            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

            # Add noise for next step (except last)
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise * 0.5  # Reduced noise

        return torch.clamp(x, -1, 1)

class Digit69Dataset:
    """Simple dataset class for loading embeddings"""

    def __init__(self, embeddings_file="digit69_embeddings.pkl", split="test"):
        self.split = split

        # Load embeddings
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)

        self.fmri_embeddings = data[split]['fmri_embeddings']
        self.original_images = data[split]['original_images']

        # Convert images to grayscale and normalize
        if len(self.original_images.shape) == 4 and self.original_images.shape[1] == 3:
            # Convert RGB to grayscale
            self.images = np.mean(self.original_images, axis=1, keepdims=True)
        else:
            self.images = self.original_images

        # Resize to 28x28 if needed
        if self.images.shape[-1] != 28:
            self.images = self._resize_images(self.images)

        # Normalize to [-1, 1]
        self.images = (self.images - 0.5) * 2

    def _resize_images(self, images):
        """Resize images to 28x28"""
        resized = []

        for img in images:
            if len(img.shape) == 3:
                img = img[0]  # Take first channel

            # Convert to PIL and resize
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.resize((28, 28))
            resized.append(np.array(pil_img) / 255.0)

        return np.array(resized)[:, None, :, :]  # Add channel dimension

    def __len__(self):
        return len(self.fmri_embeddings)

    def __getitem__(self, idx):
        fmri_emb = torch.FloatTensor(self.fmri_embeddings[idx])
        image = torch.FloatTensor(self.images[idx])

        return fmri_emb, image

class ReconstructionEvaluator:
    """Comprehensive evaluation of reconstruction quality"""
    
    def __init__(self, model_path="digit69_ldm_final.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = SimpleDiffusion(condition_dim=512).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")
        
        # Load dataset
        self.test_dataset = Digit69Dataset("digit69_embeddings.pkl", "test")
        print(f"✅ Test dataset loaded: {len(self.test_dataset)} samples")
    
    def generate_reconstructions(self, num_samples=None, num_steps=50):
        """Generate reconstructions for test samples"""
        if num_samples is None:
            num_samples = len(self.test_dataset)
        
        print(f"\n🎨 GENERATING RECONSTRUCTIONS")
        print(f"   Samples: {num_samples}")
        print(f"   Diffusion steps: {num_steps}")
        
        originals = []
        reconstructions = []
        fmri_embeddings = []
        
        with torch.no_grad():
            for i in range(num_samples):
                fmri_emb, original_img = self.test_dataset[i]
                
                # Move to device
                fmri_emb = fmri_emb.unsqueeze(0).to(self.device)
                
                # Generate reconstruction
                reconstructed = self.model.sample(fmri_emb, num_steps=num_steps)
                
                # Store results
                originals.append(original_img.numpy())
                reconstructions.append(reconstructed.cpu().numpy()[0])
                fmri_embeddings.append(fmri_emb.cpu().numpy()[0])
                
                if (i + 1) % 5 == 0:
                    print(f"   Generated {i+1}/{num_samples}")
        
        return np.array(originals), np.array(reconstructions), np.array(fmri_embeddings)
    
    def calculate_metrics(self, originals, reconstructions):
        """Calculate comprehensive reconstruction metrics"""
        print(f"\n📊 CALCULATING METRICS")
        
        metrics = {}
        
        # 1. Mean Squared Error (MSE)
        mse_values = []
        for orig, recon in zip(originals, reconstructions):
            mse = np.mean((orig - recon) ** 2)
            mse_values.append(mse)
        
        metrics['mse_mean'] = np.mean(mse_values)
        metrics['mse_std'] = np.std(mse_values)
        metrics['mse_values'] = mse_values
        
        # 2. Structural Similarity Index (SSIM)
        ssim_values = []
        for orig, recon in zip(originals, reconstructions):
            ssim = self.calculate_ssim(orig[0], recon[0])  # Take first channel
            ssim_values.append(ssim)
        
        metrics['ssim_mean'] = np.mean(ssim_values)
        metrics['ssim_std'] = np.std(ssim_values)
        metrics['ssim_values'] = ssim_values
        
        # 3. Peak Signal-to-Noise Ratio (PSNR)
        psnr_values = []
        for orig, recon in zip(originals, reconstructions):
            psnr = self.calculate_psnr(orig[0], recon[0])
            psnr_values.append(psnr)
        
        metrics['psnr_mean'] = np.mean(psnr_values)
        metrics['psnr_std'] = np.std(psnr_values)
        metrics['psnr_values'] = psnr_values
        
        # 4. Pixel-wise Correlation
        corr_values = []
        for orig, recon in zip(originals, reconstructions):
            orig_flat = orig.flatten()
            recon_flat = recon.flatten()
            corr, _ = pearsonr(orig_flat, recon_flat)
            corr_values.append(corr if not np.isnan(corr) else 0)
        
        metrics['correlation_mean'] = np.mean(corr_values)
        metrics['correlation_std'] = np.std(corr_values)
        metrics['correlation_values'] = corr_values
        
        # 5. L1 Distance (MAE)
        mae_values = []
        for orig, recon in zip(originals, reconstructions):
            mae = np.mean(np.abs(orig - recon))
            mae_values.append(mae)
        
        metrics['mae_mean'] = np.mean(mae_values)
        metrics['mae_std'] = np.std(mae_values)
        metrics['mae_values'] = mae_values
        
        print(f"✅ Metrics calculated for {len(originals)} samples")
        return metrics
    
    def calculate_ssim(self, img1, img2, window_size=11, sigma=1.5):
        """Calculate SSIM between two images"""
        # Convert to float and normalize
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Constants for SSIM
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        
        # Calculate means
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # Calculate variances and covariance
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim = numerator / denominator
        return ssim
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = max(img1.max(), img2.max())
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def visualize_results(self, originals, reconstructions, metrics, num_display=8):
        """Create comprehensive visualization of results"""
        print(f"\n🎨 CREATING VISUALIZATIONS")
        
        # 1. Sample Comparisons
        fig, axes = plt.subplots(3, num_display, figsize=(num_display * 2, 6))
        fig.suptitle('Reconstruction Results: Original vs Generated vs Difference', fontsize=16)
        
        for i in range(min(num_display, len(originals))):
            # Original
            orig_img = originals[i, 0]  # Take first channel
            axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed
            recon_img = reconstructions[i, 0]
            axes[1, i].imshow(recon_img, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            
            # Difference
            diff_img = np.abs(orig_img - recon_img)
            im = axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Metrics Distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Reconstruction Quality Metrics Distribution', fontsize=16)
        
        # MSE
        axes[0, 0].hist(metrics['mse_values'], bins=20, alpha=0.7, color='red')
        axes[0, 0].set_title(f'MSE Distribution\nMean: {metrics["mse_mean"]:.4f} ± {metrics["mse_std"]:.4f}')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        
        # SSIM
        axes[0, 1].hist(metrics['ssim_values'], bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title(f'SSIM Distribution\nMean: {metrics["ssim_mean"]:.4f} ± {metrics["ssim_std"]:.4f}')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        
        # PSNR
        axes[0, 2].hist(metrics['psnr_values'], bins=20, alpha=0.7, color='green')
        axes[0, 2].set_title(f'PSNR Distribution\nMean: {metrics["psnr_mean"]:.2f} ± {metrics["psnr_std"]:.2f} dB')
        axes[0, 2].set_xlabel('PSNR (dB)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Correlation
        axes[1, 0].hist(metrics['correlation_values'], bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_title(f'Correlation Distribution\nMean: {metrics["correlation_mean"]:.4f} ± {metrics["correlation_std"]:.4f}')
        axes[1, 0].set_xlabel('Pearson Correlation')
        axes[1, 0].set_ylabel('Frequency')
        
        # MAE
        axes[1, 1].hist(metrics['mae_values'], bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title(f'MAE Distribution\nMean: {metrics["mae_mean"]:.4f} ± {metrics["mae_std"]:.4f}')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('Frequency')
        
        # Metrics Summary
        axes[1, 2].axis('off')
        summary_text = f"""
        RECONSTRUCTION QUALITY SUMMARY
        
        MSE:         {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}
        SSIM:        {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}
        PSNR:        {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB
        Correlation: {metrics['correlation_mean']:.4f} ± {metrics['correlation_std']:.4f}
        MAE:         {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}
        
        Samples:     {len(metrics['mse_values'])}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('reconstruction_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Best and Worst Reconstructions
        # Find best and worst based on SSIM
        ssim_values = np.array(metrics['ssim_values'])
        best_indices = np.argsort(ssim_values)[-4:][::-1]  # Top 4
        worst_indices = np.argsort(ssim_values)[:4]        # Bottom 4
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Best vs Worst Reconstructions (by SSIM)', fontsize=16)
        
        for i, idx in enumerate(best_indices):
            # Original
            axes[i, 0].imshow(originals[idx, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Best {i+1}: Original')
            axes[i, 0].axis('off')
            
            # Reconstructed
            axes[i, 1].imshow(reconstructions[idx, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'SSIM: {ssim_values[idx]:.3f}')
            axes[i, 1].axis('off')
        
        for i, idx in enumerate(worst_indices):
            # Original
            axes[i, 2].imshow(originals[idx, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Worst {i+1}: Original')
            axes[i, 2].axis('off')
            
            # Reconstructed
            axes[i, 3].imshow(reconstructions[idx, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 3].set_title(f'SSIM: {ssim_values[idx]:.3f}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('best_worst_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self, metrics):
        """Print comprehensive evaluation summary"""
        print(f"\n" + "="*60)
        print(f"🎯 DIGIT69 LDM RECONSTRUCTION EVALUATION SUMMARY")
        print(f"="*60)
        
        print(f"\n📊 QUANTITATIVE METRICS:")
        print(f"   MSE (Lower is better):         {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
        print(f"   SSIM (Higher is better):       {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"   PSNR (Higher is better):       {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"   Correlation (Higher is better): {metrics['correlation_mean']:.4f} ± {metrics['correlation_std']:.4f}")
        print(f"   MAE (Lower is better):         {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        
        print(f"\n🎯 QUALITY ASSESSMENT:")
        
        # SSIM interpretation
        ssim_mean = metrics['ssim_mean']
        if ssim_mean > 0.8:
            ssim_quality = "Excellent"
        elif ssim_mean > 0.6:
            ssim_quality = "Good"
        elif ssim_mean > 0.4:
            ssim_quality = "Fair"
        else:
            ssim_quality = "Poor"
        
        print(f"   SSIM Quality: {ssim_quality} ({ssim_mean:.3f})")
        
        # Correlation interpretation
        corr_mean = metrics['correlation_mean']
        if corr_mean > 0.7:
            corr_quality = "Strong"
        elif corr_mean > 0.5:
            corr_quality = "Moderate"
        elif corr_mean > 0.3:
            corr_quality = "Weak"
        else:
            corr_quality = "Very Weak"
        
        print(f"   Correlation: {corr_quality} ({corr_mean:.3f})")
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Best SSIM:  {max(metrics['ssim_values']):.4f}")
        print(f"   Worst SSIM: {min(metrics['ssim_values']):.4f}")
        print(f"   SSIM Range: {max(metrics['ssim_values']) - min(metrics['ssim_values']):.4f}")
        
        print(f"\n🔗 COMPARISON WITH BASELINES:")
        print(f"   Random noise SSIM: ~0.0")
        print(f"   Perfect reconstruction SSIM: 1.0")
        print(f"   Our model SSIM: {ssim_mean:.3f}")
        
        print(f"\n✅ EVALUATION COMPLETED!")
        print(f"   Total samples evaluated: {len(metrics['mse_values'])}")
        print(f"   Generated files:")
        print(f"     - reconstruction_comparison.png")
        print(f"     - reconstruction_metrics.png") 
        print(f"     - best_worst_reconstructions.png")

def main():
    """Main evaluation function"""
    print("🔍 DIGIT69 LDM RECONSTRUCTION EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ReconstructionEvaluator("digit69_ldm_final.pth")
    
    # Generate reconstructions
    originals, reconstructions, fmri_embeddings = evaluator.generate_reconstructions(
        num_samples=10,  # Evaluate all test samples
        num_steps=50     # Use 50 diffusion steps for quality
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(originals, reconstructions)
    
    # Create visualizations
    evaluator.visualize_results(originals, reconstructions, metrics)
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Save results
    results = {
        'originals': originals,
        'reconstructions': reconstructions,
        'fmri_embeddings': fmri_embeddings,
        'metrics': metrics
    }
    
    with open('reconstruction_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved: reconstruction_evaluation_results.pkl")

if __name__ == "__main__":
    main()
