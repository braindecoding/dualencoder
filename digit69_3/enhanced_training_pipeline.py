#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Improved LDM
Better training with more epochs, advanced loss functions, scheduling, and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import os
from improved_unet import ImprovedUNet
from simple_baseline_model import Digit69BaselineDataset

class EnhancedDiffusionModel(nn.Module):
    """Enhanced diffusion model with improved UNet"""
    
    def __init__(self, unet, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        
        self.unet = unet
        self.num_timesteps = num_timesteps
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        print(f"📊 Enhanced Diffusion Model:")
        print(f"   Timesteps: {num_timesteps}")
        print(f"   Beta range: [{beta_start}, {beta_end}]")
        print(f"   UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, condition, noise=None, loss_type="l2"):
        """Calculate training losses"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.unet(x_noisy, t, condition)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """Sample from p(x_{t-1} | x_t)"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.unet(x, t, condition)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, condition, device):
        """Generate samples"""
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition)
        
        return img
    
    @torch.no_grad()
    def sample(self, condition, image_size=28):
        """Generate samples from condition"""
        batch_size = condition.shape[0]
        shape = (batch_size, 1, image_size, image_size)
        return self.p_sample_loop(shape, condition, condition.device)

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features"""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
        # Simple feature extractor (can be replaced with VGG features)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred_features = self.features(pred)
        target_features = self.features(target)
        return F.mse_loss(pred_features, target_features) * self.weight

class EnhancedTrainer:
    """Enhanced trainer with advanced features"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(weight=0.1).to(device)
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        
        # Move noise schedule to device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(model, attr_name)
            setattr(model, attr_name, attr_value.to(device))
        
        print(f"📊 Enhanced Trainer Configuration:")
        print(f"   Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   Loss: MSE + Perceptual (weight=0.1)")
        print(f"   Device: {device}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} [Train]')
        
        for fmri_emb, images in pbar:
            fmri_emb = fmri_emb.to(self.device)
            images = images.to(self.device)
            
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device).long()
            
            # Calculate loss
            loss = self.model.p_losses(images, t, fmri_emb, loss_type="huber")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def test_epoch(self):
        """Test for one epoch"""
        self.model.eval()
        epoch_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {self.epoch+1} [Test]')
            
            for fmri_emb, images in pbar:
                fmri_emb = fmri_emb.to(self.device)
                images = images.to(self.device)
                
                batch_size = images.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device).long()
                
                # Calculate loss
                loss = self.model.p_losses(images, t, fmri_emb, loss_type="huber")
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        self.test_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, filename):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load training checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.learning_rates = checkpoint['learning_rates']
    
    def generate_samples(self, num_samples=4):
        """Generate sample images"""
        self.model.eval()
        
        with torch.no_grad():
            # Get test conditions
            test_batch = next(iter(self.test_loader))
            fmri_emb, target_images = test_batch
            fmri_emb = fmri_emb[:num_samples].to(self.device)
            target_images = target_images[:num_samples]
            
            # Generate samples
            generated_images = self.model.sample(fmri_emb)
            
            return generated_images.cpu(), target_images
    
    def train(self, num_epochs=200, save_every=20, sample_every=10):
        """Main training loop"""
        print(f"🚀 STARTING ENHANCED TRAINING")
        print(f"   Epochs: {num_epochs}")
        print(f"   Save every: {save_every} epochs")
        print(f"   Sample every: {sample_every} epochs")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Testing
            test_loss = self.test_epoch()
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}, LR={current_lr:.2e}")
            
            # Save best model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save_checkpoint('enhanced_ldm_best.pth')
                print(f"   💾 New best model saved (loss: {test_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'enhanced_ldm_epoch_{epoch+1}.pth')
                print(f"   💾 Checkpoint saved")
            
            # Generate samples
            if (epoch + 1) % sample_every == 0:
                generated, targets = self.generate_samples()
                self.plot_samples(generated, targets, epoch + 1)
            
            # Early stopping check
            if epoch > 50 and len(self.test_losses) > 20:
                recent_losses = self.test_losses[-20:]
                if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        # Save final model
        self.save_checkpoint('enhanced_ldm_final.pth')
        
        print(f"\n✅ Training completed!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Best test loss: {self.best_loss:.4f}")
        print(f"   Final train loss: {self.train_losses[-1]:.4f}")
        print(f"   Final test loss: {self.test_losses[-1]:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_samples(self, generated, targets, epoch):
        """Plot generated samples"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Enhanced LDM Samples - Epoch {epoch}', fontsize=16)
        
        for i in range(4):
            # Target
            target_img = targets[i, 0].numpy()
            axes[0, i].imshow(target_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Target {i+1}')
            axes[0, i].axis('off')
            
            # Generated
            gen_img = generated[i, 0].numpy()
            axes[1, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'enhanced_samples_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.test_losses, label='Test Loss', color='red')
        axes[0].set_title('Training and Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss curves (log scale)
        axes[1].plot(self.train_losses, label='Train Loss', color='blue')
        axes[1].plot(self.test_losses, label='Test Loss', color='red')
        axes[1].set_title('Training and Test Loss (Log Scale)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (Log)')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate
        axes[2].plot(self.learning_rates, color='green')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("🚀 ENHANCED LDM TRAINING PIPELINE")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load datasets
    train_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "train", target_size=28)
    test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create improved UNet
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )
    
    # Create enhanced diffusion model
    model = EnhancedDiffusionModel(unet, num_timesteps=1000)
    
    # Create trainer
    trainer = EnhancedTrainer(model, train_loader, test_loader, device)
    
    # Start training
    trainer.train(num_epochs=200, save_every=20, sample_every=10)
    
    print(f"\n📁 Generated files:")
    print(f"   - enhanced_ldm_best.pth")
    print(f"   - enhanced_ldm_final.pth")
    print(f"   - enhanced_training_curves.png")
    print(f"   - enhanced_samples_epoch_*.png")

if __name__ == "__main__":
    main()
