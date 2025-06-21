#!/usr/bin/env python3
"""
VAE Approach for Digit69 fMRI to Image Reconstruction
Variational Autoencoder with fMRI conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from simple_baseline_model import Digit69BaselineDataset

class VAEEncoder(nn.Module):
    """VAE Encoder for images"""
    
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 4x4 -> 2x2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * 2 * 2  # 1024
        
        # Latent layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        print(f"📊 VAE Encoder:")
        print(f"   Input: {input_channels}x28x28")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Flatten size: {self.flatten_size}")
    
    def forward(self, x):
        # Convolutional encoding
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """VAE Decoder for images"""
    
    def __init__(self, latent_dim=128, output_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        
        # Project latent to feature map
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        print(f"📊 VAE Decoder:")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Output: {output_channels}x28x28")
    
    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        
        # Deconvolutional decoding
        x = self.deconv_layers(x)
        
        return x

class fMRIConditioner(nn.Module):
    """fMRI conditioning network"""
    
    def __init__(self, fmri_dim=512, latent_dim=128, hidden_dims=[256, 256]):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.latent_dim = latent_dim
        
        # Build conditioning network
        layers = []
        input_dim = fmri_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layers for mu and logvar
        layers.append(nn.Linear(input_dim, latent_dim * 2))
        
        self.network = nn.Sequential(*layers)
        
        print(f"📊 fMRI Conditioner:")
        print(f"   Input: fMRI ({fmri_dim})")
        print(f"   Hidden: {hidden_dims}")
        print(f"   Output: latent params ({latent_dim * 2})")
    
    def forward(self, fmri_emb):
        # Get conditioning parameters
        params = self.network(fmri_emb)
        mu_cond, logvar_cond = params.chunk(2, dim=1)
        
        return mu_cond, logvar_cond

class ConditionalVAE(nn.Module):
    """Conditional VAE for fMRI to image reconstruction"""
    
    def __init__(self, fmri_dim=512, latent_dim=128, input_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Components
        self.encoder = VAEEncoder(input_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_channels)
        self.fmri_conditioner = fMRIConditioner(fmri_dim, latent_dim)
        
        print(f"📊 Conditional VAE:")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, images, fmri_emb):
        """Forward pass for training"""
        # Encode images
        mu_img, logvar_img = self.encoder(images)
        z_img = self.reparameterize(mu_img, logvar_img)
        
        # Get fMRI conditioning
        mu_fmri, logvar_fmri = self.fmri_conditioner(fmri_emb)
        
        # Decode from image latent
        recon_img = self.decoder(z_img)
        
        return recon_img, mu_img, logvar_img, mu_fmri, logvar_fmri
    
    def generate_from_fmri(self, fmri_emb):
        """Generate images from fMRI embeddings"""
        # Get fMRI conditioning
        mu_fmri, logvar_fmri = self.fmri_conditioner(fmri_emb)
        
        # Sample from fMRI-conditioned distribution
        z_fmri = self.reparameterize(mu_fmri, logvar_fmri)
        
        # Decode to image
        generated_img = self.decoder(z_fmri)
        
        return generated_img

def vae_loss_function(recon_x, x, mu_img, logvar_img, mu_fmri, logvar_fmri, 
                     beta=1.0, gamma=1.0):
    """VAE loss function with fMRI conditioning"""
    
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence for image latent
    kl_img = -0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp())
    
    # KL divergence between image and fMRI latents (alignment loss)
    kl_align = torch.sum(
        0.5 * (logvar_img.exp() + (mu_img - mu_fmri).pow(2)) / logvar_fmri.exp() 
        - 0.5 + 0.5 * (logvar_fmri - logvar_img)
    )
    
    # Total loss
    total_loss = recon_loss + beta * kl_img + gamma * kl_align
    
    return total_loss, recon_loss, kl_img, kl_align

class VAETrainer:
    """VAE trainer"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.test_losses = []
        
        print(f"📊 VAE Trainer Configuration:")
        print(f"   Optimizer: Adam (lr=1e-3)")
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Device: {device}")
    
    def train_epoch(self, beta=1.0, gamma=1.0):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl_img = 0
        epoch_kl_align = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} [Train]')
        
        for fmri_emb, images in pbar:
            fmri_emb = fmri_emb.to(self.device)
            images = images.to(self.device)
            
            # Forward pass
            recon_img, mu_img, logvar_img, mu_fmri, logvar_fmri = self.model(images, fmri_emb)
            
            # Calculate loss
            total_loss, recon_loss, kl_img, kl_align = vae_loss_function(
                recon_img, images, mu_img, logvar_img, mu_fmri, logvar_fmri, beta, gamma
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl_img += kl_img.item()
            epoch_kl_align += kl_align.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.2f}',
                'recon': f'{recon_loss.item():.2f}',
                'kl_img': f'{kl_img.item():.2f}',
                'kl_align': f'{kl_align.item():.2f}'
            })
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl_img = epoch_kl_img / num_batches
        avg_kl_align = epoch_kl_align / num_batches
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_recon, avg_kl_img, avg_kl_align
    
    def test_epoch(self, beta=1.0, gamma=1.0):
        """Test for one epoch"""
        self.model.eval()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl_img = 0
        epoch_kl_align = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {self.epoch+1} [Test]')
            
            for fmri_emb, images in pbar:
                fmri_emb = fmri_emb.to(self.device)
                images = images.to(self.device)
                
                # Forward pass
                recon_img, mu_img, logvar_img, mu_fmri, logvar_fmri = self.model(images, fmri_emb)
                
                # Calculate loss
                total_loss, recon_loss, kl_img, kl_align = vae_loss_function(
                    recon_img, images, mu_img, logvar_img, mu_fmri, logvar_fmri, beta, gamma
                )
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl_img += kl_img.item()
                epoch_kl_align += kl_align.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.2f}',
                    'recon': f'{recon_loss.item():.2f}',
                    'kl_img': f'{kl_img.item():.2f}',
                    'kl_align': f'{kl_align.item():.2f}'
                })
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl_img = epoch_kl_img / num_batches
        avg_kl_align = epoch_kl_align / num_batches
        
        self.test_losses.append(avg_loss)
        
        return avg_loss, avg_recon, avg_kl_img, avg_kl_align
    
    def generate_samples(self, num_samples=4):
        """Generate sample images from fMRI"""
        self.model.eval()
        
        with torch.no_grad():
            # Get test conditions
            test_batch = next(iter(self.test_loader))
            fmri_emb, target_images = test_batch
            fmri_emb = fmri_emb[:num_samples].to(self.device)
            target_images = target_images[:num_samples]
            
            # Generate from fMRI
            generated_images = self.model.generate_from_fmri(fmri_emb)
            
            return generated_images.cpu(), target_images
    
    def train(self, num_epochs=100, beta_schedule=None, gamma_schedule=None):
        """Main training loop"""
        print(f"🚀 STARTING VAE TRAINING")
        print(f"   Epochs: {num_epochs}")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Get beta and gamma for this epoch
            beta = 1.0 if beta_schedule is None else beta_schedule(epoch)
            gamma = 1.0 if gamma_schedule is None else gamma_schedule(epoch)
            
            # Training
            train_loss, train_recon, train_kl_img, train_kl_align = self.train_epoch(beta, gamma)
            
            # Testing
            test_loss, test_recon, test_kl_img, test_kl_align = self.test_epoch(beta, gamma)
            
            # Learning rate scheduling
            self.scheduler.step(test_loss)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}: Train={train_loss:.2f}, Test={test_loss:.2f}, "
                  f"Recon={test_recon:.2f}, KL_img={test_kl_img:.2f}, KL_align={test_kl_align:.2f}")
            
            # Save best model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.model.state_dict(), 'vae_best.pth')
                print(f"   💾 New best model saved (loss: {test_loss:.2f})")
            
            # Generate samples every 20 epochs
            if (epoch + 1) % 20 == 0:
                generated, targets = self.generate_samples()
                self.plot_samples(generated, targets, epoch + 1)
        
        training_time = time.time() - start_time
        
        # Save final model
        torch.save(self.model.state_dict(), 'vae_final.pth')
        
        print(f"\n✅ VAE Training completed!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Best test loss: {self.best_loss:.2f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_samples(self, generated, targets, epoch):
        """Plot generated samples"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'VAE Samples - Epoch {epoch}', fontsize=16)
        
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
        plt.savefig(f'vae_samples_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.test_losses, label='Test Loss', color='red')
        plt.title('VAE Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.test_losses, label='Test Loss', color='red')
        plt.title('VAE Training and Test Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('vae_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main VAE training function"""
    print("🚀 VAE APPROACH FOR DIGIT69")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load datasets
    train_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "train", target_size=28)
    test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create VAE model
    model = ConditionalVAE(fmri_dim=512, latent_dim=128, input_channels=1)
    
    # Create trainer
    trainer = VAETrainer(model, train_loader, test_loader, device)
    
    # Beta annealing schedule (gradually increase KL weight)
    def beta_schedule(epoch):
        return min(1.0, epoch / 50.0)
    
    # Gamma schedule (alignment loss weight)
    def gamma_schedule(epoch):
        return 1.0
    
    # Start training
    trainer.train(num_epochs=100, beta_schedule=beta_schedule, gamma_schedule=gamma_schedule)
    
    print(f"\n📁 Generated files:")
    print(f"   - vae_best.pth")
    print(f"   - vae_final.pth")
    print(f"   - vae_training_curves.png")
    print(f"   - vae_samples_epoch_*.png")

if __name__ == "__main__":
    main()
