#!/usr/bin/env python3
"""
Generative backend untuk Miyawaki dataset
Implementasi Diffusion dan GAN decoder untuk generate stimulus dari fMRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miyawaki_dual_encoder import MiyawakiDualEncoder
from miyawaki_dataset_loader import load_miyawaki_dataset, create_dataloaders
from pathlib import Path

class MiyawakiDiffusionDecoder(nn.Module):
    """Simplified decoder untuk Miyawaki (tanpa full diffusion)"""

    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.output_shape = output_shape
        self.correlation_dim = correlation_dim

        # Simple decoder network
        self.decoder = nn.Sequential(
            nn.Linear(correlation_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, np.prod(output_shape)),
            nn.Sigmoid()  # Output in [0,1] range
        )

    def forward(self, correlation_emb, target_stimulus=None):
        """
        Forward pass for decoder
        """
        batch_size = correlation_emb.size(0)

        # Generate stimulus from correlation
        generated = self.decoder(correlation_emb)
        generated = generated.view(batch_size, *self.output_shape)

        if self.training and target_stimulus is not None:
            # Return both generated and target for loss computation
            return generated, target_stimulus
        else:
            # Inference mode
            return generated

class MiyawakiGANDecoder(nn.Module):
    """GAN-style decoder untuk Miyawaki"""
    
    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.output_shape = output_shape
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(correlation_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, np.prod(output_shape)),
            nn.Sigmoid()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(np.prod(output_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, correlation_emb):
        """Generate stimulus from correlation embedding"""
        generated = self.generator(correlation_emb)
        return generated.view(-1, *self.output_shape)
    
    def discriminate(self, stimulus):
        """Discriminate real vs fake stimulus"""
        stimulus_flat = stimulus.view(stimulus.size(0), -1)
        return self.discriminator(stimulus_flat)

def train_diffusion_decoder(dual_encoder, diffusion_decoder, train_loader, test_loader, 
                           num_epochs=50, lr=1e-3):
    """Train diffusion decoder"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze dual encoder
    for param in dual_encoder.parameters():
        param.requires_grad = False
    dual_encoder.eval()
    
    diffusion_decoder = diffusion_decoder.to(device)
    optimizer = torch.optim.AdamW(diffusion_decoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    test_losses = []
    
    print("Training Diffusion Decoder...")
    
    for epoch in range(num_epochs):
        # Training
        diffusion_decoder.train()
        train_loss = 0.0
        
        for batch in train_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            
            # Get correlation embedding from dual encoder
            with torch.no_grad():
                outputs = dual_encoder(fmri, stimulus)
                correlation_emb = outputs['correlation']
            
            optimizer.zero_grad()
            
            # Decoder loss
            result = diffusion_decoder(correlation_emb, stimulus)
            if isinstance(result, tuple):
                generated, target = result
            else:
                generated = result
                target = stimulus
            loss = F.mse_loss(generated, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        diffusion_decoder.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(device)
                stimulus = batch['stimulus'].to(device)
                
                outputs = dual_encoder(fmri, stimulus)
                correlation_emb = outputs['correlation']

                result = diffusion_decoder(correlation_emb, stimulus)
                if isinstance(result, tuple):
                    generated, target = result
                else:
                    generated = result
                    target = stimulus
                loss = F.mse_loss(generated, target)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
    
    return train_losses, test_losses

def train_gan_decoder(dual_encoder, gan_decoder, train_loader, test_loader,
                     num_epochs=50, lr=1e-3):
    """Train GAN decoder"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze dual encoder
    for param in dual_encoder.parameters():
        param.requires_grad = False
    dual_encoder.eval()
    
    gan_decoder = gan_decoder.to(device)
    
    # Separate optimizers for generator and discriminator
    gen_optimizer = torch.optim.AdamW(gan_decoder.generator.parameters(), lr=lr, weight_decay=1e-4)
    disc_optimizer = torch.optim.AdamW(gan_decoder.discriminator.parameters(), lr=lr, weight_decay=1e-4)
    
    gen_losses = []
    disc_losses = []
    
    print("Training GAN Decoder...")
    
    for epoch in range(num_epochs):
        gan_decoder.train()
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        
        for batch in train_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            batch_size = fmri.size(0)
            
            # Get correlation embedding
            with torch.no_grad():
                outputs = dual_encoder(fmri, stimulus)
                correlation_emb = outputs['correlation']
            
            # Train Discriminator
            disc_optimizer.zero_grad()
            
            # Real samples
            real_pred = gan_decoder.discriminate(stimulus)
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            
            # Fake samples
            fake_stimulus = gan_decoder(correlation_emb)
            fake_pred = gan_decoder.discriminate(fake_stimulus.detach())
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            
            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train Generator
            gen_optimizer.zero_grad()
            
            fake_stimulus = gan_decoder(correlation_emb)
            fake_pred = gan_decoder.discriminate(fake_stimulus)
            gen_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
            
            # Add reconstruction loss
            recon_loss = F.mse_loss(fake_stimulus, stimulus)
            total_gen_loss = gen_loss + 10.0 * recon_loss  # Weight reconstruction loss
            
            total_gen_loss.backward()
            gen_optimizer.step()
            
            epoch_gen_loss += total_gen_loss.item()
            epoch_disc_loss += disc_loss.item()
        
        epoch_gen_loss /= len(train_loader)
        epoch_disc_loss /= len(train_loader)
        
        gen_losses.append(epoch_gen_loss)
        disc_losses.append(epoch_disc_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Generator Loss: {epoch_gen_loss:.4f}")
            print(f"  Discriminator Loss: {epoch_disc_loss:.4f}")
    
    return gen_losses, disc_losses

def evaluate_generation_quality(dual_encoder, decoder, test_loader, decoder_type='diffusion'):
    """Evaluate generation quality"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dual_encoder.eval()
    decoder.eval()
    
    generated_stimuli = []
    real_stimuli = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            label = batch['label']
            
            # Get correlation embedding
            outputs = dual_encoder(fmri, stimulus)
            correlation_emb = outputs['correlation']
            
            # Generate stimulus
            generated = decoder(correlation_emb)
            
            generated_stimuli.append(generated.cpu())
            real_stimuli.append(stimulus.cpu())
            labels.append(label)
    
    generated_stimuli = torch.cat(generated_stimuli, dim=0)
    real_stimuli = torch.cat(real_stimuli, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Visualize results
    n_samples = min(8, len(generated_stimuli))
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    for i in range(n_samples):
        # Real stimulus
        axes[0, i].imshow(real_stimuli[i], cmap='gray')
        axes[0, i].set_title(f'Real (Class {labels[i]})')
        axes[0, i].axis('off')
        
        # Generated stimulus
        axes[1, i].imshow(generated_stimuli[i], cmap='gray')
        axes[1, i].set_title(f'Generated')
        axes[1, i].axis('off')
    
    plt.suptitle(f'{decoder_type.capitalize()} Generation Results')
    plt.tight_layout()
    plt.savefig(f'miyawaki_{decoder_type}_generation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute MSE
    mse = F.mse_loss(generated_stimuli, real_stimuli).item()
    print(f"{decoder_type.capitalize()} Generation MSE: {mse:.4f}")
    
    return generated_stimuli, real_stimuli

def main():
    """Main function untuk training generative backends"""
    
    # Load trained dual encoder
    model_path = 'miyawaki_dual_encoder.pth'
    if not Path(model_path).exists():
        print(f"Dual encoder model not found: {model_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    dual_encoder = MiyawakiDualEncoder(fmri_dim=967, latent_dim=512)
    dual_encoder.load_state_dict(checkpoint['model_state_dict'])
    dual_encoder = dual_encoder.to(device)
    
    # Load dataset
    filepath = Path("../dataset/miyawaki_structured_28x28.mat")
    dataset_dict = load_miyawaki_dataset(filepath)
    train_loader, test_loader = create_dataloaders(dataset_dict, batch_size=8)
    
    # Train Diffusion Decoder
    print("="*60)
    diffusion_decoder = MiyawakiDiffusionDecoder(correlation_dim=512)
    diff_train_losses, diff_test_losses = train_diffusion_decoder(
        dual_encoder, diffusion_decoder, train_loader, test_loader, num_epochs=30
    )
    
    # Evaluate Diffusion
    print("\nEvaluating Diffusion Generation:")
    diff_generated, diff_real = evaluate_generation_quality(
        dual_encoder, diffusion_decoder, test_loader, 'diffusion'
    )
    
    # Train GAN Decoder
    print("="*60)
    gan_decoder = MiyawakiGANDecoder(correlation_dim=512)
    gan_gen_losses, gan_disc_losses = train_gan_decoder(
        dual_encoder, gan_decoder, train_loader, test_loader, num_epochs=30
    )
    
    # Evaluate GAN
    print("\nEvaluating GAN Generation:")
    gan_generated, gan_real = evaluate_generation_quality(
        dual_encoder, gan_decoder, test_loader, 'gan'
    )
    
    # Save models
    torch.save({
        'diffusion_state_dict': diffusion_decoder.state_dict(),
        'gan_state_dict': gan_decoder.state_dict(),
        'diff_losses': (diff_train_losses, diff_test_losses),
        'gan_losses': (gan_gen_losses, gan_disc_losses)
    }, 'miyawaki_generative_backends.pth')
    
    print("\nTraining completed! Generative backends saved.")

if __name__ == "__main__":
    main()
