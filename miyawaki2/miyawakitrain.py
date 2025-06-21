#!/usr/bin/env python3
"""
MiyawakaTrainer - Original 3-Phase Training Architecture
Separate component training approach
"""

import torch
import torch.nn.functional as F
from fmriencoder import fMRI_Encoder
from stimencoder import Shape_Encoder
from clipcorrtrain import CLIP_Correlation
from diffusion import Diffusion_Decoder
from gan import GAN_Decoder

class MiyawakaTrainer:
    def __init__(self):
        # Initialize models
        self.fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512)
        self.shape_encoder = Shape_Encoder(latent_dim=512)
        self.clip_correlation = CLIP_Correlation(latent_dim=512)
        self.diffusion_decoder = Diffusion_Decoder(correlation_dim=512, output_shape=(28, 28))
        self.gan_decoder = GAN_Decoder(correlation_dim=512, output_shape=(28, 28))
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            list(self.fmri_encoder.parameters()) + list(self.shape_encoder.parameters()),
            lr=1e-4
        )
        self.correlation_optimizer = torch.optim.Adam(self.clip_correlation.parameters(), lr=1e-4)
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion_decoder.parameters(), lr=1e-4)
        self.gan_optimizer = torch.optim.Adam(self.gan_decoder.parameters(), lr=1e-4)
    
    def train_phase1_encoders(self, train_loader, epochs=100):
        """Phase 1: Train latent encoders"""
        for epoch in range(epochs):
            total_loss = 0
            for fmri_batch, stim_batch in train_loader:
                # Forward pass
                fmri_latent = self.fmri_encoder(fmri_batch)
                stim_latent = self.shape_encoder(stim_batch)
                
                # Contrastive loss
                corr_loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                
                # Backward pass
                self.encoder_optimizer.zero_grad()
                corr_loss.backward()
                self.encoder_optimizer.step()
                
                total_loss += corr_loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_phase2_correlation(self, train_loader, epochs=50):
        """Phase 2: Train correlation learning"""
        for epoch in range(epochs):
            total_loss = 0
            for fmri_batch, stim_batch in train_loader:
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri_batch)
                    stim_latent = self.shape_encoder(stim_batch)
                
                # Train correlation
                CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)
                corr_loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                
                self.correlation_optimizer.zero_grad()
                corr_loss.backward()
                self.correlation_optimizer.step()
                
                total_loss += corr_loss.item()
            
            print(f"Correlation Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_phase3_decoders(self, train_loader, test_loader, epochs=100):
        """Phase 3: Train generative decoders"""
        for epoch in range(epochs):
            # Train diffusion
            diff_loss_total = 0
            for fmri_batch, stim_batch in train_loader:
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri_batch)
                    stim_latent = self.shape_encoder(stim_batch)
                    CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)
                
                # Get test fMRI latents (simulate testing)
                fmriTest_latent = fmri_latent  # In practice, use separate test data
                
                # Train diffusion
                stimPred_diff = self.diffusion_decoder(CLIP_corr, fmriTest_latent)
                diff_loss = F.mse_loss(stimPred_diff, stim_batch)
                
                self.diffusion_optimizer.zero_grad()
                diff_loss.backward()
                self.diffusion_optimizer.step()
                
                diff_loss_total += diff_loss.item()
            
            # Train GAN
            gan_loss_total = 0
            for fmri_batch, stim_batch in train_loader:
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri_batch)
                    stim_latent = self.shape_encoder(stim_batch)
                    CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)

                # Train GAN
                stimPred_gan = self.gan_decoder(CLIP_corr, fmri_latent)
                gan_loss = F.mse_loss(stimPred_gan, stim_batch)

                self.gan_optimizer.zero_grad()
                gan_loss.backward()
                self.gan_optimizer.step()

                gan_loss_total += gan_loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Diffusion Loss: {diff_loss_total/len(train_loader):.4f}, GAN Loss: {gan_loss_total/len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """Evaluate the trained model"""
        self.fmri_encoder.eval()
        self.shape_encoder.eval()
        self.clip_correlation.eval()
        self.diffusion_decoder.eval()
        self.gan_decoder.eval()

        all_diffusion_preds = []
        all_gan_preds = []
        all_targets = []
        all_fmri_latents = []
        all_stim_latents = []

        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri']
                stimulus = batch['stimulus']

                # Get latent representations
                fmri_latent = self.fmri_encoder(fmri)
                stim_latent = self.shape_encoder(stimulus)

                # Get correlation
                CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)

                # Generate predictions
                stimPred_diff = self.diffusion_decoder(CLIP_corr, fmri_latent)
                stimPred_gan = self.gan_decoder(CLIP_corr, fmri_latent)

                all_diffusion_preds.append(stimPred_diff)
                all_gan_preds.append(stimPred_gan)
                all_targets.append(stimulus)
                all_fmri_latents.append(fmri_latent)
                all_stim_latents.append(stim_latent)

        # Concatenate results
        diffusion_preds = torch.cat(all_diffusion_preds, dim=0)
        gan_preds = torch.cat(all_gan_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        fmri_latents = torch.cat(all_fmri_latents, dim=0)
        stim_latents = torch.cat(all_stim_latents, dim=0)

        return {
            'diffusion_preds': diffusion_preds,
            'gan_preds': gan_preds,
            'targets': targets,
            'fmri_latents': fmri_latents,
            'stim_latents': stim_latents
        }

    def to(self, device):
        """Move all models to device"""
        self.fmri_encoder = self.fmri_encoder.to(device)
        self.shape_encoder = self.shape_encoder.to(device)
        self.clip_correlation = self.clip_correlation.to(device)
        self.diffusion_decoder = self.diffusion_decoder.to(device)
        self.gan_decoder = self.gan_decoder.to(device)
        return self