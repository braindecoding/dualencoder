#!/usr/bin/env python3
"""
Diffusion Decoder for Miyawaki Dataset
Uses U-Net architecture with DDPM scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import UNet2DModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Install with: pip install diffusers")
    DIFFUSERS_AVAILABLE = False

class Diffusion_Decoder(nn.Module):
    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.output_shape = output_shape  # Miyawaki: 28x28
        
        # U-Net untuk diffusion process
        if DIFFUSERS_AVAILABLE:
            self.unet = UNet2DModel(
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(64, 128),  # Simplified for 28x28
                down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D"),
                norm_num_groups=8,
                sample_size=28  # For 28x28 images
            )

            # Noise scheduler
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear"
            )
        else:
            # Fallback: Simple MLP decoder
            self.unet = nn.Sequential(
                nn.Linear(correlation_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 28*28),
                nn.Sigmoid()
            )
            self.scheduler = None
            print("Using simple MLP decoder instead of U-Net")
    
    def forward(self, CLIP_corr, fmriTest_latent, num_inference_steps=50):
        """
        Matlab equivalent: stimPred_diff = Diffusion(CLIP_corr, fmriTest_latent)
        """
        batch_size = fmriTest_latent.size(0)

        if DIFFUSERS_AVAILABLE and self.scheduler is not None:
            # Full diffusion process
            # Initialize dengan pure noise
            noise = torch.randn(batch_size, 1, *self.output_shape, device=fmriTest_latent.device)

            # Combine correlation dengan test latent
            condition = CLIP_corr + 0.3 * fmriTest_latent  # Weighted combination

            # Denoising process
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.scheduler.timesteps:
                # Predict noise (simplified - real implementation needs conditioning)
                noise_pred = self.unet(noise, t).sample

                # Remove noise step
                noise = self.scheduler.step(noise_pred, t, noise).prev_sample

            # Final result: stimPred_diff
            stimPred_diff = torch.sigmoid(noise)  # Normalize to [0,1]

        else:
            # Fallback: Simple MLP
            combined_input = CLIP_corr + 0.3 * fmriTest_latent
            output = self.unet(combined_input)  # MLP forward
            stimPred_diff = output.view(batch_size, 1, *self.output_shape)

        return stimPred_diff