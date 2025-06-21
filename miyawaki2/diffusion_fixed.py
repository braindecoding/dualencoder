#!/usr/bin/env python3
"""
Fixed Diffusion Decoder for Miyawaki2
Proper conditioning, training approach, and simplified architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import UNet2DModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available, using CNN decoder")

class FixedDiffusion_Decoder(nn.Module):
    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.correlation_dim = correlation_dim
        self.output_shape = output_shape
        self.training_mode = True
        
        if DIFFUSERS_AVAILABLE:
            # Simplified U-Net (no attention, fewer parameters)
            self.unet = UNet2DModel(
                in_channels=1,
                out_channels=1,
                layers_per_block=1,  # Reduced complexity
                block_out_channels=(32, 64),  # Smaller channels
                down_block_types=("DownBlock2D", "DownBlock2D"),  # No attention
                up_block_types=("UpBlock2D", "UpBlock2D"),  # No attention
                norm_num_groups=8,
                sample_size=28
            )

            # Simplified scheduler (fewer timesteps)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=50,  # Much fewer steps
                beta_schedule="linear",
                prediction_type="epsilon"  # Predict noise
            )
            
            # Conditioning network - project to spatial conditioning
            self.condition_proj = nn.Sequential(
                nn.Linear(correlation_dim * 2, 256),  # correlation + fmri_latent
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 7*7*4),  # Spatial conditioning: 7x7x4
                nn.ReLU()
            )
            
            # Conditioning injection layer
            self.condition_conv = nn.Conv2d(4, 32, kernel_size=3, padding=1)
            
        else:
            # Fallback: Better CNN decoder
            self.decoder = nn.Sequential(
                nn.Linear(correlation_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 7*7*64),
                nn.ReLU()
            )
            
            self.conv_decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7x7 -> 14x14
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 14x14 -> 28x28
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, 1, 1),  # Final conv
                nn.Sigmoid()
            )
            
            self.scheduler = None
            self.condition_proj = None
            print("Using CNN decoder instead of diffusion")
    
    def add_noise(self, clean_images, noise, timesteps):
        """Add noise to clean images for training"""
        if self.scheduler is None:
            return clean_images  # Fallback mode
            
        # Get noise schedule
        alphas_cumprod = self.scheduler.alphas_cumprod.to(clean_images.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(clean_images.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(clean_images.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # Add noise
        noisy_images = sqrt_alpha_prod * clean_images + sqrt_one_minus_alpha_prod * noise
        return noisy_images
    
    def forward_training(self, CLIP_corr, fmriTest_latent, target_images):
        """Training forward pass - returns noise prediction loss"""
        if not DIFFUSERS_AVAILABLE or self.scheduler is None:
            # Fallback CNN mode
            combined_input = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
            features = self.decoder(combined_input)
            features = features.view(-1, 64, 7, 7)
            output = self.conv_decoder(features)
            return output, F.mse_loss(output, target_images)
        
        batch_size = target_images.shape[0]
        device = target_images.device
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(target_images)
        
        # Add noise to target images
        noisy_images = self.add_noise(target_images, noise, timesteps)
        
        # Prepare conditioning
        combined_condition = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
        spatial_condition = self.condition_proj(combined_condition)
        spatial_condition = spatial_condition.view(-1, 4, 7, 7)
        
        # Upsample conditioning to match input size
        spatial_condition = F.interpolate(spatial_condition, size=28, mode='bilinear', align_corners=False)
        condition_features = self.condition_conv(spatial_condition)
        
        # Add conditioning to noisy images (simple injection)
        conditioned_input = noisy_images + 0.1 * condition_features.mean(dim=1, keepdim=True)
        
        # Predict noise
        noise_pred = self.unet(conditioned_input, timesteps).sample
        
        # Compute loss (noise prediction)
        loss = F.mse_loss(noise_pred, noise)
        
        return noise_pred, loss
    
    def forward_inference(self, CLIP_corr, fmriTest_latent, num_inference_steps=10):
        """Inference forward pass - denoising process"""
        if not DIFFUSERS_AVAILABLE or self.scheduler is None:
            # Fallback CNN mode
            combined_input = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
            features = self.decoder(combined_input)
            features = features.view(-1, 64, 7, 7)
            output = self.conv_decoder(features)
            return output
        
        batch_size = CLIP_corr.shape[0]
        device = CLIP_corr.device
        
        # Prepare conditioning
        combined_condition = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
        spatial_condition = self.condition_proj(combined_condition)
        spatial_condition = spatial_condition.view(-1, 4, 7, 7)
        spatial_condition = F.interpolate(spatial_condition, size=28, mode='bilinear', align_corners=False)
        condition_features = self.condition_conv(spatial_condition)
        
        # Start with pure noise
        image = torch.randn(batch_size, 1, *self.output_shape, device=device)
        
        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Add conditioning
            conditioned_input = image + 0.1 * condition_features.mean(dim=1, keepdim=True)

            # Predict noise
            with torch.no_grad():
                timestep_tensor = t.unsqueeze(0).repeat(batch_size).to(device)
                noise_pred = self.unet(conditioned_input, timestep_tensor).sample

            # Remove noise
            image = self.scheduler.step(noise_pred, t, image).prev_sample
        
        # Clamp to [0, 1]
        image = torch.clamp(image, 0, 1)
        return image
    
    def forward(self, CLIP_corr, fmriTest_latent, target_images=None, num_inference_steps=10):
        """
        Forward pass - training or inference mode
        """
        if self.training and target_images is not None:
            # Training mode - return noise prediction and loss
            return self.forward_training(CLIP_corr, fmriTest_latent, target_images)
        else:
            # Inference mode - return generated images
            return self.forward_inference(CLIP_corr, fmriTest_latent, num_inference_steps)

class SimpleCNN_Decoder(nn.Module):
    """Simple CNN decoder as alternative to diffusion"""
    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.output_shape = output_shape
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(correlation_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7*7*64),
            nn.ReLU()
        )
        
        # Convolutional decoder
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 14x14 -> 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, CLIP_corr, fmriTest_latent, target_images=None):
        """Forward pass"""
        # Combine inputs
        combined_input = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
        
        # Project to feature map
        features = self.input_proj(combined_input)
        features = features.view(-1, 64, 7, 7)
        
        # Decode to image
        output = self.conv_decoder(features)
        
        if self.training and target_images is not None:
            loss = F.mse_loss(output, target_images)
            return output, loss
        else:
            return output

# Alias for backward compatibility
Diffusion_Decoder = FixedDiffusion_Decoder
