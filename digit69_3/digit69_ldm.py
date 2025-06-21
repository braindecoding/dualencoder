#!/usr/bin/env python3
"""
Digit69 Latent Diffusion Model
fMRI embedding conditioned image reconstruction using diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with group normalization"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        if condition_dim is not None:
            self.condition_mlp = nn.Linear(condition_dim, out_channels)
        else:
            self.condition_mlp = None
            
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb, condition_emb=None):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add condition embedding if provided
        if condition_emb is not None and self.condition_mlp is not None:
            condition_emb = self.condition_mlp(condition_emb)
            h = h + condition_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(min(8, channels), channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_norm(x)
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.permute(0, 2, 3, 1).view(b, h * w, c)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        out = torch.bmm(attention, v)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        
        return self.to_out(out) + x

class UNet2D(nn.Module):
    """UNet for diffusion model with fMRI conditioning"""
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        model_channels=64,
        condition_dim=512,
        num_res_blocks=2,
        attention_resolutions=[8, 16],
        channel_mult=[1, 2, 4],
        time_embed_dim=256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        
        ch = model_channels
        input_ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(input_ch, ch * mult, time_embed_dim, condition_dim)
                )
                input_ch = ch * mult
                
                if ch * mult in [model_channels * m for m in attention_resolutions]:
                    self.encoder_attentions.append(AttentionBlock(ch * mult))
                else:
                    self.encoder_attentions.append(nn.Identity())
            
            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(nn.Conv2d(ch * mult, ch * mult, 3, stride=2, padding=1))
                self.encoder_attentions.append(nn.Identity())
                input_ch = ch * mult
        
        # Middle
        mid_ch = ch * channel_mult[-1]
        self.middle_block1 = ResidualBlock(mid_ch, mid_ch, time_embed_dim, condition_dim)
        self.middle_attention = AttentionBlock(mid_ch)
        self.middle_block2 = ResidualBlock(mid_ch, mid_ch, time_embed_dim, condition_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                if level == 0 and i == 0:
                    # First decoder block
                    self.decoder_blocks.append(
                        ResidualBlock(mid_ch, ch * mult, time_embed_dim, condition_dim)
                    )
                else:
                    # Skip connections double the input channels
                    self.decoder_blocks.append(
                        ResidualBlock(ch * mult * 2, ch * mult, time_embed_dim, condition_dim)
                    )
                
                if ch * mult in [model_channels * m for m in attention_resolutions]:
                    self.decoder_attentions.append(AttentionBlock(ch * mult))
                else:
                    self.decoder_attentions.append(nn.Identity())
            
            if level < len(channel_mult) - 1:
                self.decoder_blocks.append(nn.ConvTranspose2d(ch * mult, ch * mult, 4, stride=2, padding=1))
                self.decoder_attentions.append(nn.Identity())
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, model_channels), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, condition=None):
        """
        Forward pass
        Args:
            x: Input tensor (B, C, H, W)
            timesteps: Timestep tensor (B,)
            condition: fMRI embedding (B, condition_dim)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for block, attention in zip(self.encoder_blocks, self.encoder_attentions):
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, condition)
            else:  # Downsampling
                h = block(h)
            
            h = attention(h)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block1(h, time_emb, condition)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_emb, condition)
        
        # Decoder
        skip_idx = len(skip_connections) - 1
        
        for block, attention in zip(self.decoder_blocks, self.decoder_attentions):
            if isinstance(block, ResidualBlock):
                # Concatenate skip connection
                if skip_idx >= 0:
                    skip = skip_connections[skip_idx]
                    if h.shape[2:] != skip.shape[2:]:
                        # Resize if needed
                        skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
                    h = torch.cat([h, skip], dim=1)
                    skip_idx -= 1
                
                h = block(h, time_emb, condition)
            else:  # Upsampling
                h = block(h)
            
            h = attention(h)
        
        # Output
        return self.output_conv(h)

class Digit69LDM(nn.Module):
    """Complete Digit69 Latent Diffusion Model"""
    
    def __init__(
        self,
        image_size=28,
        in_channels=1,
        condition_dim=512,
        model_channels=64,
        num_timesteps=1000
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps
        
        # UNet model
        self.unet = UNet2D(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            condition_dim=condition_dim
        )
        
        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule for diffusion"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(self, x0, t, noise=None):
        """Add noise to images according to diffusion schedule"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def forward(self, x, t, condition):
        """Predict noise given noisy image, timestep, and condition"""
        return self.unet(x, t, condition)
    
    @torch.no_grad()
    def sample(self, condition, shape=None, num_inference_steps=50):
        """Generate image from fMRI condition"""
        if shape is None:
            shape = (condition.shape[0], self.in_channels, self.image_size, self.image_size)
        
        device = condition.device
        
        # Start from random noise
        img = torch.randn(shape, device=device)
        
        # Sampling timesteps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.unet(img, t_batch, condition)
            
            # Remove noise
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            img = (1 / torch.sqrt(alpha_t)) * (img - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            # Add noise for next step (except last step)
            if t > 0:
                noise = torch.randn_like(img)
                img = img + torch.sqrt(beta_t) * noise
        
        return torch.clamp(img, -1, 1)

def test_model():
    """Test model architecture"""
    print("🧪 Testing Digit69LDM Architecture")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = Digit69LDM(
        image_size=28,
        in_channels=1,
        condition_dim=512,
        model_channels=64
    ).to(device)
    
    # Test inputs
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    conditions = torch.randn(batch_size, 512).to(device)
    
    print(f"📊 Input shapes:")
    print(f"   Images: {images.shape}")
    print(f"   Timesteps: {timesteps.shape}")
    print(f"   Conditions: {conditions.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Training forward (noise prediction)
        predicted_noise = model(images, timesteps, conditions)
        print(f"   Predicted noise: {predicted_noise.shape}")
        
        # Sampling
        generated = model.sample(conditions, num_inference_steps=10)
        print(f"   Generated images: {generated.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\n✅ Model test completed successfully!")
    
    return model

if __name__ == "__main__":
    test_model()
