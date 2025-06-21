#!/usr/bin/env python3
"""
Simplified Digit69 Latent Diffusion Model
A simpler, more robust implementation for fMRI-conditioned digit generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TimeEmbedding(nn.Module):
    """Simple time embedding for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, time):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    """Simple residual block with conditioning"""
    
    def __init__(self, in_ch, out_ch, time_dim, condition_dim=None):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_dim, out_ch)
        
        if condition_dim is not None:
            self.condition_mlp = nn.Linear(condition_dim, out_ch)
        else:
            self.condition_mlp = None
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Use simpler normalization
        self.norm1 = nn.Identity()  # Skip normalization for simplicity
        self.norm2 = nn.Identity()
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_emb, condition_emb=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add condition embedding
        if condition_emb is not None and self.condition_mlp is not None:
            condition_emb = self.condition_mlp(condition_emb)
            h = h + condition_emb[:, :, None, None]
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.skip(x)

class SimpleUNet(nn.Module):
    """Simplified UNet for digit generation"""
    
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, condition_dim=512, time_dim=128):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, time_dim, condition_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim, condition_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim, condition_dim)
        
        # Middle
        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_dim, condition_dim)
        
        # Decoder
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2, time_dim, condition_dim)  # 4 + 4 = 8
        self.dec2 = ResBlock(base_ch * 4, base_ch, time_dim, condition_dim)      # 2 + 2 = 4
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim, condition_dim)      # 1 + 1 = 2
        
        # Output
        self.out = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, timesteps, condition):
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Encoder
        h1 = self.enc1(x, time_emb, condition)          # 28x28
        h2 = self.enc2(self.pool(h1), time_emb, condition)  # 14x14
        h3 = self.enc3(self.pool(h2), time_emb, condition)  # 7x7
        
        # Middle
        h = self.mid(h3, time_emb, condition)           # 7x7
        
        # Decoder with skip connections
        h = self.upsample(h)                            # 14x14
        h = torch.cat([h, h2], dim=1)                   # Concatenate skip
        h = self.dec3(h, time_emb, condition)
        
        h = self.upsample(h)                            # 28x28
        h = torch.cat([h, h1], dim=1)                   # Concatenate skip
        h = self.dec2(h, time_emb, condition)
        
        h = self.dec1(h, time_emb, condition)
        
        return self.out(h)

class SimpleDiffusion(nn.Module):
    """Simple diffusion model for digit generation"""
    
    def __init__(self, image_size=28, condition_dim=512, num_timesteps=1000):
        super().__init__()
        
        self.image_size = image_size
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps
        
        # UNet model
        self.unet = SimpleUNet(
            in_ch=1,
            out_ch=1,
            base_ch=64,
            condition_dim=condition_dim,
            time_dim=128
        )
        
        # Noise schedule (linear)
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def add_noise(self, x0, t, noise=None):
        """Add noise to clean images"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x, t, condition):
        """Predict noise"""
        return self.unet(x, t, condition)

    @torch.no_grad()
    def sample(self, condition, num_steps=50):
        """Generate images from condition"""
        device = condition.device
        batch_size = condition.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, 1, self.image_size, self.image_size, device=device)
        
        # Sampling steps
        step_size = self.num_timesteps // num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), self.num_timesteps - 1 - i * step_size, device=device, dtype=torch.long)
            t = torch.clamp(t, 0, self.num_timesteps - 1)
            
            # Predict noise
            predicted_noise = self.unet(x, t, condition)
            
            # Remove noise (simplified DDPM step)
            alpha_t = self.alphas[t[0]]
            alpha_cumprod_t = self.alphas_cumprod[t[0]]
            beta_t = self.betas[t[0]]
            
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            # Add noise for next step (except last)
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise * 0.5  # Reduced noise
        
        return torch.clamp(x, -1, 1)

def test_simple_model():
    """Test the simplified model"""
    print("🧪 Testing Simple Digit69 Diffusion Model")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Initialize model
    model = SimpleDiffusion(
        image_size=28,
        condition_dim=512,
        num_timesteps=1000
    ).to(device)
    
    # Test data
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    conditions = torch.randn(batch_size, 512).to(device)
    
    print(f"\n📊 Input shapes:")
    print(f"   Images: {images.shape}")
    print(f"   Timesteps: {timesteps.shape}")
    print(f"   Conditions: {conditions.shape}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Training forward
            predicted_noise = model(images, timesteps, conditions)
            print(f"   ✅ Predicted noise: {predicted_noise.shape}")
            
            # Test noise addition
            noisy_images, noise = model.add_noise(images, timesteps)
            print(f"   ✅ Noisy images: {noisy_images.shape}")
            
            # Test sampling
            generated = model.sample(conditions, num_steps=10)
            print(f"   ✅ Generated images: {generated.shape}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\n✅ Simple model test completed successfully!")
    
    return model

if __name__ == "__main__":
    test_simple_model()
