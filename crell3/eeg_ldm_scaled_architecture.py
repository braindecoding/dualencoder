#!/usr/bin/env python3
"""
EEG LDM with Scaled-Up Architecture
Enhanced UNet with attention mechanisms and larger capacity for better quality generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Check for CLIP availability
try:
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP library available")
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ CLIP library not available - using fallback mode")

class AttentionBlock(nn.Module):
    """Self-attention block for UNet"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, h * w)  # (b, c, hw)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)  # (b, hw, c)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out

class ResidualBlock(nn.Module):
    """Enhanced residual block with group normalization"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, time_emb, condition_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_out
        
        # Add condition embedding
        cond_out = self.condition_mlp(condition_emb)[:, :, None, None]
        h = h + cond_out
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class ScaledUNet(nn.Module):
    """Scaled-up UNet with attention mechanisms and larger capacity"""
    
    def __init__(self, in_channels=1, out_channels=1, condition_dim=512):
        super().__init__()
        
        # Scaled-up channel dimensions
        self.channels = [64, 128, 256, 512, 768]  # Much larger than [32, 64, 128, 256]
        self.condition_dim = condition_dim
        
        # Time embedding
        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(128, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            # Two residual blocks per level
            self.encoder_blocks.append(nn.ModuleList([
                ResidualBlock(self.channels[i], self.channels[i], time_emb_dim, condition_dim),
                ResidualBlock(self.channels[i], self.channels[i], time_emb_dim, condition_dim)
            ]))
            
            # Add attention at higher resolutions (channels >= 256)
            if self.channels[i] >= 256:
                self.encoder_attentions.append(AttentionBlock(self.channels[i]))
            else:
                self.encoder_attentions.append(nn.Identity())
            
            # Downsample
            self.encoder_downsample.append(
                nn.Conv2d(self.channels[i], self.channels[i + 1], 3, stride=2, padding=1)
            )
        
        # Middle block with attention
        mid_channels = self.channels[-1]
        self.middle_block1 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, condition_dim)
        self.middle_attention = AttentionBlock(mid_channels)
        self.middle_block2 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, condition_dim)
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i in range(len(self.channels) - 1, 0, -1):
            # Upsample
            self.decoder_upsample.append(
                nn.ConvTranspose2d(self.channels[i], self.channels[i - 1], 4, stride=2, padding=1)
            )
            
            # Two residual blocks per level (with skip connections)
            self.decoder_blocks.append(nn.ModuleList([
                ResidualBlock(self.channels[i - 1] * 2, self.channels[i - 1], time_emb_dim, condition_dim),
                ResidualBlock(self.channels[i - 1], self.channels[i - 1], time_emb_dim, condition_dim)
            ]))
            
            # Add attention at higher resolutions
            if self.channels[i - 1] >= 256:
                self.decoder_attentions.append(AttentionBlock(self.channels[i - 1]))
            else:
                self.decoder_attentions.append(nn.Identity())
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, self.channels[0])
        self.output_conv = nn.Conv2d(self.channels[0], out_channels, 3, padding=1)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Scaled UNet Architecture:")
        print(f"   Channels: {self.channels}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Attention layers: {sum(1 for c in self.channels if c >= 256)} levels")
        
    def get_time_embedding(self, timesteps):
        """Sinusoidal time embedding"""
        half_dim = 64
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
        
    def forward(self, x, timesteps, condition):
        # Time embedding
        time_emb = self.get_time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Input projection
        h = self.input_conv(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for i, (blocks, attention, downsample) in enumerate(
            zip(self.encoder_blocks, self.encoder_attentions, self.encoder_downsample)
        ):
            # Residual blocks
            for block in blocks:
                h = block(h, time_emb, condition)
            
            # Attention
            h = attention(h)
            
            # Store for skip connection
            skip_connections.append(h)
            
            # Downsample
            h = downsample(h)
        
        # Middle block
        h = self.middle_block1(h, time_emb, condition)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_emb, condition)
        
        # Decoder
        for i, (upsample, blocks, attention) in enumerate(
            zip(self.decoder_upsample, self.decoder_blocks, self.decoder_attentions)
        ):
            # Upsample
            h = upsample(h)
            
            # Skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            
            # Residual blocks
            for block in blocks:
                h = block(h, time_emb, condition)
            
            # Attention
            h = attention(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h

class ScaledEEGDataset(Dataset):
    """Dataset for Scaled EEG LDM training"""
    
    def __init__(self, embeddings_file="crell_embeddings_20250622_173213.pkl", 
                 split="train", target_size=64):  # Increased to 64x64
        self.split = split
        self.target_size = target_size
        
        # Load Crell embeddings and data
        with open(embeddings_file, 'rb') as f:
            emb_data = pickle.load(f)

        # Extract embeddings and labels directly
        all_embeddings = emb_data['embeddings']
        all_labels = emb_data['labels']

        # Load original Crell data for stimulus images
        crell_data_file = "crell_processed_data_correct.pkl"
        with open(crell_data_file, 'rb') as f:
            crell_data = pickle.load(f)
        
        # Get stimulus images from validation set
        all_images = crell_data['validation']['images']
        
        # Split data for train/val since Crell only has validation data
        n_samples = len(all_embeddings)
        if split == "train":
            # Use first 80% for training
            end_idx = int(0.8 * n_samples)
            self.eeg_embeddings = all_embeddings[:end_idx]
            self.labels = all_labels[:end_idx]
            self.original_images = all_images[:end_idx]
        else:  # val/test
            # Use last 20% for validation/testing
            start_idx = int(0.8 * n_samples)
            self.eeg_embeddings = all_embeddings[start_idx:]
            self.labels = all_labels[start_idx:]
            self.original_images = all_images[start_idx:]

        print(f"📊 Loaded {split} data:")
        print(f"   EEG embeddings: {self.eeg_embeddings.shape}")
        print(f"   Labels: {len(self.labels)}")
        print(f"   Original images: {len(self.original_images)} images")
        
        # Process images to target size
        self.images = self._process_images()
        
        print(f"   Processed images: {self.images.shape}")
        print(f"   Target resolution: {target_size}x{target_size}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")
    
    def _process_images(self):
        """Process images to target format with higher resolution"""
        import numpy as np
        from PIL import Image
        
        processed_images = []
        
        for img in self.original_images:
            # Handle different image types
            if hasattr(img, 'mode'):  # PIL Image
                img_array = np.array(img)
            else:  # numpy array
                img_array = img
            
            # Convert to grayscale if RGB
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 3:  # RGB
                    img_array = np.mean(img_array, axis=2)
                elif img_array.shape[2] == 1:  # Single channel
                    img_array = img_array[:, :, 0]

            # Ensure 2D
            if len(img_array.shape) > 2:
                img_array = img_array.squeeze()

            # Resize to target size (higher resolution)
            if img_array.shape[0] != self.target_size or img_array.shape[1] != self.target_size:
                pil_img = Image.fromarray(img_array.astype(np.uint8))
                # Use LANCZOS for high-quality upsampling
                pil_resized = pil_img.resize((self.target_size, self.target_size), Image.LANCZOS)
                img_array = np.array(pil_resized)

            # Normalize to [-1, 1]
            img_array = img_array.astype(np.float32)
            if img_array.max() > 1.0:  # If not already normalized
                img_array = img_array / 255.0  # [0, 1]
            img_array = (img_array - 0.5) * 2  # [-1, 1]
            
            processed_images.append(img_array)
        
        # Stack and add channel dimension
        images = np.array(processed_images)  # (N, H, W)
        images = images[:, None, :, :]  # (N, 1, H, W)
        
        return images
    
    def __len__(self):
        return len(self.eeg_embeddings)
    
    def __getitem__(self, idx):
        eeg_emb = torch.FloatTensor(self.eeg_embeddings[idx])
        image = torch.FloatTensor(self.images[idx])
        label = self.labels[idx]
        
        return eeg_emb, image, label
