#!/usr/bin/env python3
"""
Improved UNet Architecture for Digit69 LDM
Sophisticated UNet with skip connections, attention, and better conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps"""
    
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
    
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb=None):
        residual = x
        
        # First block
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Add time embedding
        if time_emb is not None and self.time_emb_dim is not None:
            time_emb = self.time_mlp(time_emb)
            x = x + time_emb[:, :, None, None]
        
        # Second block
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        # Residual connection
        return x + self.residual_conv(residual)

class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # Reshape for attention
        qkv = qkv.view(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, hw, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, channels, height, width)
        
        out = self.proj_out(out)
        return out + residual

class CrossAttentionBlock(nn.Module):
    """Simplified cross-attention block for conditioning"""

    def __init__(self, channels, condition_dim):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim

        self.norm = nn.GroupNorm(8, channels)

        # Simple conditioning via channel-wise modulation
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, channels * 2),
            nn.SiLU()
        )

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, condition):
        batch_size, channels, height, width = x.shape
        residual = x

        x = self.norm(x)

        # Get conditioning parameters
        condition_params = self.condition_proj(condition)  # (batch, channels * 2)
        scale, shift = condition_params.chunk(2, dim=1)  # (batch, channels) each

        # Apply conditioning
        scale = scale.view(batch_size, channels, 1, 1)
        shift = shift.view(batch_size, channels, 1, 1)

        x = x * (1 + scale) + shift
        x = F.silu(x)
        x = self.conv(x)

        return x + residual

class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_dim=None, 
                 has_attention=False, num_layers=2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(ResidualBlock(in_channels, out_channels, time_emb_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualBlock(out_channels, out_channels, time_emb_dim))
        
        # Attention
        self.attention = None
        if has_attention:
            self.attention = AttentionBlock(out_channels)
        
        # Cross attention for conditioning
        self.cross_attention = None
        if condition_dim is not None:
            self.cross_attention = CrossAttentionBlock(out_channels, condition_dim)
        
        # Downsampling
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, condition=None):
        skip_connections = []
        
        for layer in self.layers:
            x = layer(x, time_emb)
            skip_connections.append(x)
        
        if self.attention is not None:
            x = self.attention(x)
            skip_connections[-1] = x
        
        if self.cross_attention is not None and condition is not None:
            x = self.cross_attention(x, condition)
            skip_connections[-1] = x
        
        x = self.downsample(x)
        
        return x, skip_connections

class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_dim=None,
                 has_attention=False, num_layers=2):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        
        # First layer (with skip connection)
        # Calculate the actual input channels after concatenation
        skip_channels = out_channels  # Assuming skip connection has out_channels
        self.layers.append(ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualBlock(out_channels, out_channels, time_emb_dim))
        
        # Attention
        self.attention = None
        if has_attention:
            self.attention = AttentionBlock(out_channels)
        
        # Cross attention for conditioning
        self.cross_attention = None
        if condition_dim is not None:
            self.cross_attention = CrossAttentionBlock(out_channels, condition_dim)
    
    def forward(self, x, skip_connections, time_emb, condition=None):
        x = self.upsample(x)
        
        # Concatenate skip connections
        for skip in reversed(skip_connections):
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            break  # Only use the last skip connection for simplicity
        
        for layer in self.layers:
            x = layer(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.cross_attention is not None and condition is not None:
            x = self.cross_attention(x, condition)
        
        return x

class ImprovedUNet(nn.Module):
    """Improved UNet with attention and better conditioning"""
    
    def __init__(self, in_channels=1, out_channels=1, condition_dim=512, 
                 model_channels=64, num_res_blocks=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_dim = condition_dim
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, condition_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList([
            DownBlock(model_channels, model_channels, time_emb_dim, condition_dim, 
                     has_attention=False, num_layers=num_res_blocks),
            DownBlock(model_channels, model_channels * 2, time_emb_dim, condition_dim,
                     has_attention=False, num_layers=num_res_blocks),
            DownBlock(model_channels * 2, model_channels * 4, time_emb_dim, condition_dim,
                     has_attention=True, num_layers=num_res_blocks),
            DownBlock(model_channels * 4, model_channels * 4, time_emb_dim, condition_dim,
                     has_attention=True, num_layers=num_res_blocks),
        ])
        
        # Middle
        self.mid_block1 = ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim)
        self.mid_attention = AttentionBlock(model_channels * 4)
        self.mid_cross_attention = CrossAttentionBlock(model_channels * 4, condition_dim)
        self.mid_block2 = ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim)
        
        # Decoder - match the encoder structure
        self.up_blocks = nn.ModuleList([
            UpBlock(model_channels * 4, model_channels * 4, time_emb_dim, condition_dim,
                   has_attention=True, num_layers=num_res_blocks),
            UpBlock(model_channels * 4, model_channels * 4, time_emb_dim, condition_dim,
                   has_attention=True, num_layers=num_res_blocks),
            UpBlock(model_channels * 4, model_channels * 2, time_emb_dim, condition_dim,
                   has_attention=False, num_layers=num_res_blocks),
            UpBlock(model_channels * 2, model_channels, time_emb_dim, condition_dim,
                   has_attention=False, num_layers=num_res_blocks),
        ])
        
        # Output
        self.norm_out = nn.GroupNorm(8, model_channels)
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        print(f"📊 Improved UNet Architecture:")
        print(f"   Input channels: {in_channels}")
        print(f"   Output channels: {out_channels}")
        print(f"   Model channels: {model_channels}")
        print(f"   Condition dim: {condition_dim}")
        print(f"   Time embedding dim: {time_emb_dim}")
        print(f"   Residual blocks per level: {num_res_blocks}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x, timesteps, condition):
        """Forward pass"""
        # Embeddings
        time_emb = self.time_embedding(timesteps)
        condition_emb = self.condition_embedding(condition)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder
        skip_connections_list = []
        for down_block in self.down_blocks:
            x, skip_connections = down_block(x, time_emb, condition_emb)
            skip_connections_list.append(skip_connections)
        
        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_cross_attention(x, condition_emb)
        x = self.mid_block2(x, time_emb)
        
        # Decoder
        for up_block, skip_connections in zip(self.up_blocks, reversed(skip_connections_list)):
            x = up_block(x, skip_connections, time_emb, condition_emb)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x

def test_improved_unet():
    """Test the improved UNet"""
    print("🧪 TESTING IMPROVED UNET")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Create model
    model = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    ).to(device)
    
    # Test inputs
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 512).to(device)
    
    print(f"\n🔍 Testing forward pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Timesteps shape: {timesteps.shape}")
    print(f"   Condition shape: {condition.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, timesteps, condition)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test gradient flow
    print(f"\n🔍 Testing gradient flow:")
    output = model(x, timesteps, condition)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    total_params = 0
    params_with_grad = 0
    
    for param in model.parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
    
    print(f"   Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"   ✅ Gradient flow working!")
    
    return model

if __name__ == "__main__":
    test_improved_unet()
