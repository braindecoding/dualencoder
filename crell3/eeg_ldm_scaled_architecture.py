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

class ScaledEEGDiffusion(nn.Module):
    """Scaled EEG diffusion model with enhanced architecture"""

    def __init__(self, condition_dim=512, image_size=64, num_timesteps=100):
        super().__init__()

        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps

        # Enhanced UNet with scaled architecture
        self.unet = ScaledUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=condition_dim
        )

        # Cosine beta schedule (better than linear)
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Scaled EEG Diffusion Model:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Image resolution: {image_size}x{image_size}")
        print(f"   Beta schedule: Cosine")

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule - better than linear for image generation"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(self, x0, t, noise=None):
        """Add noise to images"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x, t, condition):
        """Predict noise"""
        return self.unet(x, t, condition)

    @torch.no_grad()
    def sample(self, condition, shape=None, num_inference_steps=50):  # Increased from 20 to 50
        """Generate image from EEG condition with more inference steps"""
        if shape is None:
            shape = (condition.shape[0], 1, self.image_size, self.image_size)

        device = condition.device

        # Start from noise
        x = torch.randn(shape, device=device)

        # DDIM sampling with more steps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_batch = t.repeat(shape[0])

            # Predict noise
            noise_pred = self.unet(x, t_batch, condition)

            # DDIM step
            if i < len(timesteps) - 1:
                alpha_t = self.alphas_cumprod[t]
                alpha_t_next = self.alphas_cumprod[timesteps[i + 1]]

                x = torch.sqrt(alpha_t_next) * (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t) + \
                    torch.sqrt(1 - alpha_t_next) * noise_pred
            else:
                # Final step
                alpha_t = self.alphas_cumprod[t]
                x = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        return torch.clamp(x, -1, 1)

def train_scaled_ldm():
    """Train Scaled EEG LDM with enhanced architecture"""
    print("🚀 TRAINING SCALED EEG LDM")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets with higher resolution
    train_dataset = ScaledEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="train",
        target_size=64  # Increased resolution
    )
    test_dataset = ScaledEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size for larger model
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize scaled model
    model = ScaledEEGDiffusion(
        condition_dim=512,
        image_size=64,  # Higher resolution
        num_timesteps=100
    ).to(device)

    # Training setup with lower learning rate for larger model
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,  # Lower learning rate
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # Training parameters
    num_epochs = 200  # More epochs for better convergence
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"🎯 Scaled Architecture Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 8")
    print(f"   Learning rate: 5e-5")
    print(f"   Image resolution: 64x64")
    print(f"   Beta schedule: Cosine")
    print(f"   Inference steps: 50")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for eeg_emb, images, labels in train_pbar:
            eeg_emb = eeg_emb.to(device)
            images = images.to(device)

            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

            # Forward diffusion
            noisy_images, noise = model.forward_diffusion(images, t)

            # Predict noise
            noise_pred = model(noisy_images, t, eeg_emb)

            # Calculate loss (simple L1 loss for now)
            loss = F.l1_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_test_loss = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")

            for eeg_emb, images, labels in test_pbar:
                eeg_emb = eeg_emb.to(device)
                images = images.to(device)

                batch_size = images.shape[0]
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

                noisy_images, noise = model.forward_diffusion(images, t)
                noise_pred = model(noisy_images, t, eeg_emb)

                loss = F.l1_loss(noise_pred, noise)
                epoch_test_loss += loss.item()
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'eeg_ldm_scaled_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'eeg_ldm_scaled_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save(model.state_dict(), 'eeg_ldm_scaled_final.pth')

    print(f"\n✅ Scaled architecture training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")

    return model, train_losses, test_losses

@torch.no_grad()
def evaluate_scaled_ldm():
    """Evaluate Scaled EEG LDM with comprehensive metrics"""
    print("🔍 EVALUATING SCALED EEG LDM")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load test data with higher resolution
    test_dataset = ScaledEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load trained model
    model = ScaledEEGDiffusion(
        condition_dim=512,
        image_size=64,
        num_timesteps=100
    ).to(device)

    try:
        model.load_state_dict(torch.load('eeg_ldm_scaled_best.pth', map_location=device))
        print("✅ Loaded best scaled model")
    except FileNotFoundError:
        print("❌ Best model not found, using random weights")

    model.eval()

    # Generate predictions
    predictions = []
    targets = []
    labels = []
    correlations = []

    print("🎯 Generating predictions with scaled architecture...")

    for i, (eeg_emb, image, label) in enumerate(tqdm(test_loader, desc="Generating")):
        eeg_emb = eeg_emb.to(device)
        image = image.to(device)

        # Generate image with more inference steps
        generated = model.sample(eeg_emb, num_inference_steps=50)

        # Store results
        pred_img = generated[0, 0].cpu().numpy()
        target_img = image[0, 0].cpu().numpy()

        predictions.append(pred_img)
        targets.append(target_img)
        labels.append(label)

        # Calculate correlation
        pred_flat = pred_img.flatten()
        target_flat = target_img.flatten()

        if len(np.unique(pred_flat)) > 1 and len(np.unique(target_flat)) > 1:
            corr, _ = pearsonr(pred_flat, target_flat)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)

    predictions = np.array(predictions)
    targets = np.array(targets)
    correlations = np.array(correlations)

    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mean_corr = correlations.mean()
    std_corr = correlations.std()

    print(f"\n📊 Scaled Architecture Results:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Mean Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")
    print(f"   Resolution: 64x64 (4x higher than baseline)")

    # Letter mapping for display
    letter_mapping = {0: 'a', 1: 'd', 2: 'e', 3: 'f', 4: 'j', 5: 'n', 6: 'o', 7: 's', 8: 't', 9: 'v'}

    # Visualize results
    n_samples = min(5, len(predictions))
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        # Original
        orig_img = targets[i]
        axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
        label_val = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
        true_letter = letter_mapping[label_val]
        axes[0, i].set_title(f'Original {i+1}\nLetter: {true_letter} ({label_val})', fontweight='bold')
        axes[0, i].axis('off')

        # Generated
        gen_img = predictions[i]
        axes[1, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Generated {i+1}\nCorr: {correlations[i]:.3f}', fontweight='bold')
        axes[1, i].axis('off')

        # Metrics
        axes[2, i].text(0.5, 0.7, f'MSE: {mean_squared_error(targets[i].flatten(), predictions[i].flatten()):.4f}',
                       ha='center', va='center', fontsize=10, transform=axes[2, i].transAxes)
        axes[2, i].text(0.5, 0.5, f'Corr: {correlations[i]:.4f}',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       transform=axes[2, i].transAxes)
        axes[2, i].text(0.5, 0.3, f'Resolution: 64x64',
                       ha='center', va='center', fontsize=10, transform=axes[2, i].transAxes)
        axes[2, i].set_xlim(0, 1)
        axes[2, i].set_ylim(0, 1)
        axes[2, i].axis('off')

    plt.suptitle('Scaled EEG LDM: Enhanced Architecture Results\n64x64 Resolution, 2M+ Parameters, Attention Mechanisms',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eeg_ldm_scaled_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Scaled architecture evaluation completed!")
    print(f"📊 Results saved to: eeg_ldm_scaled_results.png")

    return predictions, targets, correlations, labels

if __name__ == "__main__":
    print("🏗️ SCALED EEG LDM ARCHITECTURE")
    print("=" * 80)
    print("🚀 Enhanced UNet with Attention Mechanisms and Higher Resolution")
    print("=" * 80)

    print(f"\n🎯 Scaled Architecture Features:")
    print(f"   ✅ Enhanced UNet channels: [64, 128, 256, 512, 768]")
    print(f"   ✅ Self-attention mechanisms at high resolutions")
    print(f"   ✅ Residual blocks with time + condition embedding")
    print(f"   ✅ Higher resolution: 64x64 (4x improvement)")
    print(f"   ✅ Cosine beta schedule (better than linear)")
    print(f"   ✅ More inference steps: 50 (vs 20)")
    print(f"   ✅ Target 2M+ parameters (vs 976K)")

    print(f"\n🧠 Expected Scaled Advantages:")
    print(f"   🎯 Much better capacity for complex EEG-image mapping")
    print(f"   📈 Higher resolution for detailed letter generation")
    print(f"   🎨 Attention mechanisms for better feature learning")
    print(f"   🔄 Improved diffusion process with cosine schedule")
    print(f"   💪 Significantly reduced noise in generated images")

    # Train the scaled model
    print(f"\n🚀 Starting scaled architecture training...")
    model, train_losses, test_losses = train_scaled_ldm()

    # Evaluate the scaled model
    print(f"\n🔍 Starting scaled architecture evaluation...")
    predictions, targets, correlations, labels = evaluate_scaled_ldm()

    print(f"\n🎯 SCALED ARCHITECTURE TRAINING COMPLETED!")
    print(f"🚀 Expected major improvements in image quality and reduced noise!")
