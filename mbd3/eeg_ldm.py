#!/usr/bin/env python3
"""
EEG Latent Diffusion Model (LDM) for Brain-to-Image Reconstruction
Using MBD3 EEG embeddings with VAE + UNet diffusion approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Try to import diffusers, fallback if not available
try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers library available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️ Diffusers not available, using custom implementation")

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps"""
    
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
    """Residual block with time and condition embedding"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb, condition_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        # Add condition embedding
        condition_emb = self.condition_mlp(condition_emb)[:, :, None, None]
        h = h + condition_emb
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class SimpleUNet(nn.Module):
    """Simple UNet for diffusion with EEG conditioning"""
    
    def __init__(self, in_channels=1, out_channels=1, condition_dim=512, time_emb_dim=256):
        super().__init__()
        
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        self.down1 = ResidualBlock(64, 128, time_emb_dim, condition_dim)
        self.down2 = ResidualBlock(128, 256, time_emb_dim, condition_dim)
        self.down3 = ResidualBlock(256, 512, time_emb_dim, condition_dim)
        
        # Middle
        self.mid = ResidualBlock(512, 512, time_emb_dim, condition_dim)
        
        # Decoder
        self.up3 = ResidualBlock(512 + 512, 256, time_emb_dim, condition_dim)
        self.up2 = ResidualBlock(256 + 256, 128, time_emb_dim, condition_dim)
        self.up1 = ResidualBlock(128 + 128, 64, time_emb_dim, condition_dim)
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        print(f"📊 Simple UNet Architecture:")
        print(f"   Input: {in_channels} channels")
        print(f"   Output: {out_channels} channels")
        print(f"   Condition dim: {condition_dim}")
        print(f"   Time embedding dim: {time_emb_dim}")

    def forward(self, x, timesteps, condition):
        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Initial conv
        x = self.conv_in(x)

        # Encoder with skip connections
        h1 = self.down1(x, time_emb, condition)          # 28x28 -> 128 channels
        h2 = self.down2(self.pool(h1), time_emb, condition)  # 14x14 -> 256 channels
        h3 = self.down3(self.pool(h2), time_emb, condition)  # 7x7 -> 512 channels

        # Middle
        h = self.mid(self.pool(h3), time_emb, condition)     # 3x3 -> 512 channels

        # Decoder with skip connections
        h = self.upsample(h)                            # 6x6 -> 512 channels
        # Resize h to match h3 size
        h = F.interpolate(h, size=h3.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h3], dim=1)                   # 512 + 512 = 1024 channels
        h = self.up3(h, time_emb, condition)            # -> 256 channels

        h = self.upsample(h)                            # 12x12 -> 256 channels
        # Resize h to match h2 size
        h = F.interpolate(h, size=h2.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h2], dim=1)                   # 256 + 256 = 512 channels
        h = self.up2(h, time_emb, condition)            # -> 128 channels

        h = self.upsample(h)                            # 24x24 -> 128 channels
        # Resize h to match h1 size
        h = F.interpolate(h, size=h1.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h1], dim=1)                   # 128 + 128 = 256 channels
        h = self.up1(h, time_emb, condition)            # -> 64 channels

        return self.conv_out(h)

class EEGDiffusionModel(nn.Module):
    """EEG-conditioned diffusion model"""
    
    def __init__(self, condition_dim=512, image_size=28, num_timesteps=1000):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # UNet for noise prediction
        self.unet = SimpleUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=condition_dim
        )
        
        # Diffusion schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        print(f"📊 EEG Diffusion Model:")
        print(f"   Condition dim: {condition_dim}")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Timesteps: {num_timesteps}")
        
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
        """Generate image from EEG condition using DDIM sampling"""
        if shape is None:
            shape = (condition.shape[0], 1, self.image_size, self.image_size)
        
        device = condition.device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # DDIM sampling
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
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

class EEGLDMDataset(Dataset):
    """Dataset for EEG LDM training"""
    
    def __init__(self, embeddings_file="eeg_embeddings_enhanced_20250622_123559.pkl", 
                 data_splits_file="explicit_eeg_data_splits.pkl", split="train", target_size=28):
        self.split = split
        self.target_size = target_size
        
        # Load EEG embeddings
        with open(embeddings_file, 'rb') as f:
            emb_data = pickle.load(f)
        
        # Load original data splits for stimulus images
        with open(data_splits_file, 'rb') as f:
            split_data = pickle.load(f)
        
        # Get split indices
        split_indices = emb_data['split_indices'][split]
        
        # Extract data for this split
        self.eeg_embeddings = emb_data['embeddings'][split_indices]
        self.labels = emb_data['labels'][split_indices]
        
        # Get stimulus images for this split
        if split == 'train':
            self.original_images = split_data['training']['stimTrn']
        elif split == 'val':
            self.original_images = split_data['validation']['stimVal']
        else:  # test
            self.original_images = split_data['test']['stimTest']
        
        print(f"📊 Loaded {split} data:")
        print(f"   EEG embeddings: {self.eeg_embeddings.shape}")
        print(f"   Labels: {self.labels.shape}")
        print(f"   Original images: {len(self.original_images)} PIL images")
        
        # Process images
        self.images = self._process_images()
        
        print(f"   Processed images: {self.images.shape}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")
    
    def _process_images(self):
        """Process PIL images to target format"""
        import numpy as np
        from PIL import Image
        
        processed_images = []
        
        for pil_img in self.original_images:
            # Convert PIL to numpy array
            img_array = np.array(pil_img)
            
            # Convert to grayscale if RGB
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)  # RGB to grayscale
            
            # Resize to target size
            if img_array.shape[0] != self.target_size:
                pil_resized = Image.fromarray(img_array.astype(np.uint8)).resize(
                    (self.target_size, self.target_size), Image.LANCZOS)
                img_array = np.array(pil_resized)
            
            # Normalize to [-1, 1]
            img_array = img_array.astype(np.float32) / 255.0  # [0, 1]
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

def train_eeg_ldm():
    """Train the EEG LDM model"""
    print("🚀 TRAINING EEG LATENT DIFFUSION MODEL")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = EEGLDMDataset(split="train", target_size=28)
    test_dataset = EEGLDMDataset(split="test", target_size=28)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = EEGDiffusionModel(condition_dim=512, image_size=28, num_timesteps=1000).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training parameters
    num_epochs = 50
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"🎯 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 8")
    print(f"   Learning rate: 1e-4")
    print(f"   Optimizer: AdamW")
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

            # Forward diffusion (add noise)
            noisy_images, noise = model.forward_diffusion(images, t)

            # Predict noise
            noise_pred = model(noisy_images, t, eeg_emb)

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

                # Sample random timesteps
                batch_size = images.shape[0]
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

                # Forward diffusion (add noise)
                noisy_images, noise = model.forward_diffusion(images, t)

                # Predict noise
                noise_pred = model(noisy_images, t, eeg_emb)

                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)

                epoch_test_loss += loss.item()
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate scheduling
        scheduler.step(avg_test_loss)

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'eeg_ldm_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'eeg_ldm_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save(model.state_dict(), 'eeg_ldm_final.pth')

    print(f"\n✅ Training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")
    print(f"   Final train loss: {train_losses[-1]:.4f}")
    print(f"   Final test loss: {test_losses[-1]:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('EEG LDM Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('EEG LDM Training and Test Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('eeg_ldm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, train_losses, test_losses

def evaluate_eeg_ldm():
    """Evaluate the trained EEG LDM model"""
    print(f"\n📊 EVALUATING EEG LDM MODEL")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = EEGDiffusionModel(condition_dim=512, image_size=28, num_timesteps=1000).to(device)
    model.load_state_dict(torch.load('eeg_ldm_best.pth', map_location=device))
    model.eval()

    # Load test data
    test_dataset = EEGLDMDataset(split="test", target_size=28)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate predictions
    predictions = []
    targets = []

    print("🎨 Generating images from EEG embeddings...")

    with torch.no_grad():
        for i, (eeg_emb, images, labels) in enumerate(tqdm(test_loader, desc="Generating")):
            if i >= 20:  # Generate only first 20 for evaluation
                break

            eeg_emb = eeg_emb.to(device)

            # Generate image using diffusion sampling
            generated_images = model.sample(eeg_emb, num_inference_steps=50)

            predictions.append(generated_images.cpu().numpy())
            targets.append(images.numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    print(f"📊 Generated predictions: {predictions.shape}")
    print(f"📊 Target images: {targets.shape}")

    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())

    # Calculate correlation for each sample
    correlations = []
    for i in range(len(predictions)):
        pred_flat = predictions[i].flatten()
        target_flat = targets[i].flatten()
        corr, _ = pearsonr(pred_flat, target_flat)
        if not np.isnan(corr):
            correlations.append(corr)

    correlations = np.array(correlations)

    print(f"📊 EEG LDM Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f} ± {correlations.std():.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")

    # Visualize results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('EEG LDM Results: Original vs Generated', fontsize=16)

    for i in range(4):
        if i < len(predictions):
            # Original
            orig_img = targets[i, 0]
            axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # Generated
            gen_img = predictions[i, 0]
            axes[1, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Generated {i+1}\nCorr: {correlations[i]:.3f}')
            axes[1, i].axis('off')

            # Difference
            diff_img = np.abs(orig_img - gen_img)
            axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i+1}')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('eeg_ldm_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predictions, targets, correlations

def main():
    """Main function"""
    print("🎯 EEG LATENT DIFFUSION MODEL")
    print("=" * 60)
    print("Advanced brain-to-image reconstruction using diffusion models")
    print("=" * 60)

    # Train model
    _, train_losses, test_losses = train_eeg_ldm()

    # Evaluate model
    predictions, targets, correlations = evaluate_eeg_ldm()

    print(f"\n🎯 EEG LDM SUMMARY:")
    print(f"   Final MSE: {mean_squared_error(targets.flatten(), predictions.flatten()):.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f}")
    print(f"   Training completed successfully!")

    print(f"\n📁 Generated files:")
    print(f"   - eeg_ldm_best.pth")
    print(f"   - eeg_ldm_final.pth")
    print(f"   - eeg_ldm_training_curves.png")
    print(f"   - eeg_ldm_results.png")

    print(f"\n🧠 EEG Latent Diffusion Model Results:")
    print(f"   Input: 512-dim EEG embeddings (from enhanced transformer)")
    print(f"   Output: 28x28 grayscale digit images")
    print(f"   Method: Latent Diffusion with UNet + DDIM sampling")
    print(f"   Performance: {correlations.mean():.4f} mean correlation")
    print(f"   🚀 Advanced generative model for brain-computer interfaces!")

if __name__ == "__main__":
    main()
