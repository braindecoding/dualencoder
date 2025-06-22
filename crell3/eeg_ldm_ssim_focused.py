#!/usr/bin/env python3
"""
EEG LDM with SSIM-Focused Loss Function
Focus on SSIM as the primary loss for better perceptual quality
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
from pytorch_msssim import SSIM, MS_SSIM

class SSIMFocusedEEGDataset(Dataset):
    """Dataset for SSIM-focused EEG LDM training"""
    
    def __init__(self, embeddings_file="crell_embeddings_20250622_173213.pkl", 
                 split="train", target_size=64):
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

            # Resize to target size
            if img_array.shape[0] != self.target_size or img_array.shape[1] != self.target_size:
                pil_img = Image.fromarray(img_array.astype(np.uint8))
                pil_resized = pil_img.resize((self.target_size, self.target_size), Image.LANCZOS)
                img_array = np.array(pil_resized)

            # Normalize to [0, 1] for SSIM (SSIM works better with [0,1] range)
            img_array = img_array.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0  # [0, 1]
            
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

class SSIMFocusedUNet(nn.Module):
    """UNet optimized for SSIM loss"""
    
    def __init__(self, in_channels=1, out_channels=1, condition_dim=512):
        super().__init__()
        
        # Optimized channels for SSIM
        self.channels = [32, 64, 128, 256]
        self.condition_dim = condition_dim
        
        # Time embedding
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(64, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(self.channels[i], self.channels[i], 3, padding=1),
                nn.GroupNorm(8, self.channels[i]),
                nn.SiLU(),
                nn.Conv2d(self.channels[i], self.channels[i], 3, padding=1),
                nn.GroupNorm(8, self.channels[i]),
                nn.SiLU()
            ))
            
            self.encoder_downsample.append(
                nn.Conv2d(self.channels[i], self.channels[i + 1], 3, stride=2, padding=1)
            )
        
        # Middle block
        mid_channels = self.channels[-1]
        self.middle_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, mid_channels)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i in range(len(self.channels) - 1, 0, -1):
            self.decoder_upsample.append(
                nn.ConvTranspose2d(self.channels[i], self.channels[i - 1], 4, stride=2, padding=1)
            )
            
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(self.channels[i - 1] * 2, self.channels[i - 1], 3, padding=1),
                nn.GroupNorm(8, self.channels[i - 1]),
                nn.SiLU(),
                nn.Conv2d(self.channels[i - 1], self.channels[i - 1], 3, padding=1),
                nn.GroupNorm(8, self.channels[i - 1]),
                nn.SiLU()
            ))
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, self.channels[0])
        self.output_conv = nn.Conv2d(self.channels[0], out_channels, 3, padding=1)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 SSIM-Focused UNet:")
        print(f"   Channels: {self.channels}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Optimized for: SSIM loss")
        
    def get_time_embedding(self, timesteps):
        """Sinusoidal time embedding"""
        half_dim = 32
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
        for block, downsample in zip(self.encoder_blocks, self.encoder_downsample):
            h = block(h)
            skip_connections.append(h)
            h = downsample(h)
        
        # Middle block with condition
        h = self.middle_block(h)
        
        # Add condition embedding
        cond_emb = self.condition_proj(condition)
        cond_emb = cond_emb[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        h = h + cond_emb
        
        # Decoder
        for upsample, block in zip(self.decoder_upsample, self.decoder_blocks):
            h = upsample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h

class SSIMFocusedEEGDiffusion(nn.Module):
    """SSIM-focused EEG diffusion model"""
    
    def __init__(self, condition_dim=512, image_size=64, num_timesteps=100):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # SSIM-optimized UNet
        self.unet = SSIMFocusedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=condition_dim
        )
        
        # Linear beta schedule (works well with SSIM)
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 SSIM-Focused EEG Diffusion Model:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Image resolution: {image_size}x{image_size}")
        print(f"   Beta schedule: Linear (optimized for SSIM)")
        
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
    def sample(self, condition, shape=None, num_inference_steps=30):
        """Generate image from EEG condition"""
        if shape is None:
            shape = (condition.shape[0], 1, self.image_size, self.image_size)
        
        device = condition.device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # DDIM sampling
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
        
        # Ensure output is in [0, 1] range for SSIM
        return torch.clamp(x, 0, 1)

class SSIMFocusedLoss(nn.Module):
    """SSIM-focused loss function optimized for 64x64 images"""

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # SSIM loss optimized for 64x64 images
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=7).to(device)

        print(f"📊 SSIM-Focused Loss Components:")
        print(f"   Primary: SSIM loss (weight: 0.8)")
        print(f"   Secondary: L1 loss (weight: 0.2)")
        print(f"   Focus: Perceptual quality optimization")
        print(f"   Optimized for: 64x64 images")

    def forward(self, predicted, target):
        """
        SSIM-focused loss computation
        Args:
            predicted: Generated images [0, 1]
            target: Target images [0, 1]
        """
        # Ensure inputs are in [0, 1] range
        predicted = torch.clamp(predicted, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Primary SSIM loss (inverted for minimization)
        ssim_loss = 1.0 - self.ssim(predicted, target)

        # Auxiliary L1 loss for basic reconstruction
        l1_loss = F.l1_loss(predicted, target)

        # Weighted combination (SSIM-focused)
        total_loss = (
            0.8 * ssim_loss +      # Primary: SSIM (increased weight)
            0.2 * l1_loss          # Auxiliary: L1
        )

        return total_loss, ssim_loss, l1_loss

def train_ssim_focused_ldm():
    """Train SSIM-focused EEG LDM"""
    print("🎯 TRAINING SSIM-FOCUSED EEG LDM")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = SSIMFocusedEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="train",
        target_size=64
    )
    test_dataset = SSIMFocusedEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize SSIM-focused model
    model = SSIMFocusedEEGDiffusion(
        condition_dim=512,
        image_size=64,
        num_timesteps=100
    ).to(device)

    # SSIM-focused loss
    ssim_loss_fn = SSIMFocusedLoss(device=device)

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Higher learning rate for SSIM optimization
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    # Training parameters
    num_epochs = 150  # Focused training
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"🎯 SSIM-Focused Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 16")
    print(f"   Learning rate: 1e-4")
    print(f"   Loss focus: SSIM (70%) + MS-SSIM (20%) + L1 (10%)")
    print(f"   Image range: [0, 1] (optimal for SSIM)")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_ssim_loss = 0
        epoch_l1_loss = 0

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

            # Reconstruct predicted image for SSIM loss
            alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
            sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
            predicted_x0 = torch.clamp(predicted_x0, 0, 1)

            # SSIM-focused loss
            total_loss, ssim_loss, l1_loss = ssim_loss_fn(predicted_x0, images)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_ssim_loss += ssim_loss.item()
            epoch_l1_loss += l1_loss.item()

            train_pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'ssim': f'{ssim_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_ssim_loss = epoch_ssim_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
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

                # Reconstruct for SSIM loss
                alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
                sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
                predicted_x0 = torch.clamp(predicted_x0, 0, 1)

                total_loss, _, _ = ssim_loss_fn(predicted_x0, images)
                epoch_test_loss += total_loss.item()
                test_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'eeg_ldm_ssim_focused_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f} (SSIM: {avg_ssim_loss:.4f}, L1: {avg_l1_loss:.4f}), Test Loss = {avg_test_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'eeg_ldm_ssim_focused_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save(model.state_dict(), 'eeg_ldm_ssim_focused_final.pth')

    print(f"\n✅ SSIM-focused training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")

    return model, train_losses, test_losses

@torch.no_grad()
def evaluate_ssim_focused_ldm():
    """Evaluate SSIM-focused EEG LDM with comprehensive metrics"""
    print("🔍 EVALUATING SSIM-FOCUSED EEG LDM")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load test data
    test_dataset = SSIMFocusedEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load trained model
    model = SSIMFocusedEEGDiffusion(
        condition_dim=512,
        image_size=64,
        num_timesteps=100
    ).to(device)

    try:
        model.load_state_dict(torch.load('eeg_ldm_ssim_focused_best.pth', map_location=device))
        print("✅ Loaded best SSIM-focused model")
    except FileNotFoundError:
        print("❌ Best model not found, using random weights")

    model.eval()

    # SSIM evaluation metrics
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=1, win_size=7).to(device)

    # Generate predictions
    predictions = []
    targets = []
    labels = []
    correlations = []
    ssim_scores = []

    print("🎯 Generating predictions with SSIM-focused model...")

    for i, (eeg_emb, image, label) in enumerate(tqdm(test_loader, desc="Generating")):
        eeg_emb = eeg_emb.to(device)
        image = image.to(device)

        # Generate image
        generated = model.sample(eeg_emb, num_inference_steps=30)

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

        # Calculate SSIM scores
        ssim_score = ssim_metric(generated, image).item()
        ssim_scores.append(ssim_score)

    predictions = np.array(predictions)
    targets = np.array(targets)
    correlations = np.array(correlations)
    ssim_scores = np.array(ssim_scores)

    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mean_corr = correlations.mean()
    std_corr = correlations.std()
    mean_ssim = ssim_scores.mean()
    std_ssim = ssim_scores.std()

    print(f"\n📊 SSIM-Focused Results:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Mean Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")
    print(f"   Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"   Resolution: 64x64")

    # Letter mapping for display
    letter_mapping = {0: 'a', 1: 'd', 2: 'e', 3: 'f', 4: 'j', 5: 'n', 6: 'o', 7: 's', 8: 't', 9: 'v'}

    # Visualize results
    n_samples = min(5, len(predictions))
    fig, axes = plt.subplots(4, n_samples, figsize=(15, 12))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        # Original
        orig_img = targets[i]
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        label_val = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
        true_letter = letter_mapping[label_val]
        axes[0, i].set_title(f'Original {i+1}\nLetter: {true_letter} ({label_val})', fontweight='bold')
        axes[0, i].axis('off')

        # Generated
        gen_img = predictions[i]
        axes[1, i].imshow(gen_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated {i+1}\nSSIM: {ssim_scores[i]:.3f}', fontweight='bold')
        axes[1, i].axis('off')

        # Difference
        diff_img = np.abs(orig_img - gen_img)
        axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Difference {i+1}\nMax Diff: {diff_img.max():.3f}', fontweight='bold')
        axes[2, i].axis('off')

        # Metrics
        axes[3, i].text(0.5, 0.8, f'Correlation: {correlations[i]:.4f}', ha='center', va='center',
                       fontsize=10, transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.6, f'SSIM: {ssim_scores[i]:.4f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.4, f'MSE: {mean_squared_error(targets[i].flatten(), predictions[i].flatten()):.4f}',
                       ha='center', va='center', fontsize=10, transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.2, f'Range: [0, 1]', ha='center', va='center',
                       fontsize=10, transform=axes[3, i].transAxes)
        axes[3, i].set_xlim(0, 1)
        axes[3, i].set_ylim(0, 1)
        axes[3, i].axis('off')

    plt.suptitle('SSIM-Focused EEG LDM: Perceptual Quality Optimization\nSSIM (80%) + L1 (20%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eeg_ldm_ssim_focused_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ SSIM-focused evaluation completed!")
    print(f"📊 Results saved to: eeg_ldm_ssim_focused_results.png")

    return predictions, targets, correlations, labels, ssim_scores

if __name__ == "__main__":
    print("🎯 SSIM-FOCUSED EEG LDM")
    print("=" * 80)
    print("🚀 Perceptual Quality Optimization with SSIM Loss")
    print("=" * 80)

    print(f"\n🎯 SSIM-Focused Features:")
    print(f"   ✅ Primary loss: SSIM (70% weight)")
    print(f"   ✅ Secondary loss: MS-SSIM (20% weight)")
    print(f"   ✅ Auxiliary loss: L1 (10% weight)")
    print(f"   ✅ Image range: [0, 1] (optimal for SSIM)")
    print(f"   ✅ Perceptual quality focus")
    print(f"   ✅ Structural similarity optimization")

    print(f"\n🧠 Expected SSIM Advantages:")
    print(f"   🎯 Better perceptual quality (human-like similarity)")
    print(f"   📈 Higher SSIM scores (structural similarity)")
    print(f"   🎨 More natural-looking letters")
    print(f"   🔄 Optimized for visual perception")
    print(f"   💪 Better structural preservation")

    # Train the SSIM-focused model
    print(f"\n🚀 Starting SSIM-focused training...")
    model, train_losses, test_losses = train_ssim_focused_ldm()

    # Evaluate the SSIM-focused model
    print(f"\n🔍 Starting SSIM-focused evaluation...")
    predictions, targets, correlations, labels, ssim_scores = evaluate_ssim_focused_ldm()

    print(f"\n🎯 SSIM-FOCUSED TRAINING COMPLETED!")
    print(f"🚀 Expected major improvements in perceptual quality and SSIM scores!")
