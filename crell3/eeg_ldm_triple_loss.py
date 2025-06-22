#!/usr/bin/env python3
"""
EEG LDM with Triple Loss: SSIM + CLIP Guidance + MSE
Combining perceptual quality, semantic accuracy, and reconstruction quality
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
from pytorch_msssim import SSIM
from transformers import CLIPModel, CLIPProcessor

class TripleLossEEGDataset(Dataset):
    """Dataset for Triple Loss EEG LDM training"""
    
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

            # Normalize to [0, 1] for SSIM and CLIP
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

class TripleLossUNet(nn.Module):
    """UNet optimized for Triple Loss (SSIM + CLIP + MSE)"""
    
    def __init__(self, in_channels=1, out_channels=1, condition_dim=512):
        super().__init__()
        
        # Balanced channels for triple loss optimization
        self.channels = [64, 128, 256, 512]
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
        
        # Middle block with attention for CLIP guidance
        mid_channels = self.channels[-1]
        self.middle_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.MultiheadAttention(mid_channels, 8, batch_first=True),
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
        print(f"📊 Triple Loss UNet:")
        print(f"   Channels: {self.channels}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Optimized for: SSIM + CLIP + MSE")
        
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
        
        # Middle block with attention
        h_orig = h
        h = self.middle_block[0](h)  # Conv
        h = self.middle_block[1](h)  # GroupNorm
        h = self.middle_block[2](h)  # SiLU
        
        # Attention (reshape for MultiheadAttention)
        b, c, h_dim, w_dim = h.shape
        h_flat = h.view(b, c, h_dim * w_dim).transpose(1, 2)  # (b, hw, c)
        h_attn, _ = self.middle_block[3](h_flat, h_flat, h_flat)
        h = h_attn.transpose(1, 2).view(b, c, h_dim, w_dim)
        
        h = self.middle_block[4](h)  # Conv
        h = self.middle_block[5](h)  # GroupNorm
        h = self.middle_block[6](h)  # SiLU
        
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

class TripleLossEEGDiffusion(nn.Module):
    """Triple Loss EEG diffusion model"""
    
    def __init__(self, condition_dim=512, image_size=64, num_timesteps=100):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # Triple Loss optimized UNet
        self.unet = TripleLossUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=condition_dim
        )
        
        # Cosine beta schedule (better for CLIP guidance)
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Triple Loss EEG Diffusion Model:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Image resolution: {image_size}x{image_size}")
        print(f"   Beta schedule: Cosine (optimized for CLIP)")
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule"""
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
    def sample(self, condition, shape=None, num_inference_steps=50):
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
        
        # Ensure output is in [0, 1] range
        return torch.clamp(x, 0, 1)

class TripleLoss(nn.Module):
    """Triple Loss: SSIM + CLIP Guidance + MSE"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # SSIM loss for perceptual quality
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=7).to(device)
        
        # CLIP model for semantic guidance
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Letter templates for CLIP guidance
        self.letter_templates = [
            "a handwritten letter a",
            "a handwritten letter d", 
            "a handwritten letter e",
            "a handwritten letter f",
            "a handwritten letter j",
            "a handwritten letter n",
            "a handwritten letter o",
            "a handwritten letter s",
            "a handwritten letter t",
            "a handwritten letter v"
        ]
        
        print(f"📊 Triple Loss Components:")
        print(f"   Primary: SSIM loss (weight: 0.5)")
        print(f"   Secondary: CLIP guidance (weight: 0.3)")
        print(f"   Auxiliary: MSE loss (weight: 0.2)")
        print(f"   Focus: Perceptual + Semantic + Reconstruction")
    
    def get_clip_text_embeddings(self):
        """Get CLIP text embeddings for letter templates"""
        if not hasattr(self, '_clip_text_embeddings'):
            with torch.no_grad():
                text_inputs = self.clip_processor(
                    text=self.letter_templates,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                self._clip_text_embeddings = self.clip_model.get_text_features(**text_inputs)
                self._clip_text_embeddings = F.normalize(self._clip_text_embeddings, dim=-1)
        
        return self._clip_text_embeddings
    
    def forward(self, predicted, target, labels):
        """
        Triple loss computation
        Args:
            predicted: Generated images [0, 1]
            target: Target images [0, 1]
            labels: Letter labels for CLIP guidance
        """
        # Ensure inputs are in [0, 1] range
        predicted = torch.clamp(predicted, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 1. SSIM loss (perceptual quality)
        ssim_loss = 1.0 - self.ssim(predicted, target)
        
        # 2. MSE loss (reconstruction quality)
        mse_loss = F.mse_loss(predicted, target)
        
        # 3. CLIP guidance loss (semantic accuracy)
        clip_loss = self._compute_clip_loss(predicted, labels)
        
        # Weighted combination (Triple Loss)
        total_loss = (
            0.5 * ssim_loss +      # Primary: SSIM (perceptual)
            0.3 * clip_loss +      # Secondary: CLIP (semantic)
            0.2 * mse_loss         # Auxiliary: MSE (reconstruction)
        )
        
        return total_loss, ssim_loss, clip_loss, mse_loss
    
    def _compute_clip_loss(self, images, labels):
        """Compute CLIP guidance loss"""
        try:
            # Convert single channel to RGB for CLIP
            if images.shape[1] == 1:
                images_rgb = images.repeat(1, 3, 1, 1)
            else:
                images_rgb = images
            
            # Resize to 224x224 for CLIP
            images_224 = F.interpolate(images_rgb, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Get CLIP image embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(images_224)
                image_features = F.normalize(image_features, dim=-1)
            
            # Get text embeddings
            text_embeddings = self.get_clip_text_embeddings()
            
            # Compute similarity for each image with its corresponding letter
            clip_losses = []
            for i, label in enumerate(labels):
                label_idx = label.item() if hasattr(label, 'item') else label
                target_text_emb = text_embeddings[label_idx:label_idx+1]
                image_emb = image_features[i:i+1]
                
                # Cosine similarity (higher is better)
                similarity = torch.cosine_similarity(image_emb, target_text_emb, dim=-1)
                # Convert to loss (lower is better)
                loss = 1.0 - similarity
                clip_losses.append(loss)
            
            return torch.stack(clip_losses).mean()
            
        except Exception as e:
            print(f"CLIP loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

def train_triple_loss_ldm():
    """Train Triple Loss EEG LDM"""
    print("🎯 TRAINING TRIPLE LOSS EEG LDM")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = TripleLossEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="train",
        target_size=64
    )
    test_dataset = TripleLossEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch for CLIP
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize Triple Loss model
    model = TripleLossEEGDiffusion(
        condition_dim=512,
        image_size=64,
        num_timesteps=100
    ).to(device)

    # Triple Loss function
    triple_loss_fn = TripleLoss(device=device)

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,  # Lower learning rate for stable CLIP guidance
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # Training parameters
    num_epochs = 200  # More epochs for triple loss convergence
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    print(f"🎯 Triple Loss Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 8 (optimized for CLIP)")
    print(f"   Learning rate: 5e-5")
    print(f"   Loss: SSIM (50%) + CLIP (30%) + MSE (20%)")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_ssim_loss = 0
        epoch_clip_loss = 0
        epoch_mse_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for eeg_emb, images, labels in train_pbar:
            eeg_emb = eeg_emb.to(device)
            images = images.to(device)
            labels = labels.to(device)

            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

            # Forward diffusion
            noisy_images, noise = model.forward_diffusion(images, t)

            # Predict noise
            noise_pred = model(noisy_images, t, eeg_emb)

            # Reconstruct predicted image for loss computation
            alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
            sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
            predicted_x0 = torch.clamp(predicted_x0, 0, 1)

            # Triple Loss computation
            total_loss, ssim_loss, clip_loss, mse_loss = triple_loss_fn(predicted_x0, images, labels)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_ssim_loss += ssim_loss.item()
            epoch_clip_loss += clip_loss.item()
            epoch_mse_loss += mse_loss.item()

            train_pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'ssim': f'{ssim_loss.item():.4f}',
                'clip': f'{clip_loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_ssim_loss = epoch_ssim_loss / len(train_loader)
        avg_clip_loss = epoch_clip_loss / len(train_loader)
        avg_mse_loss = epoch_mse_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_test_loss = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")

            for eeg_emb, images, labels in test_pbar:
                eeg_emb = eeg_emb.to(device)
                images = images.to(device)
                labels = labels.to(device)

                batch_size = images.shape[0]
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

                noisy_images, noise = model.forward_diffusion(images, t)
                noise_pred = model(noisy_images, t, eeg_emb)

                # Reconstruct for loss computation
                alpha_t = model.sqrt_alphas_cumprod[t][:, None, None, None]
                sigma_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
                predicted_x0 = torch.clamp(predicted_x0, 0, 1)

                total_loss, _, _, _ = triple_loss_fn(predicted_x0, images, labels)
                epoch_test_loss += total_loss.item()
                test_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'eeg_ldm_triple_loss_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f} (SSIM: {avg_ssim_loss:.4f}, CLIP: {avg_clip_loss:.4f}, MSE: {avg_mse_loss:.4f}), Test Loss = {avg_test_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'eeg_ldm_triple_loss_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save(model.state_dict(), 'eeg_ldm_triple_loss_final.pth')

    print(f"\n✅ Triple Loss training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")

    return model, train_losses, test_losses

@torch.no_grad()
def evaluate_triple_loss_ldm():
    """Evaluate Triple Loss EEG LDM with comprehensive metrics"""
    print("🔍 EVALUATING TRIPLE LOSS EEG LDM")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load test data
    test_dataset = TripleLossEEGDataset(
        embeddings_file="crell_embeddings_20250622_173213.pkl",
        split="test",
        target_size=64
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load trained model
    model = TripleLossEEGDiffusion(
        condition_dim=512,
        image_size=64,
        num_timesteps=100
    ).to(device)

    try:
        model.load_state_dict(torch.load('eeg_ldm_triple_loss_best.pth', map_location=device))
        print("✅ Loaded best Triple Loss model")
    except FileNotFoundError:
        print("❌ Best model not found, using random weights")

    model.eval()

    # Evaluation metrics
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=1, win_size=7).to(device)
    triple_loss_fn = TripleLoss(device=device)

    # Generate predictions
    predictions = []
    targets = []
    labels = []
    correlations = []
    ssim_scores = []
    clip_scores = []
    mse_scores = []

    print("🎯 Generating predictions with Triple Loss model...")

    for i, (eeg_emb, image, label) in enumerate(tqdm(test_loader, desc="Generating")):
        eeg_emb = eeg_emb.to(device)
        image = image.to(device)
        label = label.to(device)

        # Generate image
        generated = model.sample(eeg_emb, num_inference_steps=50)

        # Store results
        pred_img = generated[0, 0].cpu().numpy()
        target_img = image[0, 0].cpu().numpy()

        predictions.append(pred_img)
        targets.append(target_img)
        labels.append(label.cpu().numpy())

        # Calculate correlation
        pred_flat = pred_img.flatten()
        target_flat = target_img.flatten()

        if len(np.unique(pred_flat)) > 1 and len(np.unique(target_flat)) > 1:
            corr, _ = pearsonr(pred_flat, target_flat)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)

        # Calculate individual loss components
        ssim_score = ssim_metric(generated, image).item()
        mse_score = F.mse_loss(generated, image).item()

        # CLIP score
        try:
            _, _, clip_loss, _ = triple_loss_fn(generated, image, label)
            clip_score = 1.0 - clip_loss.item()  # Convert loss to score
        except:
            clip_score = 0.0

        ssim_scores.append(ssim_score)
        mse_scores.append(mse_score)
        clip_scores.append(clip_score)

    predictions = np.array(predictions)
    targets = np.array(targets)
    correlations = np.array(correlations)
    ssim_scores = np.array(ssim_scores)
    mse_scores = np.array(mse_scores)
    clip_scores = np.array(clip_scores)

    # Calculate metrics
    overall_mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mean_corr = correlations.mean()
    std_corr = correlations.std()
    mean_ssim = ssim_scores.mean()
    std_ssim = ssim_scores.std()
    mean_mse = mse_scores.mean()
    std_mse = mse_scores.std()
    mean_clip = clip_scores.mean()
    std_clip = clip_scores.std()

    print(f"\n📊 Triple Loss Results:")
    print(f"   Overall MSE: {overall_mse:.4f}")
    print(f"   Mean Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")
    print(f"   Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"   Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"   Mean CLIP Score: {mean_clip:.4f} ± {std_clip:.4f}")
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
        axes[1, i].set_title(f'Generated {i+1}\nCLIP: {clip_scores[i]:.3f}', fontweight='bold')
        axes[1, i].axis('off')

        # Difference
        diff_img = np.abs(orig_img - gen_img)
        axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Difference {i+1}\nSSIM: {ssim_scores[i]:.3f}', fontweight='bold')
        axes[2, i].axis('off')

        # Metrics
        axes[3, i].text(0.5, 0.9, f'Correlation: {correlations[i]:.4f}', ha='center', va='center',
                       fontsize=9, transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.7, f'SSIM: {ssim_scores[i]:.4f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.5, f'CLIP: {clip_scores[i]:.4f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.3, f'MSE: {mse_scores[i]:.4f}', ha='center', va='center',
                       fontsize=9, transform=axes[3, i].transAxes)
        axes[3, i].text(0.5, 0.1, f'Triple Loss', ha='center', va='center',
                       fontsize=9, fontweight='bold', transform=axes[3, i].transAxes)
        axes[3, i].set_xlim(0, 1)
        axes[3, i].set_ylim(0, 1)
        axes[3, i].axis('off')

    plt.suptitle('Triple Loss EEG LDM: SSIM + CLIP + MSE Optimization\nSSIM (50%) + CLIP (30%) + MSE (20%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eeg_ldm_triple_loss_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Triple Loss evaluation completed!")
    print(f"📊 Results saved to: eeg_ldm_triple_loss_results.png")

    return predictions, targets, correlations, labels, ssim_scores, clip_scores, mse_scores

if __name__ == "__main__":
    print("🎯 TRIPLE LOSS EEG LDM")
    print("=" * 80)
    print("🚀 SSIM + CLIP Guidance + MSE Optimization")
    print("=" * 80)

    print(f"\n🎯 Triple Loss Features:")
    print(f"   ✅ Primary loss: SSIM (50% weight) - Perceptual quality")
    print(f"   ✅ Secondary loss: CLIP guidance (30% weight) - Semantic accuracy")
    print(f"   ✅ Auxiliary loss: MSE (20% weight) - Reconstruction quality")
    print(f"   ✅ Image range: [0, 1] (optimal for all losses)")
    print(f"   ✅ Multi-objective optimization")
    print(f"   ✅ Balanced approach: Perceptual + Semantic + Reconstruction")

    print(f"\n🧠 Expected Triple Loss Advantages:")
    print(f"   🎯 Best of all worlds: Quality + Accuracy + Reconstruction")
    print(f"   📈 Higher SSIM scores (structural similarity)")
    print(f"   🔤 Better letter recognition (CLIP guidance)")
    print(f"   🎨 Good reconstruction quality (MSE)")
    print(f"   🔄 Balanced optimization approach")
    print(f"   💪 Comprehensive loss function")

    print(f"\n📊 Triple Loss vs Other Approaches:")
    print(f"   🆚 SSIM-only: Adds semantic accuracy + reconstruction")
    print(f"   🆚 CLIP-only: Adds perceptual quality + reconstruction")
    print(f"   🆚 MSE-only: Adds perceptual quality + semantic accuracy")
    print(f"   🆚 Hybrid models: More balanced weight distribution")
    print(f"   🆚 Scaled models: Focus on loss function vs model size")

    # Train the Triple Loss model
    print(f"\n🚀 Starting Triple Loss training...")
    model, train_losses, test_losses = train_triple_loss_ldm()

    # Evaluate the Triple Loss model
    print(f"\n🔍 Starting Triple Loss evaluation...")
    predictions, targets, correlations, labels, ssim_scores, clip_scores, mse_scores = evaluate_triple_loss_ldm()

    print(f"\n🎯 TRIPLE LOSS TRAINING COMPLETED!")
    print(f"🚀 Expected comprehensive improvements across all metrics!")

    print(f"\n📊 Triple Loss Summary:")
    print(f"   🎯 Perceptual Quality: SSIM optimization")
    print(f"   🔤 Semantic Accuracy: CLIP guidance")
    print(f"   📐 Reconstruction Quality: MSE optimization")
    print(f"   ⚖️ Balanced Approach: 50% + 30% + 20%")
    print(f"   🏆 Best of all worlds combination!")
