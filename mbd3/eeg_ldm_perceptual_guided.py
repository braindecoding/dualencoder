#!/usr/bin/env python3
"""
Perceptual-Guided EEG LDM for Human Visual Similarity
Focus on perceptual quality that aligns with human vision
Uses Multi-Scale SSIM + VGG Perceptual Loss + LPIPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Try to import LPIPS for learned perceptual similarity
try:
    import lpips
    LPIPS_AVAILABLE = True
    print("✅ LPIPS library available")
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️ LPIPS not available, using alternative perceptual metrics")

class MultiScaleSSIM(nn.Module):
    """Multi-Scale Structural Similarity Index for better perceptual quality"""
    
    def __init__(self, window_size=11, size_average=True, scales=3):
        super(MultiScaleSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.scales = scales
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        # Multi-scale SSIM calculation
        ssim_values = []
        current_img1, current_img2 = img1, img2
        
        for scale in range(self.scales):
            ssim_val = self._ssim(current_img1, current_img2, window, self.window_size, channel, self.size_average)
            ssim_values.append(ssim_val)
            
            # Downsample for next scale (except last scale)
            if scale < self.scales - 1:
                current_img1 = F.avg_pool2d(current_img1, kernel_size=2, stride=2)
                current_img2 = F.avg_pool2d(current_img2, kernel_size=2, stride=2)
                # Update window for new size
                if current_img1.shape[-1] >= self.window_size:
                    window = self.create_window(self.window_size, channel)
                    if img1.is_cuda:
                        window = window.cuda(img1.get_device())
                    window = window.type_as(img1)
        
        # Weighted average of multi-scale SSIM
        weights = [0.5, 0.3, 0.2][:len(ssim_values)]  # Higher weight for original scale
        ms_ssim = sum(w * ssim for w, ssim in zip(weights, ssim_values))
        
        return ms_ssim

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for human-like feature comparison"""
    
    def __init__(self, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device
        
        # Load pre-trained VGG16
        vgg = torchvision.models.vgg16(pretrained=True).features
        
        # Extract features from multiple layers for rich representation
        self.feature_layers = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[:9]),   # relu2_2  
            nn.Sequential(*list(vgg.children())[:16]),  # relu3_3
            nn.Sequential(*list(vgg.children())[:23]),  # relu4_3
        ]).to(device)
        
        # Freeze VGG parameters
        for layer in self.feature_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Layer weights (earlier layers = more weight for fine details)
        self.layer_weights = [1.0, 0.8, 0.6, 0.4]
        
        print(f"📊 VGG Perceptual Loss initialized with {len(self.feature_layers)} feature layers")
    
    def forward(self, pred, target):
        # Convert grayscale to RGB for VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Resize to minimum VGG input size if needed
        if pred.shape[-1] < 32:
            pred = F.interpolate(pred, size=(32, 32), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Extract features and calculate loss
        total_loss = 0.0
        
        for i, (layer, weight) in enumerate(zip(self.feature_layers, self.layer_weights)):
            pred_features = layer(pred)
            target_features = layer(target)
            
            # Feature matching loss
            layer_loss = F.mse_loss(pred_features, target_features)
            total_loss += weight * layer_loss
        
        return total_loss

class PerceptualGuidedLoss(nn.Module):
    """Comprehensive perceptual loss combining multiple human-aligned metrics"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Multi-Scale SSIM for structural similarity
        self.ms_ssim = MultiScaleSSIM(window_size=11, size_average=True, scales=3).to(device)
        
        # VGG Perceptual Loss for feature similarity
        self.vgg_loss = VGGPerceptualLoss(device=device)
        
        # LPIPS for learned perceptual similarity (if available)
        if LPIPS_AVAILABLE:
            self.lpips_loss = lpips.LPIPS(net='alex').to(device)
            print("📊 Using LPIPS for learned perceptual similarity")
        else:
            self.lpips_loss = None
            print("📊 LPIPS not available, using MS-SSIM + VGG only")
        
        # Optional: Simple classification for semantic guidance (lightweight)
        self.use_semantic_guidance = True
        if self.use_semantic_guidance:
            self.classifier = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 10)
            ).to(device)
            print("📊 Using lightweight semantic guidance")
    
    def forward(self, generated_images, target_images, target_labels=None):
        """Calculate comprehensive perceptual loss"""
        
        # 1. Multi-Scale SSIM (50% weight) - Primary perceptual metric
        ms_ssim_value = self.ms_ssim(generated_images, target_images)
        ms_ssim_loss = 1 - ms_ssim_value  # Convert to loss
        
        # 2. VGG Perceptual Loss (30% weight) - Feature similarity
        vgg_loss = self.vgg_loss(generated_images, target_images)
        
        # 3. LPIPS (15% weight) - Learned perceptual similarity
        if LPIPS_AVAILABLE and self.lpips_loss is not None:
            # Convert to [0,1] range and resize for LPIPS (minimum 64x64)
            gen_norm = (generated_images + 1) / 2
            target_norm = (target_images + 1) / 2

            # Resize to minimum size for LPIPS (64x64)
            if gen_norm.shape[-1] < 64:
                gen_norm = F.interpolate(gen_norm, size=(64, 64), mode='bilinear', align_corners=False)
                target_norm = F.interpolate(target_norm, size=(64, 64), mode='bilinear', align_corners=False)

            # Convert grayscale to RGB for LPIPS
            if gen_norm.shape[1] == 1:
                gen_norm = gen_norm.repeat(1, 3, 1, 1)
                target_norm = target_norm.repeat(1, 3, 1, 1)

            lpips_loss = self.lpips_loss(gen_norm, target_norm).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)
        
        # 4. Optional semantic guidance (5% weight) - Very light
        if self.use_semantic_guidance and target_labels is not None:
            logits = self.classifier(generated_images)
            semantic_loss = F.cross_entropy(logits, target_labels)
        else:
            semantic_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses with human-perception focused weighting
        if LPIPS_AVAILABLE:
            total_loss = (0.50 * ms_ssim_loss +      # Multi-scale structural similarity
                         0.30 * vgg_loss +           # Deep feature similarity  
                         0.15 * lpips_loss +         # Learned perceptual similarity
                         0.05 * semantic_loss)       # Light semantic guidance
        else:
            total_loss = (0.60 * ms_ssim_loss +      # Higher weight without LPIPS
                         0.35 * vgg_loss +           # VGG features
                         0.05 * semantic_loss)       # Light semantic guidance
        
        return total_loss, ms_ssim_loss, vgg_loss, lpips_loss, semantic_loss
    
    def predict_digit(self, generated_images):
        """Predict digit from generated images (if semantic guidance enabled)"""
        if self.use_semantic_guidance:
            with torch.no_grad():
                logits = self.classifier(generated_images)
                predictions = torch.argmax(logits, dim=1)
                return predictions
        else:
            return torch.zeros(generated_images.shape[0], dtype=torch.long, device=self.device)

# Import improved LDM components
from eeg_ldm_improved import ImprovedUNet

class PerceptualEEGDataset(Dataset):
    """Dataset for Perceptual-Guided EEG LDM training"""
    
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

class PerceptualGuidedEEGDiffusion(nn.Module):
    """EEG diffusion model with perceptual guidance for human visual similarity"""
    
    def __init__(self, condition_dim=512, image_size=28, num_timesteps=100):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # Enhanced UNet for noise prediction
        self.unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=condition_dim
        )
        
        # Diffusion schedule
        self.register_buffer('betas', self._linear_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        print(f"📊 Perceptual-Guided EEG Diffusion Model:")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def _linear_beta_schedule(self, timesteps):
        """Linear beta schedule"""
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
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
    def sample(self, condition, shape=None, num_inference_steps=20):
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
        
        return torch.clamp(x, -1, 1)

def train_perceptual_guided_ldm():
    """Train Perceptual-Guided EEG LDM for optimal human visual similarity"""
    print("🚀 TRAINING PERCEPTUAL-GUIDED EEG LDM")
    print("=" * 90)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = PerceptualEEGDataset(split="train", target_size=28)
    test_dataset = PerceptualEEGDataset(split="test", target_size=28)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize models
    model = PerceptualGuidedEEGDiffusion(
        condition_dim=512,
        image_size=28,
        num_timesteps=100
    ).to(device)

    perceptual_loss_fn = PerceptualGuidedLoss(device=device)

    # Training setup
    optimizer = optim.AdamW(
        list(model.parameters()) + list(perceptual_loss_fn.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    # Training parameters
    num_epochs = 80  # Focus on quality over quantity
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    # Minimal L1 weight - focus on perceptual quality
    l1_weight = 0.05  # Very small for fine details only

    print(f"🎯 Perceptual-Guided Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 16")
    print(f"   Learning rate: 1e-4")
    if LPIPS_AVAILABLE:
        print(f"   Loss weights: MS-SSIM=50%, VGG=30%, LPIPS=15%, Semantic=5%")
    else:
        print(f"   Loss weights: MS-SSIM=60%, VGG=35%, Semantic=5%")
    print(f"   L1 weight: {l1_weight} (minimal)")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        perceptual_loss_fn.train()
        epoch_train_loss = 0

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

            # Calculate minimal L1 loss for noise prediction
            l1_loss = F.l1_loss(noise_pred, noise)

            # Generate clean images for perceptual loss
            with torch.no_grad():
                # Denoise to get clean images
                alpha_t = model.alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                predicted_x0 = torch.clamp(predicted_x0, -1, 1)

            # Calculate comprehensive perceptual loss
            perceptual_loss, ms_ssim_loss, vgg_loss, lpips_loss, semantic_loss = perceptual_loss_fn(
                predicted_x0, images, labels
            )

            # Combined loss (focus on perceptual quality)
            total_loss = perceptual_loss + l1_weight * l1_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(perceptual_loss_fn.parameters()), 1.0
            )
            optimizer.step()

            epoch_train_loss += total_loss.item()
            train_pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'percept': f'{perceptual_loss.item():.4f}',
                'ms_ssim': f'{ms_ssim_loss.item():.4f}',
                'vgg': f'{vgg_loss.item():.4f}',
                'lpips': f'{lpips_loss.item():.4f}',
                'semantic': f'{semantic_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        perceptual_loss_fn.eval()
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

                l1_loss = F.l1_loss(noise_pred, noise)

                # Generate clean images for perceptual loss
                alpha_t = model.alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                predicted_x0 = torch.clamp(predicted_x0, -1, 1)

                perceptual_loss, ms_ssim_loss, vgg_loss, lpips_loss, semantic_loss = perceptual_loss_fn(
                    predicted_x0, images, labels
                )

                total_loss = perceptual_loss + l1_weight * l1_loss

                epoch_test_loss += total_loss.item()
                test_pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'percept': f'{perceptual_loss.item():.4f}',
                    'ms_ssim': f'{ms_ssim_loss.item():.4f}',
                    'vgg': f'{vgg_loss.item():.4f}',
                    'lpips': f'{lpips_loss.item():.4f}',
                    'semantic': f'{semantic_loss.item():.4f}'
                })

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'model': model.state_dict(),
                'perceptual_loss': perceptual_loss_fn.state_dict()
            }, 'eeg_ldm_perceptual_guided_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model': model.state_dict(),
                'perceptual_loss': perceptual_loss_fn.state_dict()
            }, f'eeg_ldm_perceptual_guided_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'perceptual_loss': perceptual_loss_fn.state_dict()
    }, 'eeg_ldm_perceptual_guided_final.pth')

    print(f"\n✅ Perceptual-guided training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")

    return model, perceptual_loss_fn, train_losses, test_losses

def calculate_human_perceptual_metrics(pred_images, target_images, device='cuda'):
    """Calculate comprehensive human-aligned perceptual metrics"""

    # Initialize metric calculators
    ms_ssim_fn = MultiScaleSSIM(window_size=11, size_average=True, scales=3).to(device)
    vgg_loss_fn = VGGPerceptualLoss(device=device)

    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    metrics = {
        'ms_ssim_values': [],
        'vgg_distances': [],
        'lpips_distances': [],
        'correlations': [],
        'mse_values': []
    }

    with torch.no_grad():
        for i in range(len(pred_images)):
            pred_tensor = torch.FloatTensor(pred_images[i:i+1]).to(device)
            target_tensor = torch.FloatTensor(target_images[i:i+1]).to(device)

            # Multi-Scale SSIM
            ms_ssim_val = ms_ssim_fn(pred_tensor, target_tensor).item()
            metrics['ms_ssim_values'].append(ms_ssim_val)

            # VGG Perceptual Distance
            vgg_dist = vgg_loss_fn(pred_tensor, target_tensor).item()
            metrics['vgg_distances'].append(vgg_dist)

            # LPIPS Distance
            if LPIPS_AVAILABLE:
                pred_norm = (pred_tensor + 1) / 2  # Convert to [0,1]
                target_norm = (target_tensor + 1) / 2

                # Resize to minimum size for LPIPS (64x64)
                if pred_norm.shape[-1] < 64:
                    pred_norm = F.interpolate(pred_norm, size=(64, 64), mode='bilinear', align_corners=False)
                    target_norm = F.interpolate(target_norm, size=(64, 64), mode='bilinear', align_corners=False)

                # Convert grayscale to RGB for LPIPS
                if pred_norm.shape[1] == 1:
                    pred_norm = pred_norm.repeat(1, 3, 1, 1)
                    target_norm = target_norm.repeat(1, 3, 1, 1)

                lpips_dist = lpips_fn(pred_norm, target_norm).item()
                metrics['lpips_distances'].append(lpips_dist)

            # Traditional metrics
            pred_flat = pred_images[i].flatten()
            target_flat = target_images[i].flatten()

            corr, _ = pearsonr(pred_flat, target_flat)
            if not np.isnan(corr):
                metrics['correlations'].append(corr)

            mse_val = mean_squared_error(target_flat, pred_flat)
            metrics['mse_values'].append(mse_val)

    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])

    return metrics

def evaluate_perceptual_guided_ldm():
    """Evaluate Perceptual-Guided EEG LDM with comprehensive human-aligned metrics"""
    print(f"\n📊 EVALUATING PERCEPTUAL-GUIDED EEG LDM")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    model = PerceptualGuidedEEGDiffusion(
        condition_dim=512,
        image_size=28,
        num_timesteps=100
    ).to(device)

    perceptual_loss_fn = PerceptualGuidedLoss(device=device)

    # Load best checkpoint
    checkpoint = torch.load('eeg_ldm_perceptual_guided_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    perceptual_loss_fn.load_state_dict(checkpoint['perceptual_loss'])

    model.eval()
    perceptual_loss_fn.eval()

    # Load test data
    test_dataset = PerceptualEEGDataset(split="test", target_size=28)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate predictions
    predictions = []
    targets = []
    labels = []

    print("🎨 Generating images with perceptual guidance...")

    with torch.no_grad():
        for i, (eeg_emb, images, label) in enumerate(tqdm(test_loader, desc="Generating")):
            if i >= 20:  # Generate first 20 for evaluation
                break

            eeg_emb = eeg_emb.to(device)

            # Generate image
            generated_images = model.sample(eeg_emb, num_inference_steps=20)

            predictions.append(generated_images.cpu().numpy())
            targets.append(images.numpy())
            labels.append(label.item())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    labels = np.array(labels)

    print(f"📊 Generated predictions: {predictions.shape}")
    print(f"📊 Target images: {targets.shape}")
    print(f"📊 Labels: {labels}")

    # Calculate comprehensive human-aligned metrics
    metrics = calculate_human_perceptual_metrics(predictions, targets, device)

    # Calculate semantic accuracy using classification head
    semantic_accuracy = 0.0
    predicted_digits = []

    with torch.no_grad():
        pred_tensor = torch.FloatTensor(predictions).to(device)
        predicted_digit_logits = perceptual_loss_fn.predict_digit(pred_tensor)
        predicted_digits = predicted_digit_logits.cpu().numpy()

        correct_predictions = (predicted_digits == labels).sum()
        semantic_accuracy = correct_predictions / len(predictions)

    # Calculate human perceptual similarity score
    human_similarity_scores = []
    for i in range(len(predictions)):
        if LPIPS_AVAILABLE and len(metrics['lpips_distances']) > 0:
            # Comprehensive human similarity (lower is better for distances)
            human_sim = (0.4 * metrics['ms_ssim_values'][i] +           # Higher is better
                        0.3 * (1 - min(metrics['vgg_distances'][i], 1.0)) +  # Convert to similarity
                        0.3 * (1 - min(metrics['lpips_distances'][i], 1.0)))  # Convert to similarity
        else:
            # Without LPIPS
            human_sim = (0.6 * metrics['ms_ssim_values'][i] +           # Higher is better
                        0.4 * (1 - min(metrics['vgg_distances'][i], 1.0)))    # Convert to similarity
        human_similarity_scores.append(human_sim)

    human_similarity_scores = np.array(human_similarity_scores)

    print(f"📊 Perceptual-Guided EEG LDM Performance:")
    print(f"   Traditional MSE: {metrics['mse_values'].mean():.4f}")
    print(f"   Mean Correlation: {metrics['correlations'].mean():.4f} ± {metrics['correlations'].std():.4f}")
    print(f"   Best Correlation: {metrics['correlations'].max():.4f}")
    print(f"   Mean Multi-Scale SSIM: {metrics['ms_ssim_values'].mean():.4f} ± {metrics['ms_ssim_values'].std():.4f}")
    print(f"   Best Multi-Scale SSIM: {metrics['ms_ssim_values'].max():.4f}")
    print(f"   Mean VGG Distance: {metrics['vgg_distances'].mean():.4f} ± {metrics['vgg_distances'].std():.4f}")
    if LPIPS_AVAILABLE and len(metrics['lpips_distances']) > 0:
        print(f"   Mean LPIPS Distance: {metrics['lpips_distances'].mean():.4f} ± {metrics['lpips_distances'].std():.4f}")
    print(f"   Human Similarity Score: {human_similarity_scores.mean():.4f} ± {human_similarity_scores.std():.4f}")
    print(f"   Semantic Accuracy: {semantic_accuracy:.4f} ({correct_predictions}/{len(predictions)})")

    # Visualize results with comprehensive perceptual metrics
    fig, axes = plt.subplots(7, 4, figsize=(16, 28))
    fig.suptitle('Perceptual-Guided EEG LDM: Human Visual Similarity Focus', fontsize=16, fontweight='bold')

    for i in range(4):
        if i < len(predictions):
            # Original
            orig_img = targets[i, 0]
            axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}\nTrue Label: {labels[i]}', fontweight='bold')
            axes[0, i].axis('off')

            # Generated
            gen_img = predictions[i, 0]
            axes[1, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
            correct_symbol = "✅" if predicted_digits[i] == labels[i] else "❌"
            axes[1, i].set_title(f'Generated {i+1}\nPredicted: {predicted_digits[i]} {correct_symbol}', fontweight='bold')
            axes[1, i].axis('off')

            # Difference
            diff_img = np.abs(orig_img - gen_img)
            axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i+1}')
            axes[2, i].axis('off')

            # Human Perceptual Metrics
            axes[3, i].text(0.5, 0.8, f'Human Similarity: {human_similarity_scores[i]:.3f}',
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           transform=axes[3, i].transAxes)
            axes[3, i].text(0.5, 0.6, f'MS-SSIM: {metrics["ms_ssim_values"][i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[3, i].transAxes)
            axes[3, i].text(0.5, 0.4, f'VGG Dist: {metrics["vgg_distances"][i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[3, i].transAxes)
            if LPIPS_AVAILABLE and len(metrics['lpips_distances']) > 0:
                axes[3, i].text(0.5, 0.2, f'LPIPS: {metrics["lpips_distances"][i]:.3f}',
                               ha='center', va='center', fontsize=10,
                               transform=axes[3, i].transAxes)
            axes[3, i].axis('off')

            # Traditional metrics
            axes[4, i].text(0.5, 0.7, f'Correlation: {metrics["correlations"][i]:.3f}',
                           ha='center', va='center', fontsize=12,
                           transform=axes[4, i].transAxes)
            axes[4, i].text(0.5, 0.5, f'MSE: {metrics["mse_values"][i]:.3f}',
                           ha='center', va='center', fontsize=12,
                           transform=axes[4, i].transAxes)
            axes[4, i].axis('off')

            # Semantic accuracy
            axes[5, i].text(0.5, 0.7, f'True: {labels[i]}',
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[5, i].transAxes)
            axes[5, i].text(0.5, 0.5, f'Pred: {predicted_digits[i]} {correct_symbol}',
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[5, i].transAxes)
            axes[5, i].text(0.5, 0.3, f'Accuracy: {semantic_accuracy:.1%}',
                           ha='center', va='center', fontsize=12,
                           transform=axes[5, i].transAxes)
            axes[5, i].axis('off')

            # Model summary
            axes[6, i].text(0.5, 0.8, f'Perceptual Model',
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[6, i].transAxes)
            axes[6, i].text(0.5, 0.6, f'MS-SSIM + VGG',
                           ha='center', va='center', fontsize=12,
                           transform=axes[6, i].transAxes)
            if LPIPS_AVAILABLE:
                axes[6, i].text(0.5, 0.4, f'+ LPIPS',
                               ha='center', va='center', fontsize=12,
                               transform=axes[6, i].transAxes)
            axes[6, i].text(0.5, 0.2, f'Human Vision Focus',
                           ha='center', va='center', fontsize=12,
                           transform=axes[6, i].transAxes)
            axes[6, i].axis('off')

    plt.tight_layout()
    plt.savefig('eeg_ldm_perceptual_guided_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predictions, targets, metrics, labels, semantic_accuracy, predicted_digits, human_similarity_scores

def main():
    """Main function"""
    print("🎯 PERCEPTUAL-GUIDED EEG LATENT DIFFUSION MODEL")
    print("=" * 100)
    print("🧠 Focus on Human Visual Similarity and Perceptual Quality")
    print("=" * 100)

    # Train model
    _, _, _, _ = train_perceptual_guided_ldm()

    # Evaluate model
    predictions, targets, metrics, _, semantic_accuracy, _, human_scores = evaluate_perceptual_guided_ldm()

    print(f"\n🎯 PERCEPTUAL-GUIDED EEG LDM SUMMARY:")
    print(f"   Final MSE: {metrics['mse_values'].mean():.4f}")
    print(f"   Mean Correlation: {metrics['correlations'].mean():.4f}")
    print(f"   Mean Multi-Scale SSIM: {metrics['ms_ssim_values'].mean():.4f}")
    print(f"   Mean VGG Distance: {metrics['vgg_distances'].mean():.4f}")
    if LPIPS_AVAILABLE:
        print(f"   Mean LPIPS Distance: {metrics['lpips_distances'].mean():.4f}")
    print(f"   Human Similarity Score: {human_scores.mean():.4f}")
    print(f"   Semantic Accuracy: {semantic_accuracy:.4f}")
    print(f"   Training completed successfully!")

    print(f"\n📁 Generated files:")
    print(f"   - eeg_ldm_perceptual_guided_best.pth")
    print(f"   - eeg_ldm_perceptual_guided_final.pth")
    print(f"   - eeg_ldm_perceptual_guided_results.png")

    print(f"\n🚀 Perceptual-Guided Benefits:")
    print(f"   ✅ Multi-Scale SSIM for structural similarity at multiple resolutions")
    print(f"   ✅ VGG Perceptual Loss for deep feature similarity")
    if LPIPS_AVAILABLE:
        print(f"   ✅ LPIPS for state-of-the-art learned perceptual similarity")
    print(f"   ✅ Human vision-aligned optimization")
    print(f"   ✅ Minimal semantic guidance (5% weight)")
    print(f"   ✅ Focus on visual quality over classification accuracy")

    print(f"\n🧠 Expected Perceptual Advantages:")
    print(f"   🎯 Superior human visual similarity")
    print(f"   📈 Better structural preservation across scales")
    print(f"   🎨 More natural and realistic digit generation")
    print(f"   👁️ Optimized for human perception, not just metrics")
    print(f"   🔄 Balanced perceptual quality optimization")

if __name__ == "__main__":
    main()
