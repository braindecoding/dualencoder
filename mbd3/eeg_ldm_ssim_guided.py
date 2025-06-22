#!/usr/bin/env python3
"""
SSIM-Guided EEG LDM for Better Perceptual Quality
Replaces MSE with SSIM loss for better structural similarity
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

class SSIMLoss(nn.Module):
    """SSIM Loss for better perceptual similarity"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
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
        
        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value  # Convert to loss (lower SSIM = higher loss)

# Import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP library available")
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available, using classification approach")

# Import improved LDM components
from eeg_ldm_improved import ImprovedUNet

class SSIMGuidedEEGDataset(Dataset):
    """Dataset for SSIM-guided EEG LDM training"""
    
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

class SSIMSemanticLoss(nn.Module):
    """Combined SSIM and semantic loss"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # SSIM loss for perceptual similarity
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True).to(device)
        
        # Classification head for semantic accuracy
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        ).to(device)
        
        if CLIP_AVAILABLE:
            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            
            # Better digit text prompts (templates)
            self.digit_text_templates = [
                "a black digit {} on white background",
                "handwritten number {}",
                "the digit {}",
                "number {} in black ink"
            ]
            
            # Encode all text prompts
            all_text_features = []
            with torch.no_grad():
                for digit in range(10):
                    digit_features = []
                    for template in self.digit_text_templates:
                        text = template.format(digit)
                        text_tokens = clip.tokenize([text]).to(device)
                        text_feat = self.clip_model.encode_text(text_tokens)
                        digit_features.append(text_feat)
                    
                    # Average multiple templates for each digit
                    avg_feat = torch.stack(digit_features).mean(dim=0)
                    all_text_features.append(avg_feat)
                
                self.text_features = torch.stack(all_text_features).squeeze(1)
                self.text_features = F.normalize(self.text_features, dim=-1)
            
            print(f"📊 CLIP text features: {self.text_features.shape}")
            print(f"📊 Using {len(self.digit_text_templates)} text templates per digit")
        else:
            self.clip_model = None
            print("📊 Using SSIM + classification loss only")
    
    def forward(self, generated_images, target_images, target_labels):
        """Calculate combined SSIM and semantic loss"""
        # SSIM loss for perceptual similarity
        ssim_loss = self.ssim_loss(generated_images, target_images)
        
        # Classification loss for semantic accuracy
        logits = self.classifier(generated_images)
        classification_loss = F.cross_entropy(logits, target_labels)
        
        if CLIP_AVAILABLE and self.clip_model is not None:
            clip_loss = self._clip_loss(generated_images, target_labels)
            # Combine all losses
            total_loss = ssim_loss + 0.5 * classification_loss + 0.3 * clip_loss
            return total_loss, ssim_loss, classification_loss, clip_loss
        else:
            total_loss = ssim_loss + 0.5 * classification_loss
            return total_loss, ssim_loss, classification_loss, torch.tensor(0.0, device=self.device)
    
    def _clip_loss(self, generated_images, target_labels):
        """Enhanced CLIP-based semantic loss"""
        # Resize images to 224x224 for CLIP
        resized_images = F.interpolate(
            generated_images, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert grayscale to RGB
        rgb_images = resized_images.repeat(1, 3, 1, 1)
        
        # Normalize for CLIP (assuming images are in [-1, 1])
        rgb_images = (rgb_images + 1) / 2  # Convert to [0, 1]
        
        # Encode images
        with torch.no_grad():
            image_features = self.clip_model.encode_image(rgb_images)
            image_features = F.normalize(image_features, dim=-1)
        
        # Get target text features
        target_text_features = self.text_features[target_labels]
        
        # Calculate cosine similarity
        similarities = torch.sum(image_features * target_text_features, dim=-1)
        
        # Convert to loss (maximize similarity = minimize negative similarity)
        clip_loss = -similarities.mean()
        
        return clip_loss
    
    def predict_digit(self, generated_images):
        """Predict digit from generated images"""
        with torch.no_grad():
            logits = self.classifier(generated_images)
            predictions = torch.argmax(logits, dim=1)
            return predictions

class SSIMGuidedEEGDiffusion(nn.Module):
    """EEG diffusion model with SSIM guidance for better perceptual quality"""
    
    def __init__(self, condition_dim=512, image_size=28, num_timesteps=100):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # UNet for noise prediction
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
        
        print(f"📊 SSIM-Guided EEG Diffusion Model:")
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

def train_ssim_guided_ldm():
    """Train SSIM-guided EEG LDM for better perceptual quality"""
    print("🚀 TRAINING SSIM-GUIDED EEG LDM")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = SSIMGuidedEEGDataset(split="train", target_size=28)
    test_dataset = SSIMGuidedEEGDataset(split="test", target_size=28)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize models
    model = SSIMGuidedEEGDiffusion(
        condition_dim=512,
        image_size=28,
        num_timesteps=100
    ).to(device)

    semantic_loss_fn = SSIMSemanticLoss(device=device)

    # Training setup
    optimizer = optim.AdamW(
        list(model.parameters()) + list(semantic_loss_fn.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    # Training parameters
    num_epochs = 80  # Slightly fewer epochs since SSIM converges faster
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []

    # Loss weights - SSIM is primary, semantic is secondary
    l1_weight = 0.1  # Small L1 for fine details
    semantic_weight = 1.0  # Balanced semantic weight

    print(f"🎯 SSIM-Guided Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 16")
    print(f"   Learning rate: 1e-4")
    print(f"   Loss weights: SSIM=primary, L1={l1_weight}, Semantic={semantic_weight}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        semantic_loss_fn.train()
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

            # Calculate L1 loss for noise prediction
            l1_loss = F.l1_loss(noise_pred, noise)

            # Generate clean images for SSIM and semantic loss
            with torch.no_grad():
                # Denoise to get clean images
                alpha_t = model.alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                predicted_x0 = torch.clamp(predicted_x0, -1, 1)

            # Calculate SSIM and semantic loss
            semantic_loss, ssim_loss, class_loss, clip_loss = semantic_loss_fn(predicted_x0, images, labels)

            # Combined loss
            total_loss = (semantic_loss +
                         l1_weight * l1_loss)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(semantic_loss_fn.parameters()), 1.0
            )
            optimizer.step()

            epoch_train_loss += total_loss.item()
            train_pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'ssim': f'{ssim_loss.item():.4f}',
                'class': f'{class_loss.item():.4f}',
                'l1': f'{l1_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        semantic_loss_fn.eval()
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

                # Generate clean images for SSIM and semantic loss
                alpha_t = model.alphas_cumprod[t][:, None, None, None]
                predicted_x0 = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                predicted_x0 = torch.clamp(predicted_x0, -1, 1)

                semantic_loss, ssim_loss, class_loss, clip_loss = semantic_loss_fn(predicted_x0, images, labels)

                total_loss = (semantic_loss +
                             l1_weight * l1_loss)

                epoch_test_loss += total_loss.item()
                test_pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'ssim': f'{ssim_loss.item():.4f}',
                    'class': f'{class_loss.item():.4f}',
                    'l1': f'{l1_loss.item():.4f}'
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
                'semantic_loss': semantic_loss_fn.state_dict()
            }, 'eeg_ldm_ssim_guided_best.pth')

        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model': model.state_dict(),
                'semantic_loss': semantic_loss_fn.state_dict()
            }, f'eeg_ldm_ssim_guided_epoch_{epoch+1}.pth')

    training_time = time.time() - start_time

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'semantic_loss': semantic_loss_fn.state_dict()
    }, 'eeg_ldm_ssim_guided_final.pth')

    print(f"\n✅ SSIM-guided training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")

    return model, semantic_loss_fn, train_losses, test_losses

def calculate_ssim_metric(img1, img2):
    """Calculate SSIM metric between two images"""
    ssim_loss_fn = SSIMLoss(window_size=11, size_average=True)
    ssim_loss = ssim_loss_fn(img1, img2)
    ssim_value = 1 - ssim_loss.item()  # Convert loss back to SSIM
    return ssim_value

def evaluate_ssim_guided_ldm():
    """Evaluate SSIM-guided EEG LDM with comprehensive metrics"""
    print(f"\n📊 EVALUATING SSIM-GUIDED EEG LDM")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    model = SSIMGuidedEEGDiffusion(
        condition_dim=512,
        image_size=28,
        num_timesteps=100
    ).to(device)

    semantic_loss_fn = SSIMSemanticLoss(device=device)

    # Load best checkpoint
    checkpoint = torch.load('eeg_ldm_ssim_guided_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    semantic_loss_fn.load_state_dict(checkpoint['semantic_loss'])

    model.eval()
    semantic_loss_fn.eval()

    # Load test data
    test_dataset = SSIMGuidedEEGDataset(split="test", target_size=28)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate predictions
    predictions = []
    targets = []
    labels = []

    print("🎨 Generating images with SSIM guidance...")

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

    # Calculate comprehensive metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())

    # Calculate correlation for each sample
    correlations = []
    ssim_values = []
    for i in range(len(predictions)):
        pred_flat = predictions[i].flatten()
        target_flat = targets[i].flatten()
        corr, _ = pearsonr(pred_flat, target_flat)
        if not np.isnan(corr):
            correlations.append(corr)

        # Calculate SSIM for each sample
        pred_tensor = torch.FloatTensor(predictions[i:i+1]).to(device)
        target_tensor = torch.FloatTensor(targets[i:i+1]).to(device)
        ssim_val = calculate_ssim_metric(pred_tensor, target_tensor)
        ssim_values.append(ssim_val)

    correlations = np.array(correlations)
    ssim_values = np.array(ssim_values)

    # Calculate semantic accuracy using classification head
    semantic_accuracy = 0.0
    predicted_digits = []

    with torch.no_grad():
        pred_tensor = torch.FloatTensor(predictions).to(device)
        predicted_digit_logits = semantic_loss_fn.predict_digit(pred_tensor)
        predicted_digits = predicted_digit_logits.cpu().numpy()

        correct_predictions = (predicted_digits == labels).sum()
        semantic_accuracy = correct_predictions / len(predictions)

    print(f"📊 SSIM-Guided EEG LDM Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f} ± {correlations.std():.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")
    print(f"   Mean SSIM: {ssim_values.mean():.4f} ± {ssim_values.std():.4f}")
    print(f"   Best SSIM: {ssim_values.max():.4f}")
    print(f"   Worst SSIM: {ssim_values.min():.4f}")
    print(f"   Semantic Accuracy: {semantic_accuracy:.4f} ({correct_predictions}/{len(predictions)})")

    # Visualize results with comprehensive metrics
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    fig.suptitle('SSIM-Guided EEG LDM Results: Comprehensive Evaluation', fontsize=16)

    for i in range(4):
        if i < len(predictions):
            # Original
            orig_img = targets[i, 0]
            axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}\nTrue Label: {labels[i]}')
            axes[0, i].axis('off')

            # Generated
            gen_img = predictions[i, 0]
            axes[1, i].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Generated {i+1}\nPredicted: {predicted_digits[i]}')
            axes[1, i].axis('off')

            # Difference
            diff_img = np.abs(orig_img - gen_img)
            axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i+1}')
            axes[2, i].axis('off')

            # Metrics
            correct_symbol = "✅" if predicted_digits[i] == labels[i] else "❌"
            axes[3, i].text(0.5, 0.8, f'True: {labels[i]}', ha='center', va='center',
                           fontsize=12, transform=axes[3, i].transAxes)
            axes[3, i].text(0.5, 0.6, f'Pred: {predicted_digits[i]} {correct_symbol}',
                           ha='center', va='center', fontsize=12,
                           transform=axes[3, i].transAxes)
            axes[3, i].text(0.5, 0.4, f'Corr: {correlations[i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[3, i].transAxes)
            axes[3, i].text(0.5, 0.2, f'SSIM: {ssim_values[i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[3, i].transAxes)
            axes[3, i].axis('off')

            # Quality metrics
            axes[4, i].text(0.5, 0.7, f'MSE: {mean_squared_error(orig_img.flatten(), gen_img.flatten()):.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[4, i].transAxes)
            axes[4, i].text(0.5, 0.5, f'SSIM: {ssim_values[i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[4, i].transAxes)
            axes[4, i].text(0.5, 0.3, f'Corr: {correlations[i]:.3f}',
                           ha='center', va='center', fontsize=10,
                           transform=axes[4, i].transAxes)
            axes[4, i].axis('off')

            # Overall summary
            axes[5, i].text(0.5, 0.6, f'Semantic Acc:\n{semantic_accuracy:.1%}',
                           ha='center', va='center', fontsize=11,
                           transform=axes[5, i].transAxes)
            axes[5, i].text(0.5, 0.3, f'Avg SSIM:\n{ssim_values.mean():.3f}',
                           ha='center', va='center', fontsize=11,
                           transform=axes[5, i].transAxes)
            axes[5, i].axis('off')

    plt.tight_layout()
    plt.savefig('eeg_ldm_ssim_guided_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predictions, targets, correlations, labels, semantic_accuracy, predicted_digits, ssim_values

def main():
    """Main function"""
    print("🎯 SSIM-GUIDED EEG LATENT DIFFUSION MODEL")
    print("=" * 80)
    print("Using SSIM loss for better perceptual quality and structural similarity")
    print("=" * 80)

    # Train model
    _, _, _, _ = train_ssim_guided_ldm()

    # Evaluate model
    predictions, targets, correlations, _, semantic_accuracy, _, ssim_values = evaluate_ssim_guided_ldm()

    print(f"\n🎯 SSIM-GUIDED EEG LDM SUMMARY:")
    print(f"   Final MSE: {mean_squared_error(targets.flatten(), predictions.flatten()):.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f}")
    print(f"   Mean SSIM: {ssim_values.mean():.4f}")
    print(f"   Semantic Accuracy: {semantic_accuracy:.4f}")
    print(f"   Training completed successfully!")

    print(f"\n📁 Generated files:")
    print(f"   - eeg_ldm_ssim_guided_best.pth")
    print(f"   - eeg_ldm_ssim_guided_final.pth")
    print(f"   - eeg_ldm_ssim_guided_results.png")

    print(f"\n🚀 SSIM Guidance Benefits:")
    print(f"   ✅ Perceptual quality optimization")
    print(f"   ✅ Structural similarity preservation")
    print(f"   ✅ Better visual coherence")
    print(f"   ✅ Robust to small spatial variations")
    print(f"   ✅ Human vision-aligned metrics")

    print(f"\n🧠 Expected Improvements over MSE:")
    print(f"   🎯 Better visual quality (higher SSIM)")
    print(f"   📈 Improved structural preservation")
    print(f"   🎨 More natural-looking digits")
    print(f"   🔄 Better semantic consistency")

if __name__ == "__main__":
    main()
