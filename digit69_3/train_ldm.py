#!/usr/bin/env python3
"""
Train LDM for Digit69 Reconstruction
Simple training script for fMRI-conditioned digit generation
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
from pathlib import Path

class SimpleDiffusionModel(nn.Module):
    """Very simple diffusion model for proof of concept"""
    
    def __init__(self, condition_dim=512):
        super().__init__()
        
        # Simple encoder-decoder with conditioning
        self.condition_proj = nn.Linear(condition_dim, 64)
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 14x14
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 7x7
        
        # Middle with conditioning
        self.mid1 = nn.Conv2d(128 + 64, 128, 3, padding=1)  # +64 from condition
        self.mid2 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 14x14
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 28x28
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)  # 28x28
        
        # Time embedding (simplified)
        self.time_embed = nn.Embedding(1000, 64)

    def forward(self, x, t, condition):
        # Time embedding
        time_emb = self.time_embed(t)  # (B, 64)
        
        # Condition projection
        cond_emb = self.condition_proj(condition)  # (B, 64)
        
        # Combine time and condition
        combined_emb = time_emb + cond_emb  # (B, 64)
        
        # Encoder
        h1 = F.relu(self.enc1(x))      # (B, 32, 28, 28)
        h2 = F.relu(self.enc2(h1))     # (B, 64, 14, 14)
        h3 = F.relu(self.enc3(h2))     # (B, 128, 7, 7)
        
        # Add conditioning to middle layer
        # Broadcast condition to spatial dimensions
        cond_spatial = combined_emb[:, :, None, None].expand(-1, -1, 7, 7)  # (B, 64, 7, 7)
        h3_cond = torch.cat([h3, cond_spatial], dim=1)  # (B, 192, 7, 7)
        
        # Middle
        h = F.relu(self.mid1(h3_cond))
        h = F.relu(self.mid2(h))
        
        # Decoder
        h = F.relu(self.dec3(h))       # (B, 64, 14, 14)
        h = F.relu(self.dec2(h))       # (B, 32, 28, 28)
        h = self.dec1(h)               # (B, 1, 28, 28)
        
        return h

class Digit69Dataset(Dataset):
    """Dataset for digit69 embeddings"""
    
    def __init__(self, embeddings_file="digit69_embeddings.pkl", split="train"):
        self.split = split
        
        # Load embeddings
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.fmri_embeddings = data[split]['fmri_embeddings']
        self.original_images = data[split]['original_images']
        
        print(f"📊 Loaded {split} data:")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Original images: {self.original_images.shape}")
        
        # Convert images to grayscale and normalize
        if len(self.original_images.shape) == 4 and self.original_images.shape[1] == 3:
            # Convert RGB to grayscale
            self.images = np.mean(self.original_images, axis=1, keepdims=True)
        else:
            self.images = self.original_images
        
        # Resize to 28x28 if needed
        if self.images.shape[-1] != 28:
            self.images = self._resize_images(self.images)
        
        # Normalize to [-1, 1]
        self.images = (self.images - 0.5) * 2
        
        print(f"   Processed images: {self.images.shape}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")
    
    def _resize_images(self, images):
        """Resize images to 28x28"""
        from PIL import Image
        resized = []
        
        for img in images:
            if len(img.shape) == 3:
                img = img[0]  # Take first channel
            
            # Convert to PIL and resize
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.resize((28, 28))
            resized.append(np.array(pil_img) / 255.0)
        
        return np.array(resized)[:, None, :, :]  # Add channel dimension
    
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        fmri_emb = torch.FloatTensor(self.fmri_embeddings[idx])
        image = torch.FloatTensor(self.images[idx])
        
        return fmri_emb, image

def train_model():
    """Train the diffusion model"""
    print("🚀 Training Digit69 LDM")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load dataset
    train_dataset = Digit69Dataset("digit69_embeddings.pkl", "train")
    test_dataset = Digit69Dataset("digit69_embeddings.pkl", "test")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = SimpleDiffusionModel(condition_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training parameters
    num_epochs = 50
    num_timesteps = 1000
    
    # Noise schedule
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def add_noise(x0, t):
        """Add noise to images"""
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    # Training loop
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for fmri_emb, images in pbar:
            fmri_emb = fmri_emb.to(device)
            images = images.to(device)
            
            # Random timesteps
            t = torch.randint(0, num_timesteps, (images.shape[0],)).to(device)
            
            # Add noise
            noisy_images, noise = add_noise(images, t)
            
            # Predict noise
            predicted_noise = model(noisy_images, t, fmri_emb)
            
            # Loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'digit69_ldm_epoch_{epoch+1}.pth')
            print(f"✅ Model saved: digit69_ldm_epoch_{epoch+1}.pth")
        
        # Generate samples for visualization
        if (epoch + 1) % 5 == 0:
            generate_samples(model, test_loader, device, epoch+1, 
                           sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas)
    
    # Save final model
    torch.save(model.state_dict(), 'digit69_ldm_final.pth')
    print(f"✅ Final model saved: digit69_ldm_final.pth")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    return model, train_losses

@torch.no_grad()
def generate_samples(model, test_loader, device, epoch, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas):
    """Generate sample images"""
    model.eval()
    
    # Get test batch
    fmri_emb, original_images = next(iter(test_loader))
    fmri_emb = fmri_emb.to(device)
    original_images = original_images.to(device)
    
    batch_size = min(4, fmri_emb.shape[0])
    fmri_emb = fmri_emb[:batch_size]
    original_images = original_images[:batch_size]
    
    # Simple sampling (DDPM-like)
    generated = torch.randn(batch_size, 1, 28, 28).to(device)
    
    num_steps = 50
    step_size = len(betas) // num_steps
    
    for i in range(num_steps):
        t = torch.full((batch_size,), len(betas) - 1 - i * step_size).to(device)
        t = torch.clamp(t, 0, len(betas) - 1)
        
        predicted_noise = model(generated, t, fmri_emb)
        
        # Simple denoising step
        alpha_t = (1 - betas[t[0]])
        generated = (generated - betas[t[0]] * predicted_noise) / torch.sqrt(alpha_t)
        
        if i < num_steps - 1:
            generated = generated + torch.sqrt(betas[t[0]]) * torch.randn_like(generated) * 0.1
    
    # Visualize
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    
    for i in range(batch_size):
        # Original
        orig_img = original_images[i, 0].cpu().numpy()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Generated
        gen_img = torch.clamp(generated[i, 0], -1, 1).cpu().numpy()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Original vs Generated')
    plt.tight_layout()
    plt.savefig(f'samples_epoch_{epoch}.png')
    plt.show()

if __name__ == "__main__":
    model, losses = train_model()
