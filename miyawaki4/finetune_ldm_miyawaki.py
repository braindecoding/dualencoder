#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion on Miyawaki Dataset
Train LDM to generate Miyawaki-style images from fMRI embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler, DDIMScheduler
import warnings
warnings.filterwarnings("ignore")

class MiyawakiDataset(Dataset):
    """Dataset for Miyawaki fMRI-Image pairs"""
    
    def __init__(self, embeddings_data, split='train'):
        self.split = split
        self.fmri_embeddings = embeddings_data[split]['fmri_embeddings']
        self.image_embeddings = embeddings_data[split]['image_embeddings']
        self.original_images = embeddings_data[split]['original_images']
        
        print(f"📊 {split.title()} Dataset: {len(self.fmri_embeddings)} samples")
    
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        fmri = torch.FloatTensor(self.fmri_embeddings[idx])
        image_emb = torch.FloatTensor(self.image_embeddings[idx])
        
        # Convert original image to tensor
        original_img = self.original_images[idx]  # CHW format
        if isinstance(original_img, np.ndarray):
            original_img = torch.FloatTensor(original_img)
        
        # Normalize to [-1, 1] for diffusion
        original_img = (original_img - 0.5) * 2
        
        return {
            'fmri': fmri,
            'image_embedding': image_emb,
            'image': original_img,
            'idx': idx
        }

class MiyawakiLDMTrainer:
    """Trainer for fine-tuning LDM on Miyawaki dataset"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """Setup diffusion models for training"""
        print("🔧 Setting up models for fine-tuning...")
        
        # Load pre-trained components
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        
        # U-Net (trainable)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # fMRI conditioning network
        self.fmri_encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),  # CLIP text embedding dimension
            nn.LayerNorm(768)
        ).to(self.device)
        
        print("✅ Models setup complete")
        
    def encode_images(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        return latents
    
    def prepare_conditioning(self, fmri_embeddings, batch_size):
        """Prepare conditioning from fMRI embeddings"""
        # Encode fMRI to text embedding space
        fmri_encoded = self.fmri_encoder(fmri_embeddings)  # [B, 768]
        
        # Create text embedding structure
        # Standard CLIP text embeddings are [B, 77, 768]
        text_embeddings = torch.zeros(batch_size, 77, 768, device=self.device, dtype=fmri_encoded.dtype)
        
        # Put fMRI encoding in first token position
        text_embeddings[:, 0] = fmri_encoded
        
        return text_embeddings
    
    def training_step(self, batch):
        """Single training step with Miyawaki-specific optimizations"""
        fmri = batch['fmri'].to(self.device)
        images = batch['image'].to(self.device)

        batch_size = fmri.shape[0]

        # Encode images to latents
        latents = self.encode_images(images)

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample timesteps (focus on mid-range for pattern learning)
        timesteps = torch.randint(100, self.scheduler.config.num_train_timesteps - 100,
                                 (batch_size,), device=self.device).long()

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Prepare conditioning
        encoder_hidden_states = self.prepare_conditioning(fmri, batch_size)

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute multiple losses for Miyawaki patterns
        # 1. Standard MSE loss
        mse_loss = nn.functional.mse_loss(noise_pred, noise)

        # 2. Edge-preserving loss (important for high-contrast patterns)
        edge_loss = self.compute_edge_loss(noise_pred, noise)

        # 3. Frequency domain loss (for pattern preservation)
        freq_loss = self.compute_frequency_loss(noise_pred, noise)

        # Combined loss
        total_loss = mse_loss + 0.1 * edge_loss + 0.05 * freq_loss

        return total_loss

    def compute_edge_loss(self, pred, target):
        """Compute edge-preserving loss for high-contrast patterns"""
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Apply to each channel
        pred_edges_x = nn.functional.conv2d(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), sobel_x, padding=1)
        pred_edges_y = nn.functional.conv2d(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), sobel_y, padding=1)
        target_edges_x = nn.functional.conv2d(target.view(-1, 1, target.shape[-2], target.shape[-1]), sobel_x, padding=1)
        target_edges_y = nn.functional.conv2d(target.view(-1, 1, target.shape[-2], target.shape[-1]), sobel_y, padding=1)

        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)

        return nn.functional.mse_loss(pred_edges, target_edges)

    def compute_frequency_loss(self, pred, target):
        """Compute frequency domain loss for pattern preservation"""
        # FFT in spatial dimensions
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Focus on magnitude (pattern structure)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return nn.functional.mse_loss(pred_mag, target_mag)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.unet.train()
        self.fmri_encoder.train()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss = self.training_step(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.fmri_encoder.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log every 10 steps
            if batch_idx % 10 == 0:
                print(f"   Step {batch_idx}: Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} completed. Average loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def generate_sample(self, fmri_embedding, num_steps=20):
        """Generate sample image from fMRI"""
        self.unet.eval()
        self.fmri_encoder.eval()
        
        with torch.no_grad():
            # Prepare conditioning
            fmri_batch = fmri_embedding.unsqueeze(0)
            encoder_hidden_states = self.prepare_conditioning(fmri_batch, 1)
            
            # Initialize latents
            latents = torch.randn(1, 4, 64, 64, device=self.device)
            
            # Set up scheduler for inference
            self.scheduler.set_timesteps(num_steps)
            
            # Denoising loop
            for t in self.scheduler.timesteps:
                noise_pred = self.unet(latents, t, encoder_hidden_states).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode to image
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
            images = (images + 1) / 2  # [-1, 1] → [0, 1]
            images = torch.clamp(images, 0, 1)
            
            # Convert to PIL
            image = images[0].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            
            return Image.fromarray(image)
    
    def save_checkpoint(self, epoch, optimizer, loss, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'fmri_encoder_state_dict': self.fmri_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")

def load_miyawaki_data():
    """Load Miyawaki embeddings data"""
    print("📥 Loading Miyawaki embeddings...")
    
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        return None
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    print(f"✅ Data loaded successfully")
    print(f"   📊 Training samples: {len(embeddings_data['train']['fmri_embeddings'])}")
    print(f"   📊 Test samples: {len(embeddings_data['test']['fmri_embeddings'])}")
    
    return embeddings_data

def train_miyawaki_ldm():
    """Main training function"""
    print("🎯 MIYAWAKI LDM FINE-TUNING")
    print("=" * 50)
    
    # Load data
    embeddings_data = load_miyawaki_data()
    if embeddings_data is None:
        return
    
    # Create datasets
    train_dataset = MiyawakiDataset(embeddings_data, 'train')
    test_dataset = MiyawakiDataset(embeddings_data, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = MiyawakiLDMTrainer(device=device)
    
    # Setup optimizer
    trainable_params = list(trainer.unet.parameters()) + list(trainer.fmri_encoder.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        avg_loss = trainer.train_epoch(train_loader, optimizer, epoch+1)
        train_losses.append(avg_loss)
        
        # Generate sample every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"🎨 Generating sample image...")
            test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][0]).to(device)
            sample_image = trainer.generate_sample(test_fmri)
            sample_image.save(f'miyawaki_ldm_sample_epoch_{epoch+1}.png')
            print(f"💾 Sample saved: miyawaki_ldm_sample_epoch_{epoch+1}.png")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'miyawaki_ldm_checkpoint_epoch_{epoch+1}.pth'
            trainer.save_checkpoint(epoch+1, optimizer, avg_loss, checkpoint_path)
    
    # Final model save
    final_model_path = 'miyawaki_ldm_final.pth'
    trainer.save_checkpoint(num_epochs, optimizer, train_losses[-1], final_model_path)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', linewidth=2)
    plt.title('Miyawaki LDM Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('miyawaki_ldm_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final loss: {train_losses[-1]:.6f}")
    print(f"📁 Generated files:")
    print(f"   - miyawaki_ldm_final.pth")
    print(f"   - miyawaki_ldm_training_curve.png")
    print(f"   - miyawaki_ldm_sample_epoch_*.png")

if __name__ == "__main__":
    train_miyawaki_ldm()
