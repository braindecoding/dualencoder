#!/usr/bin/env python3
"""
Intensive Miyawaki LDM Training
More aggressive fine-tuning for better pattern matching
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

class IntensiveMiyawakiTrainer:
    """More aggressive trainer for Miyawaki patterns"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """Setup models with more aggressive fine-tuning"""
        print("🔧 Setting up intensive training models...")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        
        # U-Net (more layers trainable)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Enhanced fMRI conditioning network
        self.fmri_encoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768)
        ).to(self.device)
        
        # Pattern-specific embedding
        self.pattern_encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4*64*64),  # Direct latent space
            nn.Tanh()
        ).to(self.device)
        
        print("✅ Intensive training models setup complete")
    
    def enhanced_training_step(self, batch):
        """Enhanced training step with stronger pattern focus"""
        fmri = batch['fmri'].to(self.device)
        images = batch['image'].to(self.device)
        
        batch_size = fmri.shape[0]
        
        # Encode images to latents
        latents = self.encode_images(images)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Focus on early timesteps for pattern learning
        timesteps = torch.randint(50, 500, (batch_size,), device=self.device).long()
        
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Enhanced conditioning
        encoder_hidden_states = self.prepare_conditioning(fmri, batch_size)
        
        # Pattern-specific conditioning
        pattern_conditioning = self.pattern_encoder(fmri).view(batch_size, 4, 64, 64)
        
        # Add pattern conditioning to noisy latents
        conditioned_latents = noisy_latents + 0.1 * pattern_conditioning
        
        # Predict noise
        noise_pred = self.unet(conditioned_latents, timesteps, encoder_hidden_states).sample
        
        # Enhanced loss computation
        # 1. Standard MSE loss
        mse_loss = nn.functional.mse_loss(noise_pred, noise)
        
        # 2. Pattern preservation loss (stronger weight)
        pattern_loss = self.compute_pattern_loss(noise_pred, noise, images)
        
        # 3. Binary pattern loss (for Miyawaki's binary nature)
        binary_loss = self.compute_binary_loss(noise_pred, noise)
        
        # Combined loss with stronger pattern focus
        total_loss = mse_loss + 0.5 * pattern_loss + 0.3 * binary_loss
        
        return total_loss, mse_loss, pattern_loss, binary_loss
    
    def compute_pattern_loss(self, pred, target, original_images):
        """Compute pattern-specific loss"""
        # Convert to image space for pattern analysis
        pred_images = self.latents_to_images(pred)
        target_images = self.latents_to_images(target)
        
        # Binary threshold loss (Miyawaki patterns are binary)
        pred_binary = torch.where(pred_images > 0.5, 1.0, 0.0)
        target_binary = torch.where(target_images > 0.5, 1.0, 0.0)
        
        binary_loss = nn.functional.mse_loss(pred_binary, target_binary)
        
        # Edge consistency loss
        edge_loss = self.compute_edge_loss(pred_images, target_images)
        
        return binary_loss + edge_loss
    
    def compute_binary_loss(self, pred, target):
        """Compute binary pattern loss"""
        # Encourage binary-like outputs
        pred_sigmoid = torch.sigmoid(pred)
        binary_penalty = torch.mean((pred_sigmoid - 0.5)**2)
        
        return binary_penalty
    
    def latents_to_images(self, latents):
        """Convert latents to images for loss computation"""
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
        return images
    
    def intensive_train_epoch(self, dataloader, optimizer, epoch):
        """Intensive training epoch"""
        self.unet.train()
        self.fmri_encoder.train()
        self.pattern_encoder.train()
        
        total_loss = 0
        total_mse = 0
        total_pattern = 0
        total_binary = 0
        
        progress_bar = tqdm(dataloader, desc=f"Intensive Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss, mse_loss, pattern_loss, binary_loss = self.enhanced_training_step(batch)
            loss.backward()
            
            # Stronger gradient clipping
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.fmri_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.pattern_encoder.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_pattern += pattern_loss.item()
            total_binary += binary_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mse': mse_loss.item(),
                'pattern': pattern_loss.item(),
                'binary': binary_loss.item()
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_pattern = total_pattern / len(dataloader)
        avg_binary = total_binary / len(dataloader)
        
        print(f"✅ Intensive Epoch {epoch} completed:")
        print(f"   Total Loss: {avg_loss:.6f}")
        print(f"   MSE Loss: {avg_mse:.6f}")
        print(f"   Pattern Loss: {avg_pattern:.6f}")
        print(f"   Binary Loss: {avg_binary:.6f}")
        
        return avg_loss, avg_mse, avg_pattern, avg_binary

def run_intensive_training():
    """Run intensive Miyawaki training"""
    print("🚀 INTENSIVE MIYAWAKI LDM TRAINING")
    print("=" * 60)
    
    # Load data
    embeddings_path = "miyawaki4_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    from finetune_ldm_miyawaki import MiyawakiDataset
    
    # Create datasets with smaller batch size for intensive training
    train_dataset = MiyawakiDataset(embeddings_data, 'train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Smaller batch
    
    # Initialize intensive trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = IntensiveMiyawakiTrainer(device=device)
    
    # Setup optimizer with lower learning rate
    trainable_params = (list(trainer.unet.parameters()) + 
                       list(trainer.fmri_encoder.parameters()) + 
                       list(trainer.pattern_encoder.parameters()))
    optimizer = optim.AdamW(trainable_params, lr=5e-6, weight_decay=0.01)  # Lower LR
    
    # Intensive training loop
    num_epochs = 20  # More epochs
    losses = {'total': [], 'mse': [], 'pattern': [], 'binary': []}
    
    print(f"🏋️ Starting intensive training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Intensive Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        avg_loss, avg_mse, avg_pattern, avg_binary = trainer.intensive_train_epoch(
            train_loader, optimizer, epoch+1)
        
        losses['total'].append(avg_loss)
        losses['mse'].append(avg_mse)
        losses['pattern'].append(avg_pattern)
        losses['binary'].append(avg_binary)
        
        # Generate sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"🎨 Generating intensive sample...")
            test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][0]).to(device)
            sample_image = trainer.generate_sample(test_fmri, num_steps=20)
            sample_image.save(f'intensive_miyawaki_sample_epoch_{epoch+1}.png')
            print(f"💾 Intensive sample saved")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'intensive_miyawaki_checkpoint_epoch_{epoch+1}.pth'
            trainer.save_checkpoint(epoch+1, optimizer, avg_loss, checkpoint_path)
    
    # Final model save
    final_model_path = 'intensive_miyawaki_final.pth'
    trainer.save_checkpoint(num_epochs, optimizer, losses['total'][-1], final_model_path)
    
    # Plot intensive training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(losses['total'], 'b-', linewidth=2)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(losses['mse'], 'r-', linewidth=2)
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(losses['pattern'], 'g-', linewidth=2)
    plt.title('Pattern Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(losses['binary'], 'm-', linewidth=2)
    plt.title('Binary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Intensive Miyawaki LDM Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('intensive_miyawaki_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Intensive training completed!")
    print(f"📁 Generated files:")
    print(f"   - intensive_miyawaki_final.pth")
    print(f"   - intensive_miyawaki_training_curves.png")
    print(f"   - intensive_miyawaki_sample_epoch_*.png")

if __name__ == "__main__":
    run_intensive_training()
