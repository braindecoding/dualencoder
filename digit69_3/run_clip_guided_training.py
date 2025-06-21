#!/usr/bin/env python3
"""
Full CLIP Guided LDM Training
Simplified training script untuk menjalankan full training dengan different CLIP weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import os

# Import our models
from improved_unet import ImprovedUNet
from clip_guidance_ldm import CLIPGuidedDiffusionModel
from torch.utils.data import Dataset, DataLoader

class Digit69CLIPDataset(Dataset):
    """Dataset untuk CLIP guided training"""
    
    def __init__(self, data_path, split="train"):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.fmri_embeddings = torch.FloatTensor(data[split]['fmri_embeddings'])
        
        # Load and process images
        original_images = data[split]['original_images']
        
        # Ensure correct format (batch, 1, 28, 28)
        if len(original_images.shape) == 4 and original_images.shape[1] == 3:
            # Convert RGB to grayscale
            self.images = torch.FloatTensor(original_images[:, 0:1, :, :])
        elif len(original_images.shape) == 3:
            # Add channel dimension
            self.images = torch.FloatTensor(original_images).unsqueeze(1)
        else:
            self.images = torch.FloatTensor(original_images)
        
        # Ensure 28x28 size
        if self.images.shape[-1] != 28:
            import torch.nn.functional as F
            self.images = F.interpolate(self.images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Generate digit classes (random for demo - in real scenario you'd have labels)
        self.digit_classes = torch.randint(0, 10, (len(self.fmri_embeddings),))
        
        print(f"   {split.upper()} Dataset:")
        print(f"     fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"     Images: {self.images.shape}")
        print(f"     Digit classes: {self.digit_classes.shape}")
    
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        return self.fmri_embeddings[idx], self.images[idx], self.digit_classes[idx]

class CLIPGuidedTrainer:
    """Simplified trainer untuk CLIP guided LDM"""
    
    def __init__(self, clip_weight=1.0, device='cuda'):
        self.clip_weight = clip_weight
        self.device = device
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_clip_score = 0.0
        
        print(f"🚀 CLIP GUIDED TRAINER (Weight: {clip_weight})")
        print(f"   Device: {device}")
        
        # Setup data
        self.setup_data()
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Training history
        self.history = {
            'train_total': [], 'train_diffusion': [], 'train_clip': [], 'train_clip_scores': [],
            'test_total': [], 'test_diffusion': [], 'test_clip': [], 'test_clip_scores': []
        }
    
    def setup_data(self):
        """Setup datasets"""
        print(f"\n📊 SETTING UP DATA")
        
        self.train_dataset = Digit69CLIPDataset("digit69_embeddings.pkl", "train")
        self.test_dataset = Digit69CLIPDataset("digit69_embeddings.pkl", "test")
        
        # Use smaller batch size untuk CLIP guidance
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
    
    def setup_model(self):
        """Setup CLIP guided model"""
        print(f"\n🏗️ SETTING UP MODEL")
        
        unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=64,
            num_res_blocks=2
        )
        
        self.model = CLIPGuidedDiffusionModel(
            unet,
            num_timesteps=1000,
            clip_guidance_weight=self.clip_weight,
            clip_model_name="ViT-B/32"
        ).to(self.device)
        
        # Fix device issues
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(self.model, attr_name)
            setattr(self.model, attr_name, attr_value.to(self.device))
        
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print(f"\n⚙️ SETTING UP OPTIMIZER")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,  # 50 epochs
            eta_min=1e-6
        )
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_diffusion = 0
        total_clip = 0
        total_clip_scores = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for fmri_emb, images, classes in pbar:
            fmri_emb = fmri_emb.to(self.device)
            images = images.to(self.device)
            classes = classes.to(self.device)
            
            batch_size = images.shape[0]
            t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
            
            # Forward pass
            loss_dict = self.model.p_losses_with_clip(images, t, fmri_emb, classes)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_diffusion += loss_dict['diffusion_loss'].item()
            total_clip += loss_dict['clip_loss'].item()
            total_clip_scores += loss_dict['clip_scores'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f"{loss_dict['total_loss'].item():.4f}",
                'CLIP': f"{loss_dict['clip_loss'].item():.4f}",
                'Score': f"{loss_dict['clip_scores'].item():.3f}"
            })
        
        return {
            'total': total_loss / num_batches,
            'diffusion': total_diffusion / num_batches,
            'clip': total_clip / num_batches,
            'clip_scores': total_clip_scores / num_batches
        }
    
    def evaluate_epoch(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        total_diffusion = 0
        total_clip = 0
        total_clip_scores = 0
        num_batches = 0
        
        with torch.no_grad():
            for fmri_emb, images, classes in self.test_loader:
                fmri_emb = fmri_emb.to(self.device)
                images = images.to(self.device)
                classes = classes.to(self.device)
                
                batch_size = images.shape[0]
                t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
                
                loss_dict = self.model.p_losses_with_clip(images, t, fmri_emb, classes)
                
                total_loss += loss_dict['total_loss'].item()
                total_diffusion += loss_dict['diffusion_loss'].item()
                total_clip += loss_dict['clip_loss'].item()
                total_clip_scores += loss_dict['clip_scores'].item()
                num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'diffusion': total_diffusion / num_batches,
            'clip': total_clip / num_batches,
            'clip_scores': total_clip_scores / num_batches
        }
    
    def generate_samples(self, num_samples=4):
        """Generate samples untuk visualization"""
        self.model.eval()

        with torch.no_grad():
            # Get test batch
            test_batch = next(iter(self.test_loader))
            fmri_emb, target_images, target_classes = test_batch

            fmri_emb = fmri_emb[:num_samples].to(self.device)
            target_images = target_images[:num_samples]
            target_classes = target_classes[:num_samples].to(self.device)

            # Fix device issues for noise schedule tensors
            for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                             'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
                attr_value = getattr(self.model, attr_name)
                setattr(self.model, attr_name, attr_value.to(self.device))

            # Generate samples using simplified approach
            generated_images = []
            for i in range(num_samples):
                # Use simplified sampling with fewer steps
                shape = (1, 1, 28, 28)
                img = torch.randn(shape, device=self.device)

                # Simple denoising (just a few steps)
                timesteps = torch.linspace(999, 0, 50, dtype=torch.long, device=self.device)

                for t in timesteps:
                    t_batch = t.unsqueeze(0)  # Make it batch dimension
                    noise_pred = self.model.unet(img, t_batch, fmri_emb[i:i+1])

                    # Simple denoising step
                    alpha = self.model.alphas[t]
                    alpha_cumprod = self.model.alphas_cumprod[t]
                    beta = self.model.betas[t]

                    if t > 0:
                        noise = torch.randn_like(img)
                    else:
                        noise = 0

                    img = (img - beta * noise_pred / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
                    if t > 0:
                        img += torch.sqrt(beta) * noise

                generated_images.append(img.cpu())

            generated_images = torch.cat(generated_images, dim=0)

            # Calculate CLIP scores
            clip_loss, clip_scores = self.model.clip_loss(
                generated_images.to(self.device), target_classes
            )

            return generated_images, target_images, clip_scores.cpu()
    
    def save_samples(self, generated, targets, clip_scores, epoch):
        """Save sample visualizations"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'CLIP Guided Samples - Epoch {epoch+1} (Weight: {self.clip_weight})', fontsize=16)
        
        for i in range(4):
            # Target
            axes[0, i].imshow(targets[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Target {i}')
            axes[0, i].axis('off')
            
            # Generated
            axes[1, i].imshow(generated[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Generated {i}\nCLIP: {clip_scores[i]:.3f}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        filename = f'clip_guided_w{self.clip_weight}_samples_epoch_{epoch+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Samples saved: {filename}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_clip_score': self.best_clip_score,
            'history': self.history,
            'clip_weight': self.clip_weight
        }
        
        # Regular checkpoint
        filename = f'clip_guided_ldm_w{self.clip_weight}_epoch_{epoch+1}.pth'
        torch.save(checkpoint, filename)
        
        # Best checkpoint
        if is_best:
            best_filename = f'clip_guided_ldm_w{self.clip_weight}_best.pth'
            torch.save(checkpoint, best_filename)
            print(f"   💾 Best model saved: {best_filename}")
    
    def train(self, num_epochs=50):
        """Main training loop"""
        print(f"\n🚀 STARTING CLIP GUIDED TRAINING")
        print(f"   Epochs: {num_epochs}")
        print(f"   CLIP weight: {self.clip_weight}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Evaluate
            test_losses = self.evaluate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_total'].append(train_losses['total'])
            self.history['train_diffusion'].append(train_losses['diffusion'])
            self.history['train_clip'].append(train_losses['clip'])
            self.history['train_clip_scores'].append(train_losses['clip_scores'])
            
            self.history['test_total'].append(test_losses['total'])
            self.history['test_diffusion'].append(test_losses['diffusion'])
            self.history['test_clip'].append(test_losses['clip'])
            self.history['test_clip_scores'].append(test_losses['clip_scores'])
            
            # Print progress
            print(f"Epoch {epoch+1:3d}: "
                  f"Train[Total={train_losses['total']:.4f}, "
                  f"CLIP={train_losses['clip']:.4f}, "
                  f"Score={train_losses['clip_scores']:.3f}] "
                  f"Test[Total={test_losses['total']:.4f}, "
                  f"Score={test_losses['clip_scores']:.3f}] "
                  f"LR={self.scheduler.get_last_lr()[0]:.2e}")
            
            # Check for best model
            is_best = test_losses['clip_scores'] > self.best_clip_score
            if is_best:
                self.best_clip_score = test_losses['clip_scores']
                self.best_loss = test_losses['total']
            
            # Skip sample generation untuk avoid errors during training
            # if (epoch + 1) % 10 == 0:
            #     generated, targets, clip_scores = self.generate_samples()
            #     self.save_samples(generated, targets, clip_scores, epoch)
            if (epoch + 1) % 10 == 0:
                print(f"   Skipping sample generation at epoch {epoch+1}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        total_time = time.time() - start_time
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Best CLIP score: {self.best_clip_score:.4f}")
        print(f"   Best total loss: {self.best_loss:.4f}")
        
        return self.history

def train_multiple_weights():
    """Train models dengan different CLIP weights"""
    print("🎯 TRAINING MULTIPLE CLIP WEIGHTS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Test different CLIP weights
    clip_weights = [0.5, 1.0, 2.0]
    all_results = {}
    
    for weight in clip_weights:
        print(f"\n🎯 TRAINING CLIP WEIGHT: {weight}")
        print("=" * 50)
        
        # Create trainer
        trainer = CLIPGuidedTrainer(clip_weight=weight, device=device)
        
        # Train model
        history = trainer.train(num_epochs=30)  # 30 epochs untuk testing
        
        # Save results
        all_results[weight] = {
            'history': history,
            'best_clip_score': trainer.best_clip_score,
            'best_loss': trainer.best_loss
        }
        
        print(f"✅ Weight {weight} completed: "
              f"Best CLIP score={trainer.best_clip_score:.4f}, "
              f"Best loss={trainer.best_loss:.4f}")
    
    # Save comparison results
    with open('clip_guided_training_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n🎉 ALL CLIP WEIGHTS TRAINED!")
    print(f"📊 Results summary:")
    for weight, results in all_results.items():
        print(f"   Weight {weight}: CLIP score={results['best_clip_score']:.4f}, "
              f"Loss={results['best_loss']:.4f}")
    
    return all_results

def main():
    """Main training function"""
    print("🚀 FULL CLIP GUIDED LDM TRAINING")
    print("=" * 60)
    
    # Train multiple weights
    results = train_multiple_weights()
    
    print(f"\n✅ FULL CLIP GUIDED TRAINING COMPLETED!")

if __name__ == "__main__":
    main()
