#!/usr/bin/env python3
"""
Training Script for CLIP Guided Enhanced LDM
High Priority Implementation: Training-time CLIP Guidance
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
from simple_baseline_model import Digit69BaselineDataset
from improved_unet import ImprovedUNet
from clip_guidance_ldm import CLIPGuidedDiffusionModel
from torch.utils.data import DataLoader

class CLIPGuidedTrainer:
    """Enhanced trainer dengan CLIP guidance dan comprehensive evaluation"""
    
    def __init__(self, clip_guidance_weight=1.0, device='cuda'):
        self.device = device
        self.clip_guidance_weight = clip_guidance_weight
        
        print(f"🚀 CLIP GUIDED TRAINER INITIALIZATION")
        print(f"   Device: {device}")
        print(f"   CLIP guidance weight: {clip_guidance_weight}")
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_clip_score = 0.0
        self.losses = {
            'train_total': [], 'train_diffusion': [], 'train_clip': [],
            'test_total': [], 'test_diffusion': [], 'test_clip': [],
            'clip_scores': [], 'test_clip_scores': []
        }
        
        print(f"✅ Trainer initialized successfully!")
    
    def setup_datasets(self):
        """Setup train and test datasets"""
        print(f"\n📊 SETTING UP DATASETS")
        
        self.train_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "train", target_size=28)
        self.test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
        
        # Smaller batch size untuk CLIP guidance (memory intensive)
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Test samples: {len(self.test_dataset)}")
        print(f"   Batch size: 2 (optimized for CLIP guidance)")
        
        # Create digit class mapping (assuming we know the digit classes)
        # For now, we'll use random classes - in real scenario, you'd have actual labels
        self.train_digit_classes = torch.randint(0, 10, (len(self.train_dataset),))
        self.test_digit_classes = torch.randint(0, 10, (len(self.test_dataset),))
        
        print(f"   Digit classes generated (random for demo)")
    
    def setup_model(self):
        """Setup CLIP guided diffusion model"""
        print(f"\n🏗️ SETTING UP MODEL")
        
        # Create UNet
        self.unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=64,
            num_res_blocks=2
        )
        
        # Create CLIP guided model
        self.model = CLIPGuidedDiffusionModel(
            self.unet,
            num_timesteps=1000,
            clip_guidance_weight=self.clip_guidance_weight,
            clip_model_name="ViT-B/32"
        ).to(self.device)
        
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   CLIP guidance weight: {self.clip_guidance_weight}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print(f"\n⚙️ SETTING UP OPTIMIZER")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # 100 epochs
            eta_min=1e-6
        )
        
        print(f"   Optimizer: AdamW (lr=1e-4, wd=1e-2)")
        print(f"   Scheduler: CosineAnnealingLR (T_max=100)")
    
    def train_epoch(self):
        """Train one epoch dengan CLIP guidance"""
        self.model.train()
        epoch_losses = {'total': 0, 'diffusion': 0, 'clip': 0, 'clip_scores': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, (fmri_emb, images) in enumerate(pbar):
            fmri_emb = fmri_emb.to(self.device)
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Get corresponding digit classes
            start_idx = batch_idx * self.train_loader.batch_size
            end_idx = start_idx + batch_size
            target_classes = self.train_digit_classes[start_idx:end_idx].to(self.device)
            
            # Sample timesteps
            t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
            
            # Forward pass with CLIP guidance
            loss_dict = self.model.p_losses_with_clip(
                images, t, fmri_emb, target_classes
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            epoch_losses['clip'] += loss_dict['clip_loss'].item()
            epoch_losses['clip_scores'] += loss_dict['clip_scores'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f"{loss_dict['total_loss'].item():.4f}",
                'Diff': f"{loss_dict['diffusion_loss'].item():.4f}",
                'CLIP': f"{loss_dict['clip_loss'].item():.4f}",
                'Score': f"{loss_dict['clip_scores'].item():.3f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def evaluate_epoch(self):
        """Evaluate on test set"""
        self.model.eval()
        epoch_losses = {'total': 0, 'diffusion': 0, 'clip': 0, 'clip_scores': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (fmri_emb, images) in enumerate(self.test_loader):
                fmri_emb = fmri_emb.to(self.device)
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # Get corresponding digit classes
                start_idx = batch_idx * self.test_loader.batch_size
                end_idx = start_idx + batch_size
                target_classes = self.test_digit_classes[start_idx:end_idx].to(self.device)
                
                # Sample timesteps
                t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
                
                # Forward pass
                loss_dict = self.model.p_losses_with_clip(
                    images, t, fmri_emb, target_classes
                )
                
                # Accumulate losses
                epoch_losses['total'] += loss_dict['total_loss'].item()
                epoch_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                epoch_losses['clip'] += loss_dict['clip_loss'].item()
                epoch_losses['clip_scores'] += loss_dict['clip_scores'].item()
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def generate_samples(self, num_samples=4, epoch=0):
        """Generate samples untuk visualization"""
        self.model.eval()
        
        with torch.no_grad():
            # Get test batch
            test_batch = next(iter(self.test_loader))
            fmri_emb, target_images = test_batch
            fmri_emb = fmri_emb[:num_samples].to(self.device)
            target_images = target_images[:num_samples]
            
            # Get target classes
            target_classes = self.test_digit_classes[:num_samples].to(self.device)
            
            # Generate samples (simplified sampling untuk speed)
            generated_images = []
            for i in range(num_samples):
                # Use fewer timesteps untuk faster generation
                sample = self.model.sample(fmri_emb[i:i+1], image_size=28)
                generated_images.append(sample.cpu())
            
            generated_images = torch.cat(generated_images, dim=0)
            
            # Calculate CLIP scores for generated images
            clip_loss, clip_scores = self.model.clip_loss(
                torch.cat([sample.to(self.device) for sample in generated_images], dim=0),
                target_classes
            )
            
            return generated_images, target_images, clip_scores.cpu()
    
    def save_samples(self, generated, targets, clip_scores, epoch):
        """Save sample visualizations"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'CLIP Guided Samples - Epoch {epoch+1} (Weight: {self.clip_guidance_weight})', fontsize=16)
        
        for i in range(4):
            # Target
            axes[0, i].imshow(targets[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Target {i}')
            axes[0, i].axis('off')
            
            # Generated
            axes[1, i].imshow(generated[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Generated {i}\nCLIP: {clip_scores[i]:.3f}')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(targets[i, 0].numpy() - generated[i, 0].numpy())
            axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        filename = f'clip_guided_samples_w{self.clip_guidance_weight}_epoch_{epoch+1}.png'
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
            'losses': self.losses,
            'clip_guidance_weight': self.clip_guidance_weight
        }
        
        # Regular checkpoint
        filename = f'clip_guided_ldm_w{self.clip_guidance_weight}_epoch_{epoch+1}.pth'
        torch.save(checkpoint, filename)
        
        # Best checkpoint
        if is_best:
            best_filename = f'clip_guided_ldm_w{self.clip_guidance_weight}_best.pth'
            torch.save(checkpoint, best_filename)
            print(f"   💾 Best model saved: {best_filename}")
    
    def train(self, num_epochs=50):
        """Main training loop"""
        print(f"\n🚀 STARTING CLIP GUIDED TRAINING")
        print(f"   Epochs: {num_epochs}")
        print(f"   CLIP weight: {self.clip_guidance_weight}")
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
            
            # Save losses
            self.losses['train_total'].append(train_losses['total'])
            self.losses['train_diffusion'].append(train_losses['diffusion'])
            self.losses['train_clip'].append(train_losses['clip'])
            self.losses['clip_scores'].append(train_losses['clip_scores'])
            
            self.losses['test_total'].append(test_losses['total'])
            self.losses['test_diffusion'].append(test_losses['diffusion'])
            self.losses['test_clip'].append(test_losses['clip'])
            self.losses['test_clip_scores'].append(test_losses['clip_scores'])
            
            # Print progress
            print(f"Epoch {epoch+1:3d}: "
                  f"Train[Total={train_losses['total']:.4f}, "
                  f"Diff={train_losses['diffusion']:.4f}, "
                  f"CLIP={train_losses['clip']:.4f}, "
                  f"Score={train_losses['clip_scores']:.3f}] "
                  f"Test[Total={test_losses['total']:.4f}, "
                  f"Score={test_losses['clip_scores']:.3f}] "
                  f"LR={self.scheduler.get_last_lr()[0]:.2e}")
            
            # Check for best model (based on CLIP score)
            is_best = test_losses['clip_scores'] > self.best_clip_score
            if is_best:
                self.best_clip_score = test_losses['clip_scores']
                self.best_loss = test_losses['total']
            
            # Generate and save samples every 10 epochs
            if (epoch + 1) % 10 == 0:
                generated, targets, clip_scores = self.generate_samples(epoch=epoch)
                self.save_samples(generated, targets, clip_scores, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        total_time = time.time() - start_time
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Best CLIP score: {self.best_clip_score:.4f}")
        print(f"   Best total loss: {self.best_loss:.4f}")
        
        return self.losses

def main():
    """Main function untuk testing different CLIP weights"""
    print("🎯 HIGH PRIORITY CLIP GUIDANCE IMPLEMENTATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Test different CLIP weights (High Priority Task 3)
    clip_weights = [0.5, 1.0, 2.0]  # Start with 3 weights
    
    all_results = {}
    
    for weight in clip_weights:
        print(f"\n🎯 TRAINING WITH CLIP WEIGHT: {weight}")
        print("=" * 50)
        
        # Create trainer
        trainer = CLIPGuidedTrainer(clip_guidance_weight=weight, device=device)
        
        # Train model
        losses = trainer.train(num_epochs=30)  # Shorter training untuk testing
        
        # Save results
        all_results[weight] = {
            'losses': losses,
            'best_clip_score': trainer.best_clip_score,
            'best_loss': trainer.best_loss
        }
        
        print(f"✅ Weight {weight} completed: "
              f"Best CLIP score={trainer.best_clip_score:.4f}, "
              f"Best loss={trainer.best_loss:.4f}")
    
    # Save comparison results
    with open('clip_weights_comparison.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n🎉 ALL CLIP WEIGHTS TESTED!")
    print(f"📊 Results summary:")
    for weight, results in all_results.items():
        print(f"   Weight {weight}: CLIP score={results['best_clip_score']:.4f}, "
              f"Loss={results['best_loss']:.4f}")

if __name__ == "__main__":
    main()
