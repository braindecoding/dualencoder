#!/usr/bin/env python3
"""
Training Script for Improved CLIP v2.0
- Pure diffusion training (no CLIP during training)
- Architecture scaled to match Enhanced LDM
- CLIP guidance applied only during inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

from improved_clip_v2 import InferenceTimeCLIPGuidedModel, create_improved_clip_v2_model

def load_data():
    """Load digit69 embeddings and images"""
    print("📂 Loading digit69 data...")
    
    # Load embeddings
    with open('digit69_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Extract train data
    train_data = data['train']
    fmri_embeddings = train_data['fmri_embeddings']
    original_images = train_data['original_images']  # (90, 3, 224, 224)
    
    print(f"   fMRI embeddings: {fmri_embeddings.shape}")
    print(f"   Original images: {original_images.shape}")
    
    # Convert RGB images to grayscale and resize to 28x28
    images_gray = torch.FloatTensor(original_images).mean(dim=1, keepdim=True)  # (90, 1, 224, 224)
    images_resized = torch.nn.functional.interpolate(images_gray, size=(28, 28), mode='bilinear', align_corners=False)
    
    # Create labels (0-9 for digits, we'll use simple sequential labels)
    labels = torch.arange(10).repeat(9)  # 0,1,2,...,9,0,1,2,...,9 (90 samples)
    
    print(f"   Processed images: {images_resized.shape}")
    print(f"   Labels: {labels.shape}")
    
    # Convert to tensors
    fmri_embeddings = torch.FloatTensor(fmri_embeddings)
    
    # Normalize images to [-1, 1]
    images_resized = (images_resized - 0.5) * 2.0
    
    return fmri_embeddings, images_resized, labels

def create_data_loader(fmri_embeddings, images, labels, batch_size=4, train_split=0.8):
    """Create train and validation data loaders"""
    
    # Split data
    n_samples = len(fmri_embeddings)
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        fmri_embeddings[train_indices],
        images[train_indices],
        labels[train_indices]
    )
    
    val_dataset = TensorDataset(
        fmri_embeddings[val_indices],
        images[val_indices],
        labels[val_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Data split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader

def train_improved_clip_v2():
    """Train Improved CLIP v2.0 with pure diffusion approach"""
    
    print("🚀 STARTING IMPROVED CLIP v2.0 TRAINING")
    print("=" * 60)
    print("📋 Strategy:")
    print("   - Architecture: Scaled to match Enhanced LDM (64 channels, 2 blocks)")
    print("   - Training: Pure diffusion (NO CLIP during training)")
    print("   - CLIP Guidance: Inference-time only")
    print("   - Expected: Match Enhanced LDM + better CLIP scores")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    fmri_embeddings, images, labels = load_data()
    train_loader, val_loader = create_data_loader(fmri_embeddings, images, labels, batch_size=4)
    
    # Create model
    model = create_improved_clip_v2_model().to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    
    # Training settings
    num_epochs = 200  # Same as Enhanced LDM
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: 1e-4")
    print(f"   Scheduler: Cosine annealing")
    print(f"   Training type: Pure diffusion (no CLIP)")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (fmri_emb, target_images, batch_labels) in enumerate(pbar):
            fmri_emb = fmri_emb.to(device)
            target_images = target_images.to(device)
            
            optimizer.zero_grad()
            
            # Pure diffusion training step (no CLIP)
            result = model.training_step(target_images, fmri_emb)
            
            loss = result['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Type': result['training_type'],
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Update scheduler
        scheduler.step()
        
        # Average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for fmri_emb, target_images, batch_labels in val_loader:
                fmri_emb = fmri_emb.to(device)
                target_images = target_images.to(device)
                
                result = model.training_step(target_images, fmri_emb)
                val_loss += result['total_loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, 'improved_clip_v2_best.pth')
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, f'improved_clip_v2_epoch_{epoch+1}.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")
            print(f"   Best Val Loss: {best_val_loss:.4f}")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs - 1,
        'val_loss': avg_val_loss,
        'train_loss': avg_train_loss
    }, 'improved_clip_v2_final.pth')
    
    # Save training history
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'num_epochs': num_epochs,
        'model_params': sum(p.numel() for p in model.parameters()),
        'training_strategy': 'pure_diffusion_scaled_architecture'
    }
    
    with open('improved_clip_v2_training_results.pkl', 'wb') as f:
        pickle.dump(training_results, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved CLIP v2.0 Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([optimizer.param_groups[0]['lr'] * (0.5 ** (epoch // 50)) for epoch in range(num_epochs)], 
             label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('improved_clip_v2_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 IMPROVED CLIP v2.0 TRAINING COMPLETED!")
    print(f"📁 Files saved:")
    print(f"   - improved_clip_v2_best.pth (best model)")
    print(f"   - improved_clip_v2_final.pth (final model)")
    print(f"   - improved_clip_v2_training_results.pkl (training history)")
    print(f"   - improved_clip_v2_training_curves.png (training plots)")
    print(f"\n📊 Final Results:")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Training strategy: Pure diffusion (scaled architecture)")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Architecture: 64 channels, 2 blocks (matches Enhanced LDM)")
    
    return training_results

if __name__ == "__main__":
    results = train_improved_clip_v2()
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Run evaluation to compare with Enhanced LDM")
    print("   2. Test inference-time CLIP guidance")
    print("   3. Compare traditional metrics + CLIP scores")
    print("   4. Expected: Match Enhanced LDM + 5-10% better CLIP scores")
