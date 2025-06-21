#!/usr/bin/env python3
"""
Improved CLIP Guided Training with Curriculum Learning
Fixed training approach with smaller CLIP weights and gradual introduction
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
import clip

from improved_unet import ImprovedUNet
from improved_clip_guided_diffusion import ImprovedCLIPGuidedDiffusionModel

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
    # original_images is (90, 3, 224, 224) in [0, 1] range
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

def compute_clip_score(model, images, labels):
    """Compute CLIP score for evaluation"""
    device = images.device
    
    # Prepare images for CLIP
    clip_images = model.prepare_image_for_clip(images)
    
    # Get CLIP features
    with torch.no_grad():
        image_features = model.clip_model.encode_image(clip_images)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    
    text_features = model.get_clip_text_features(labels)
    
    # Compute similarity
    similarity = torch.sum(image_features * text_features, dim=-1)
    return similarity.mean().item()

def train_improved_clip_guided_model(clip_weights=[0.01, 0.05, 0.1]):
    """Train improved CLIP guided models with curriculum learning"""
    
    print("🚀 STARTING IMPROVED CLIP GUIDED TRAINING")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    fmri_embeddings, images, labels = load_data()
    train_loader, val_loader = create_data_loader(fmri_embeddings, images, labels, batch_size=4)
    
    results = {}
    
    for clip_weight in clip_weights:
        print(f"\n🎯 TRAINING WITH CLIP WEIGHT: {clip_weight}")
        print("-" * 50)
        
        # Create model
        unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=32,  # Smaller for stability
            num_res_blocks=1    # Smaller for stability
        )
        
        model = ImprovedCLIPGuidedDiffusionModel(
            unet=unet,
            num_timesteps=1000,
            clip_guidance_weight=clip_weight
        ).to(device)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Training settings
        num_epochs = 150  # Longer training for curriculum learning
        warmup_epochs = 100  # Pure diffusion training
        
        # Training history
        train_losses = []
        diffusion_losses = []
        clip_losses = []
        clip_weights_history = []
        clip_scores = []
        val_losses = []
        
        best_val_loss = float('inf')
        best_clip_score = -float('inf')
        
        print(f"📋 Training Configuration:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Warmup epochs (diffusion only): {warmup_epochs}")
        print(f"   CLIP guidance epochs: {num_epochs - warmup_epochs}")
        print(f"   Base CLIP weight: {clip_weight}")
        print(f"   Learning rate: 1e-4")
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            epoch_diffusion_loss = 0
            epoch_clip_loss = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (fmri_emb, target_images, batch_labels) in enumerate(pbar):
                fmri_emb = fmri_emb.to(device)
                target_images = target_images.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                
                # Training step with curriculum learning
                result = model.training_step(target_images, fmri_emb, batch_labels, epoch)
                
                loss = result['total_loss']
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                epoch_train_loss += loss.item()
                epoch_diffusion_loss += result['diffusion_loss'].item()
                epoch_clip_loss += result['clip_loss'].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Phase': result['training_phase'],
                    'CLIP_w': f"{result['clip_weight']:.3f}"
                })
            
            # Average losses
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_diffusion_loss = epoch_diffusion_loss / len(train_loader)
            avg_clip_loss = epoch_clip_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            diffusion_losses.append(avg_diffusion_loss)
            clip_losses.append(avg_clip_loss)
            clip_weights_history.append(model.current_clip_weight)
            
            # Validation
            model.eval()
            val_loss = 0
            val_clip_score = 0
            
            with torch.no_grad():
                for fmri_emb, target_images, batch_labels in val_loader:
                    fmri_emb = fmri_emb.to(device)
                    target_images = target_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    result = model.training_step(target_images, fmri_emb, batch_labels, epoch)
                    val_loss += result['total_loss'].item()
                    
                    # Compute CLIP score
                    if epoch >= warmup_epochs:  # Only compute CLIP score in CLIP phase
                        clip_score = compute_clip_score(model, target_images, batch_labels)
                        val_clip_score += clip_score
            
            avg_val_loss = val_loss / len(val_loader)
            avg_clip_score = val_clip_score / len(val_loader) if epoch >= warmup_epochs else 0
            
            val_losses.append(avg_val_loss)
            clip_scores.append(avg_clip_score)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'improved_clip_guided_w{clip_weight}_best.pth')
            
            if avg_clip_score > best_clip_score:
                best_clip_score = avg_clip_score
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
                print(f"   Train Loss: {avg_train_loss:.4f}")
                print(f"   Val Loss: {avg_val_loss:.4f}")
                print(f"   Diffusion Loss: {avg_diffusion_loss:.4f}")
                print(f"   CLIP Loss: {avg_clip_loss:.4f}")
                print(f"   CLIP Weight: {model.current_clip_weight:.4f}")
                print(f"   CLIP Score: {avg_clip_score:.4f}")
                print(f"   Training Phase: {model.training_phase}")
        
        # Save final model
        torch.save(model.state_dict(), f'improved_clip_guided_w{clip_weight}_final.pth')
        
        # Store results
        results[clip_weight] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'diffusion_losses': diffusion_losses,
            'clip_losses': clip_losses,
            'clip_weights_history': clip_weights_history,
            'clip_scores': clip_scores,
            'best_val_loss': best_val_loss,
            'best_clip_score': best_clip_score,
            'warmup_epochs': warmup_epochs
        }
        
        print(f"\n✅ COMPLETED CLIP WEIGHT {clip_weight}:")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Best CLIP Score: {best_clip_score:.4f}")
    
    # Save all results
    with open('improved_clip_training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n🎉 ALL IMPROVED CLIP TRAINING COMPLETED!")
    print(f"📁 Results saved to: improved_clip_training_results.pkl")
    
    return results

if __name__ == "__main__":
    # Run improved CLIP guided training
    results = train_improved_clip_guided_model(clip_weights=[0.01, 0.05, 0.1])
    
    print("\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 50)
    for clip_weight, result in results.items():
        print(f"CLIP Weight {clip_weight}:")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  Best CLIP Score: {result['best_clip_score']:.4f}")
        print(f"  Warmup Epochs: {result['warmup_epochs']}")
