#!/usr/bin/env python3
"""
Simple CLIP Guided Training
Simplified version tanpa sample generation untuk avoid errors
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import time

# Import our models
from improved_unet import ImprovedUNet
from clip_guidance_ldm import CLIPGuidedDiffusionModel
from torch.utils.data import Dataset, DataLoader

class SimpleDigit69Dataset(Dataset):
    """Simple dataset untuk CLIP guided training"""
    
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
        
        # Generate digit classes (random for demo)
        self.digit_classes = torch.randint(0, 10, (len(self.fmri_embeddings),))
        
        print(f"   {split.upper()} Dataset: {len(self.fmri_embeddings)} samples")
    
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        return self.fmri_embeddings[idx], self.images[idx], self.digit_classes[idx]

def train_clip_guided_model(clip_weight=1.0, num_epochs=20, device='cuda'):
    """Train CLIP guided model dengan weight tertentu"""
    
    print(f"\n🚀 TRAINING CLIP WEIGHT: {clip_weight}")
    print("=" * 50)
    
    # Setup data
    train_dataset = SimpleDigit69Dataset("digit69_embeddings.pkl", "train")
    test_dataset = SimpleDigit69Dataset("digit69_embeddings.pkl", "test")
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Setup model
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )
    
    model = CLIPGuidedDiffusionModel(
        unet,
        num_timesteps=1000,
        clip_guidance_weight=clip_weight,
        clip_model_name="ViT-B/32"
    ).to(device)
    
    # Fix device issues
    for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
        attr_value = getattr(model, attr_name)
        setattr(model, attr_name, attr_value.to(device))
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_clip_score = 0.0
    best_loss = float('inf')
    history = {
        'train_total': [], 'train_diffusion': [], 'train_clip': [], 'train_clip_scores': [],
        'test_total': [], 'test_diffusion': [], 'test_clip': [], 'test_clip_scores': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = {'total': 0, 'diffusion': 0, 'clip': 0, 'clip_scores': 0}
        num_train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for fmri_emb, images, classes in pbar:
            fmri_emb = fmri_emb.to(device)
            images = images.to(device)
            classes = classes.to(device)
            
            batch_size = images.shape[0]
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
            
            # Forward pass
            loss_dict = model.p_losses_with_clip(images, t, fmri_emb, classes)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            train_losses['total'] += loss_dict['total_loss'].item()
            train_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            train_losses['clip'] += loss_dict['clip_loss'].item()
            train_losses['clip_scores'] += loss_dict['clip_scores'].item()
            num_train_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f"{loss_dict['total_loss'].item():.4f}",
                'CLIP': f"{loss_dict['clip_loss'].item():.4f}",
                'Score': f"{loss_dict['clip_scores'].item():.3f}"
            })
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= num_train_batches
        
        # Evaluation
        model.eval()
        test_losses = {'total': 0, 'diffusion': 0, 'clip': 0, 'clip_scores': 0}
        num_test_batches = 0
        
        with torch.no_grad():
            for fmri_emb, images, classes in test_loader:
                fmri_emb = fmri_emb.to(device)
                images = images.to(device)
                classes = classes.to(device)
                
                batch_size = images.shape[0]
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                
                loss_dict = model.p_losses_with_clip(images, t, fmri_emb, classes)
                
                test_losses['total'] += loss_dict['total_loss'].item()
                test_losses['diffusion'] += loss_dict['diffusion_loss'].item()
                test_losses['clip'] += loss_dict['clip_loss'].item()
                test_losses['clip_scores'] += loss_dict['clip_scores'].item()
                num_test_batches += 1
        
        # Average test losses
        for key in test_losses:
            test_losses[key] /= num_test_batches
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_total'].append(train_losses['total'])
        history['train_diffusion'].append(train_losses['diffusion'])
        history['train_clip'].append(train_losses['clip'])
        history['train_clip_scores'].append(train_losses['clip_scores'])
        
        history['test_total'].append(test_losses['total'])
        history['test_diffusion'].append(test_losses['diffusion'])
        history['test_clip'].append(test_losses['clip'])
        history['test_clip_scores'].append(test_losses['clip_scores'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}: "
              f"Train[Total={train_losses['total']:.4f}, "
              f"CLIP={train_losses['clip']:.4f}, "
              f"Score={train_losses['clip_scores']:.3f}] "
              f"Test[Total={test_losses['total']:.4f}, "
              f"Score={test_losses['clip_scores']:.3f}] "
              f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Check for best model
        is_best = test_losses['clip_scores'] > best_clip_score
        if is_best:
            best_clip_score = test_losses['clip_scores']
            best_loss = test_losses['total']
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_clip_score': best_clip_score,
                'history': history,
                'clip_weight': clip_weight
            }
            
            try:
                best_filename = f'simple_clip_guided_w{clip_weight}_best.pth'
                torch.save(checkpoint, best_filename)
                print(f"   💾 Best model saved: {best_filename}")
            except Exception as e:
                print(f"   ⚠️ Failed to save model: {e}")
    
    total_time = time.time() - start_time
    print(f"\n✅ TRAINING COMPLETED!")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Best CLIP score: {best_clip_score:.4f}")
    print(f"   Best total loss: {best_loss:.4f}")
    
    return {
        'history': history,
        'best_clip_score': best_clip_score,
        'best_loss': best_loss,
        'clip_weight': clip_weight
    }

def main():
    """Main training function"""
    print("🚀 SIMPLE CLIP GUIDED LDM TRAINING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Test different CLIP weights
    clip_weights = [0.5, 1.0, 2.0]
    all_results = {}
    
    for weight in clip_weights:
        try:
            result = train_clip_guided_model(
                clip_weight=weight,
                num_epochs=20,  # Shorter training
                device=device
            )
            all_results[weight] = result
            
            print(f"✅ Weight {weight} completed: "
                  f"Best CLIP score={result['best_clip_score']:.4f}, "
                  f"Best loss={result['best_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Weight {weight} failed: {e}")
            continue
    
    # Save comparison results
    try:
        with open('simple_clip_training_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n📊 Results saved: simple_clip_training_results.pkl")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    print(f"\n🎉 ALL CLIP WEIGHTS TESTED!")
    print(f"📊 Results summary:")
    for weight, results in all_results.items():
        print(f"   Weight {weight}: CLIP score={results['best_clip_score']:.4f}, "
              f"Loss={results['best_loss']:.4f}")

if __name__ == "__main__":
    main()
