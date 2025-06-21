#!/usr/bin/env python3
"""
Quick Fix for Miyawaki Training
Address the fundamental preprocessing and training issues
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
import warnings
warnings.filterwarnings("ignore")

class FixedMiyawakiDataset(Dataset):
    """Fixed dataset with proper preprocessing"""
    
    def __init__(self, embeddings_data, split='train'):
        self.fmri_embeddings = embeddings_data[split]['fmri_embeddings']
        self.original_images = embeddings_data[split]['original_images']
        
        print(f"🔧 Fixed Dataset - {split}:")
        print(f"   Samples: {len(self.fmri_embeddings)}")
        print(f"   fMRI shape: {self.fmri_embeddings[0].shape}")
        print(f"   Image shape: {self.original_images[0].shape}")
        
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        # Get fMRI embedding
        fmri = torch.FloatTensor(self.fmri_embeddings[idx])
        
        # Get and properly preprocess image
        image = self.original_images[idx]
        
        # Convert CHW to HWC if needed
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # CRITICAL: Proper binary thresholding
        # Miyawaki patterns are binary - force them to be binary!
        image_binary = (image > 0.5).astype(np.float32)
        
        # Convert back to 3-channel for compatibility
        image_tensor = torch.FloatTensor(image_binary).unsqueeze(0).repeat(3, 1, 1)
        
        return {
            'fmri': fmri,
            'image': image_tensor,
            'binary_image': torch.FloatTensor(image_binary).unsqueeze(0)
        }

class BinaryPatternGenerator(nn.Module):
    """Simple binary pattern generator"""
    
    def __init__(self, input_dim=512, output_size=224):
        super().__init__()
        
        self.output_size = output_size
        
        # Simple but effective architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, output_size * output_size),
            nn.Sigmoid()  # Output in [0,1] for binary classification
        )
        
        # Pattern type classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  # 4 pattern types: cross, L, rectangle, T
            nn.Softmax(dim=1)
        )
    
    def forward(self, fmri_embedding):
        # Generate binary pattern
        pattern_flat = self.encoder(fmri_embedding)
        pattern = pattern_flat.view(-1, 1, self.output_size, self.output_size)
        
        # Classify pattern type
        pattern_type = self.pattern_classifier(fmri_embedding)
        
        return pattern, pattern_type

class QuickFixTrainer:
    """Quick fix trainer with proper binary loss"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = BinaryPatternGenerator().to(device)
        
    def train_step(self, batch):
        """Training step with binary-focused loss"""
        fmri = batch['fmri'].to(self.device)
        target_binary = batch['binary_image'].to(self.device)
        
        # Forward pass
        pred_pattern, pred_type = self.model(fmri)
        
        # Binary Cross Entropy Loss (CRITICAL for binary patterns)
        bce_loss = nn.functional.binary_cross_entropy(pred_pattern, target_binary)
        
        # Binary threshold loss (encourage sharp binary outputs)
        pred_binary = torch.where(pred_pattern > 0.5, 1.0, 0.0)
        threshold_loss = nn.functional.mse_loss(pred_binary, target_binary)
        
        # Edge preservation loss
        edge_loss = self.compute_edge_loss(pred_pattern, target_binary)
        
        # Combined loss with strong binary focus
        total_loss = bce_loss + 0.5 * threshold_loss + 0.3 * edge_loss
        
        return total_loss, bce_loss, threshold_loss, edge_loss, pred_pattern
    
    def compute_edge_loss(self, pred, target):
        """Compute edge preservation loss"""
        # Simple edge detection using differences
        pred_edges_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_edges_v = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_edges_h = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_edges_v = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        edge_loss_h = nn.functional.mse_loss(pred_edges_h, target_edges_h)
        edge_loss_v = nn.functional.mse_loss(pred_edges_v, target_edges_v)
        
        return edge_loss_h + edge_loss_v
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_bce = 0
        total_threshold = 0
        total_edge = 0
        
        progress_bar = tqdm(dataloader, desc=f"Quick Fix Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss, bce_loss, thresh_loss, edge_loss, pred = self.train_step(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_threshold += thresh_loss.item()
            total_edge += edge_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'bce': bce_loss.item(),
                'thresh': thresh_loss.item(),
                'edge': edge_loss.item()
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_bce = total_bce / len(dataloader)
        avg_thresh = total_threshold / len(dataloader)
        avg_edge = total_edge / len(dataloader)
        
        print(f"✅ Quick Fix Epoch {epoch} completed:")
        print(f"   Total Loss: {avg_loss:.6f}")
        print(f"   BCE Loss: {avg_bce:.6f}")
        print(f"   Threshold Loss: {avg_thresh:.6f}")
        print(f"   Edge Loss: {avg_edge:.6f}")
        
        return avg_loss, pred
    
    def generate_binary_pattern(self, fmri_embedding):
        """Generate binary pattern from fMRI"""
        self.model.eval()
        
        with torch.no_grad():
            if fmri_embedding.dim() == 1:
                fmri_embedding = fmri_embedding.unsqueeze(0)
            
            pred_pattern, pred_type = self.model(fmri_embedding)
            
            # Apply binary threshold
            binary_pattern = torch.where(pred_pattern > 0.5, 1.0, 0.0)
            
            # Convert to PIL image
            pattern_np = binary_pattern[0, 0].cpu().numpy()
            pattern_img = Image.fromarray((pattern_np * 255).astype(np.uint8))
            
            return pattern_img, pred_type[0]

def run_quick_fix_training():
    """Run quick fix training"""
    print("🚀 QUICK FIX MIYAWAKI TRAINING")
    print("=" * 60)
    print("🎯 Focus: BINARY PATTERNS with proper preprocessing")
    print()
    
    # Load data
    embeddings_path = "miyawaki4_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Create fixed datasets
    train_dataset = FixedMiyawakiDataset(embeddings_data, 'train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = QuickFixTrainer(device=device)
    
    # Setup optimizer
    optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    num_epochs = 30
    losses = []
    
    print(f"🏋️ Starting quick fix training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Quick Fix Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        avg_loss, pred_samples = trainer.train_epoch(train_loader, optimizer, epoch+1)
        losses.append(avg_loss)
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"🎨 Generating quick fix sample...")
            test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][0]).to(device)
            sample_img, pattern_type = trainer.generate_binary_pattern(test_fmri)
            sample_img.save(f'quickfix_binary_sample_epoch_{epoch+1}.png')
            print(f"💾 Quick fix sample saved")
    
    # Save final model
    torch.save(trainer.model.state_dict(), 'quickfix_binary_generator.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Quick Fix Binary Pattern Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('quickfix_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid hanging
    
    print(f"\n🎉 Quick fix training completed!")
    print(f"📁 Generated files:")
    print(f"   - quickfix_binary_generator.pth")
    print(f"   - quickfix_training_curve.png")
    print(f"   - quickfix_binary_sample_epoch_*.png")
    
    # Generate final comparison
    print(f"\n🔍 Generating final comparison...")
    test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][0]).to(device)
    final_img, pattern_type = trainer.generate_binary_pattern(test_fmri)
    final_img.save('quickfix_final_result.png')
    
    print(f"✅ QUICK FIX COMPLETE!")
    print(f"📊 Check quickfix_final_result.png for the result")

if __name__ == "__main__":
    run_quick_fix_training()
