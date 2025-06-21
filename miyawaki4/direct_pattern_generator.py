#!/usr/bin/env python3
"""
Direct Pattern Generator for Miyawaki
Bypass diffusion, generate patterns directly from fMRI
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

class DirectPatternGenerator(nn.Module):
    """Direct fMRI to Miyawaki pattern generator"""
    
    def __init__(self, input_dim=512, output_size=512):
        super().__init__()
        
        self.output_size = output_size
        
        # Pattern feature extractor
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Pattern type classifier (cross, L-shape, etc.)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 10 pattern types
            nn.Softmax(dim=1)
        )
        
        # Pattern parameter predictor
        self.pattern_params = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 8),  # position, size, rotation, etc.
            nn.Tanh()
        )
        
        # Direct image generator
        self.image_generator = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.ReLU(),
            nn.Linear(16384, output_size * output_size),
            nn.Sigmoid()  # Binary patterns
        )
    
    def forward(self, fmri_embedding):
        # Extract pattern features
        features = self.pattern_encoder(fmri_embedding)
        
        # Predict pattern type
        pattern_type = self.pattern_classifier(features)
        
        # Predict pattern parameters
        pattern_params = self.pattern_params(features)
        
        # Generate image directly
        image_flat = self.image_generator(features)
        image = image_flat.view(-1, 1, self.output_size, self.output_size)
        
        return image, pattern_type, pattern_params
    
    def generate_geometric_pattern(self, pattern_type, pattern_params, size=512):
        """Generate geometric pattern based on predicted type and params"""
        batch_size = pattern_type.shape[0]
        images = torch.zeros(batch_size, 1, size, size)
        
        for i in range(batch_size):
            # Get dominant pattern type
            dominant_type = torch.argmax(pattern_type[i]).item()
            params = pattern_params[i]
            
            # Extract parameters
            center_x = int((params[0] + 1) * size / 4 + size / 4)  # [-1,1] -> [size/4, 3*size/4]
            center_y = int((params[1] + 1) * size / 4 + size / 4)
            width = int((params[2] + 1) * size / 8 + size / 16)    # Pattern width
            height = int((params[3] + 1) * size / 8 + size / 16)   # Pattern height
            
            # Generate pattern based on type
            if dominant_type == 0:  # Cross pattern
                images[i] = self.generate_cross(size, center_x, center_y, width, height)
            elif dominant_type == 1:  # L-shape
                images[i] = self.generate_l_shape(size, center_x, center_y, width, height)
            elif dominant_type == 2:  # Rectangle
                images[i] = self.generate_rectangle(size, center_x, center_y, width, height)
            elif dominant_type == 3:  # T-shape
                images[i] = self.generate_t_shape(size, center_x, center_y, width, height)
            else:  # Default to cross
                images[i] = self.generate_cross(size, center_x, center_y, width, height)
        
        return images
    
    def generate_cross(self, size, cx, cy, w, h):
        """Generate cross pattern"""
        img = torch.zeros(1, size, size)
        
        # Horizontal bar
        y_start = max(0, cy - h//2)
        y_end = min(size, cy + h//2)
        x_start = max(0, cx - w)
        x_end = min(size, cx + w)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        # Vertical bar
        y_start = max(0, cy - h)
        y_end = min(size, cy + h)
        x_start = max(0, cx - w//2)
        x_end = min(size, cx + w//2)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        return img
    
    def generate_l_shape(self, size, cx, cy, w, h):
        """Generate L-shape pattern"""
        img = torch.zeros(1, size, size)
        
        # Horizontal part
        y_start = max(0, cy)
        y_end = min(size, cy + h//2)
        x_start = max(0, cx - w)
        x_end = min(size, cx + w//2)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        # Vertical part
        y_start = max(0, cy - h)
        y_end = min(size, cy + h//2)
        x_start = max(0, cx - w//2)
        x_end = min(size, cx)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        return img
    
    def generate_rectangle(self, size, cx, cy, w, h):
        """Generate rectangle pattern"""
        img = torch.zeros(1, size, size)
        
        y_start = max(0, cy - h//2)
        y_end = min(size, cy + h//2)
        x_start = max(0, cx - w//2)
        x_end = min(size, cx + w//2)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        return img
    
    def generate_t_shape(self, size, cx, cy, w, h):
        """Generate T-shape pattern"""
        img = torch.zeros(1, size, size)
        
        # Horizontal top
        y_start = max(0, cy - h//2)
        y_end = min(size, cy)
        x_start = max(0, cx - w)
        x_end = min(size, cx + w)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        # Vertical stem
        y_start = max(0, cy)
        y_end = min(size, cy + h)
        x_start = max(0, cx - w//4)
        x_end = min(size, cx + w//4)
        img[0, y_start:y_end, x_start:x_end] = 1.0
        
        return img

class DirectPatternTrainer:
    """Trainer for direct pattern generation"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = DirectPatternGenerator().to(device)
        
    def train_step(self, batch):
        """Training step for direct pattern generation"""
        fmri = batch['fmri'].to(self.device)
        target_images = batch['image'].to(self.device)
        
        # Convert target to grayscale and binary
        target_gray = torch.mean(target_images, dim=1, keepdim=True)  # RGB to grayscale
        target_binary = torch.where(target_gray > 0.5, 1.0, 0.0)
        
        # Forward pass
        pred_images, pattern_type, pattern_params = self.model(fmri)
        
        # Generate geometric patterns
        geometric_patterns = self.model.generate_geometric_pattern(pattern_type, pattern_params)
        geometric_patterns = geometric_patterns.to(self.device)
        
        # Losses
        # 1. Direct image reconstruction loss
        image_loss = nn.functional.mse_loss(pred_images, target_binary)
        
        # 2. Binary pattern loss
        binary_loss = nn.functional.binary_cross_entropy(pred_images, target_binary)
        
        # 3. Geometric pattern loss
        geometric_loss = nn.functional.mse_loss(geometric_patterns, target_binary)
        
        # Combined loss
        total_loss = image_loss + binary_loss + 0.5 * geometric_loss
        
        return total_loss, image_loss, binary_loss, geometric_loss, pred_images, geometric_patterns
    
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_image_loss = 0
        total_binary_loss = 0
        total_geometric_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Direct Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            loss, img_loss, bin_loss, geo_loss, pred_imgs, geo_imgs = self.train_step(batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_image_loss += img_loss.item()
            total_binary_loss += bin_loss.item()
            total_geometric_loss += geo_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'img': img_loss.item(),
                'bin': bin_loss.item(),
                'geo': geo_loss.item()
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_img_loss = total_image_loss / len(dataloader)
        avg_bin_loss = total_binary_loss / len(dataloader)
        avg_geo_loss = total_geometric_loss / len(dataloader)
        
        print(f"✅ Direct Epoch {epoch} completed:")
        print(f"   Total Loss: {avg_loss:.6f}")
        print(f"   Image Loss: {avg_img_loss:.6f}")
        print(f"   Binary Loss: {avg_bin_loss:.6f}")
        print(f"   Geometric Loss: {avg_geo_loss:.6f}")
        
        return avg_loss, pred_imgs, geo_imgs
    
    def generate_pattern(self, fmri_embedding):
        """Generate pattern from fMRI embedding"""
        self.model.eval()
        
        with torch.no_grad():
            fmri_embedding = fmri_embedding.unsqueeze(0) if fmri_embedding.dim() == 1 else fmri_embedding
            pred_images, pattern_type, pattern_params = self.model(fmri_embedding)
            geometric_patterns = self.model.generate_geometric_pattern(pattern_type, pattern_params)
            
            # Convert to PIL images
            pred_img = pred_images[0, 0].cpu().numpy()
            geo_img = geometric_patterns[0, 0].cpu().numpy()
            
            pred_pil = Image.fromarray((pred_img * 255).astype(np.uint8))
            geo_pil = Image.fromarray((geo_img * 255).astype(np.uint8))
            
            return pred_pil, geo_pil, pattern_type[0], pattern_params[0]

def run_direct_training():
    """Run direct pattern generation training"""
    print("🎯 DIRECT MIYAWAKI PATTERN GENERATION")
    print("=" * 60)
    
    # Load data
    embeddings_path = "miyawaki4_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    from finetune_ldm_miyawaki import MiyawakiDataset
    
    # Create datasets
    train_dataset = MiyawakiDataset(embeddings_data, 'train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = DirectPatternTrainer(device=device)
    
    # Setup optimizer
    optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    num_epochs = 50
    losses = []
    
    print(f"🏋️ Starting direct training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Direct Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        avg_loss, pred_imgs, geo_imgs = trainer.train_epoch(train_loader, optimizer, epoch+1)
        losses.append(avg_loss)
        
        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"🎨 Generating direct samples...")
            test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][0]).to(device)
            pred_img, geo_img, pattern_type, pattern_params = trainer.generate_pattern(test_fmri)
            
            pred_img.save(f'direct_pattern_pred_epoch_{epoch+1}.png')
            geo_img.save(f'direct_pattern_geo_epoch_{epoch+1}.png')
            print(f"💾 Direct samples saved")
    
    # Save final model
    torch.save(trainer.model.state_dict(), 'direct_pattern_generator.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Direct Pattern Generation Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('direct_pattern_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Direct training completed!")
    print(f"📁 Generated files:")
    print(f"   - direct_pattern_generator.pth")
    print(f"   - direct_pattern_training_curve.png")
    print(f"   - direct_pattern_*_epoch_*.png")

if __name__ == "__main__":
    run_direct_training()
