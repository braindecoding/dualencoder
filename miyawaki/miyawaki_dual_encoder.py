#!/usr/bin/env python3
"""
Dual Encoder implementation khusus untuk Miyawaki dataset
Menggunakan arsitektur yang disesuaikan dengan dimensi fMRI 967 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from miyawaki_dataset_loader import load_miyawaki_dataset, create_dataloaders
from pathlib import Path

class MiyawakifMRIEncoder(nn.Module):
    """fMRI Encoder khusus untuk Miyawaki (967 features)"""
    
    def __init__(self, fmri_dim=967, latent_dim=512):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(fmri_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            # Layer 2
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.08),
            
            # Layer 3
            nn.Linear(768, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()  # Normalize to [-1,1]
        )
        
    def forward(self, fmri_signals):
        # fMRI → latent space
        X_lat = self.encoder(fmri_signals)
        # Normalize ke unit sphere seperti CLIP
        X_lat = F.normalize(X_lat, p=2, dim=1)
        return X_lat

class MiyawakiStimulusEncoder(nn.Module):
    """Stimulus Encoder untuk Miyawaki visual stimuli"""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CNN untuk process 28x28 images
        self.cnn_encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            
            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            
            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        # FC layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )
        
    def forward(self, stimuli):
        # Input: (batch, 28, 28) → (batch, 1, 28, 28)
        if len(stimuli.shape) == 3:
            stimuli = stimuli.unsqueeze(1)
        
        # CNN feature extraction
        features = self.cnn_encoder(stimuli)
        features = features.flatten(1)
        
        # FC encoding
        Y_lat = self.fc_encoder(features)
        Y_lat = F.normalize(Y_lat, p=2, dim=1)
        return Y_lat

class MiyawakiCLIPCorrelation(nn.Module):
    """CLIP-style correlation learning untuk Miyawaki"""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Correlation learning network
        self.correlation_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),  # Concat X_lat + Y_lat
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(512, latent_dim),  # Output: correlation embedding
            nn.Tanh()
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, X_lat, Y_lat):
        # Concatenate latent representations
        combined = torch.cat([X_lat, Y_lat], dim=1)
        
        # Learn correlation
        correlation_embedding = self.correlation_net(combined)
        correlation_embedding = F.normalize(correlation_embedding, p=2, dim=1)
        
        return correlation_embedding
    
    def compute_contrastive_loss(self, X_lat, Y_lat):
        """CLIP-style contrastive loss"""
        batch_size = X_lat.size(0)
        
        # Compute correlation embeddings
        corr_x = self.correlation_net(torch.cat([X_lat, Y_lat], dim=1))
        corr_y = self.correlation_net(torch.cat([Y_lat, X_lat], dim=1))
        
        # Normalize
        corr_x = F.normalize(corr_x, p=2, dim=1)
        corr_y = F.normalize(corr_y, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(corr_x, corr_y.T) / self.temperature
        
        # Labels (diagonal elements should be maximum)
        labels = torch.arange(batch_size, device=X_lat.device)
        
        # Contrastive loss (both directions)
        loss_x2y = F.cross_entropy(similarity, labels)
        loss_y2x = F.cross_entropy(similarity.T, labels)
        
        return (loss_x2y + loss_y2x) / 2

class MiyawakiDualEncoder(nn.Module):
    """Complete Dual Encoder untuk Miyawaki dataset"""
    
    def __init__(self, fmri_dim=967, latent_dim=512):
        super().__init__()
        
        self.fmri_encoder = MiyawakifMRIEncoder(fmri_dim, latent_dim)
        self.stimulus_encoder = MiyawakiStimulusEncoder(latent_dim)
        self.clip_correlation = MiyawakiCLIPCorrelation(latent_dim)
        
    def forward(self, fmri_data, stimulus_data):
        # Encode to latent spaces
        X_lat = self.fmri_encoder(fmri_data)
        Y_lat = self.stimulus_encoder(stimulus_data)
        
        # Learn correlation
        correlation_embedding = self.clip_correlation(X_lat, Y_lat)
        
        return {
            'fmri_latent': X_lat,
            'stimulus_latent': Y_lat,
            'correlation': correlation_embedding
        }
    
    def compute_loss(self, fmri_data, stimulus_data):
        """Compute training loss"""
        outputs = self.forward(fmri_data, stimulus_data)
        
        X_lat = outputs['fmri_latent']
        Y_lat = outputs['stimulus_latent']
        
        # CLIP contrastive loss
        contrastive_loss = self.clip_correlation.compute_contrastive_loss(X_lat, Y_lat)
        
        # Optional: reconstruction losses
        # recon_loss_x = F.mse_loss(self.fmri_decoder(X_lat), fmri_data)
        # recon_loss_y = F.mse_loss(self.stimulus_decoder(Y_lat), stimulus_data)
        
        total_loss = contrastive_loss
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'outputs': outputs
        }

def train_miyawaki_dual_encoder(model, train_loader, test_loader, num_epochs=50, lr=1e-3):
    """Training function untuk Miyawaki dual encoder"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            
            optimizer.zero_grad()
            
            loss_dict = model.compute_loss(fmri, stimulus)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(device)
                stimulus = batch['stimulus'].to(device)
                
                loss_dict = model.compute_loss(fmri, stimulus)
                test_loss += loss_dict['total_loss'].item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-20:], label='Train Loss (last 20)')
    plt.plot(test_losses[-20:], label='Test Loss (last 20)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves (Last 20 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('miyawaki_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return train_losses, test_losses

def main():
    """Main function untuk training Miyawaki dual encoder"""
    
    # Load dataset
    filepath = Path("../dataset/miyawaki_structured_28x28.mat")
    dataset_dict = load_miyawaki_dataset(filepath)
    
    if dataset_dict is None:
        print("Failed to load dataset")
        return
    
    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(dataset_dict, batch_size=8)
    
    # Initialize model
    model = MiyawakiDualEncoder(fmri_dim=967, latent_dim=512)
    
    print("Model architecture:")
    print(f"  fMRI Encoder: {sum(p.numel() for p in model.fmri_encoder.parameters())} parameters")
    print(f"  Stimulus Encoder: {sum(p.numel() for p in model.stimulus_encoder.parameters())} parameters")
    print(f"  CLIP Correlation: {sum(p.numel() for p in model.clip_correlation.parameters())} parameters")
    print(f"  Total: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_losses, test_losses = train_miyawaki_dual_encoder(
        model, train_loader, test_loader, 
        num_epochs=100, lr=1e-3
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'dataset_metadata': dataset_dict['metadata']
    }, 'miyawaki_dual_encoder.pth')
    
    print("Training completed! Model saved as 'miyawaki_dual_encoder.pth'")

if __name__ == "__main__":
    main()
