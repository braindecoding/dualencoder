#!/usr/bin/env python3
"""
Visual Stimulus Encoder for Miyawaki Dataset
CNN architecture for 28x28 images → 512 latent features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Shape_Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Miyawaki shapes: 28x28 = 784 pixels
        self.cnn_encoder = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, 32, 3, padding=1),  # → [batch, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # → [batch, 32, 14, 14]
            
            nn.Conv2d(32, 64, 3, padding=1), # → [batch, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # → [batch, 64, 7, 7]
            
            nn.Conv2d(64, 128, 3, padding=1), # → [batch, 128, 7, 7]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))     # → [batch, 128, 4, 4]
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),    # 2048 → 512
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),     # → latent space
            nn.Tanh()
        )
    
    def forward(self, shapes):
        # stimTrn (28x28) → stim_latent
        features = self.cnn_encoder(shapes)
        features = features.flatten(1)      # Flatten to [batch, 2048]
        stim_latent = self.fc_encoder(features)
        stim_latent = F.normalize(stim_latent, p=2, dim=1)
        return stim_latent