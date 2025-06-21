#!/usr/bin/env python3
"""
CLIP-style Correlation Learning
Integrates with OpenAI CLIP for advanced cross-modal learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import clip as openai_clip  # Rename to avoid conflict
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False
    openai_clip = None

class CLIP_Correlation(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CLIP model untuk reference (optional)
        self.clip_model = None
        if CLIP_AVAILABLE:
            try:
                self.clip_model, _ = openai_clip.load('ViT-B/32', device='cpu')
                self.clip_model.eval()
                print("OpenAI CLIP model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load CLIP model: {e}")
                self.clip_model = None
        else:
            print("CLIP model not available - using standalone correlation learning")
        
        # Correlation learning network
        self.correlation_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),  # Concat X_lat + Y_lat
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            
            nn.Linear(512, latent_dim),  # Output: correlation embedding
            nn.Tanh()
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, fmri_latent, stim_latent):
        # Concatenate latent representations
        combined = torch.cat([fmri_latent, stim_latent], dim=1)
        
        # Learn correlation
        correlation_embedding = self.correlation_net(combined)
        correlation_embedding = F.normalize(correlation_embedding, p=2, dim=1)
        
        return correlation_embedding
    
    def compute_contrastive_loss(self, fmri_latent, stim_latent):
        """CLIP-style contrastive loss untuk training"""
        batch_size = fmri_latent.size(0)
        
        # Compute correlation embeddings
        corr_fmri = self.correlation_net(torch.cat([fmri_latent, stim_latent], dim=1))
        corr_stim = self.correlation_net(torch.cat([stim_latent, fmri_latent], dim=1))
        
        # Normalize
        corr_fmri = F.normalize(corr_fmri, p=2, dim=1)
        corr_stim = F.normalize(corr_stim, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(corr_fmri, corr_stim.T) / self.temperature
        
        # Labels (diagonal elements should be maximum)
        labels = torch.arange(batch_size, device=fmri_latent.device)
        
        # Contrastive loss (both directions)
        loss_fmri2stim = F.cross_entropy(similarity, labels)
        loss_stim2fmri = F.cross_entropy(similarity.T, labels)
        
        return (loss_fmri2stim + loss_stim2fmri) / 2