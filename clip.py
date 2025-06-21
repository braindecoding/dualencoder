class CLIP_Correlation(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CLIP model untuk reference
        self.clip_model, _ = clip.load('ViT-B/32')
        self.clip_model.eval()
        
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