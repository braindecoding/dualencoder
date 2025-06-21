class fMRI_Encoder(nn.Module):
    def __init__(self, fmri_dim=967, latent_dim=512):
        super().__init__()
        # Miyawaki-specific: 967 voxels → 512 latent dims
        self.encoder = nn.Sequential(
            nn.Linear(967, 2048),          # fMRI voxels → hidden
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(), 
            nn.Dropout(0.08),
            
            nn.Linear(1024, latent_dim),   # → latent space
            nn.LayerNorm(latent_dim),
            nn.Tanh()  # Normalize to [-1,1]
        )
    
    def forward(self, fmri_signals):
        # fmriTrn/fmriTest → fmri_latent
        fmri_latent = self.encoder(fmri_signals)
        # Normalize ke unit sphere seperti CLIP
        fmri_latent = F.normalize(fmri_latent, p=2, dim=1)
        return fmri_latent