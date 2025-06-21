class fMRI_Encoder(nn.Module):
    def __init__(self, fmri_dim, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fmri_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(), 
            nn.Dropout(0.08),
            
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()  # Normalize to [-1,1]
        )
    
    def forward(self, fmri_signals):
        # X → X_lat
        X_lat = self.encoder(fmri_signals)
        # Normalize ke unit sphere seperti CLIP
        X_lat = F.normalize(X_lat, p=2, dim=1)
        return X_lat