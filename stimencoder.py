class Shape_Encoder(nn.Module):
    def __init__(self, shape_dim, latent_dim=512):
        super().__init__()
        # Untuk Miyawaki shapes (misal 20x20 pixels)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, shapes):
        # Y → Y_lat
        features = self.cnn_encoder(shapes)
        features = features.flatten(1)
        Y_lat = self.fc_encoder(features)
        Y_lat = F.normalize(Y_lat, p=2, dim=1)
        return Y_lat