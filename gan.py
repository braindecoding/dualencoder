class GAN_Decoder(nn.Module):
    def __init__(self, correlation_dim=512, output_shape=(20, 20)):
        super().__init__()
        self.output_shape = output_shape
        
        # Generator
        self.generator = nn.Sequential(
            # Input: correlation + test latent
            nn.Linear(correlation_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            
            # Upsampling layers
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Discriminator untuk adversarial training
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, CLIP_corr, X_test_lat):
        """
        Matlab equivalent: Y_result = GAN(CLIP_corr, X_test)
        """
        # Combine inputs
        combined_input = torch.cat([CLIP_corr, X_test_lat], dim=1)
        
        # Generate result
        Y_result = self.generator(combined_input)
        return Y_result