class GAN_Decoder(nn.Module):
    def __init__(self, correlation_dim=512, output_shape=(28, 28)):
        super().__init__()
        self.output_shape = output_shape  # Miyawaki: 28x28
        
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
            
            # Upsampling layers untuk 28x28 output
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 1, 5, 1, 2),              # 32x32 → 28x28 (crop)
            nn.Sigmoid()
        )
        
        # Discriminator untuk adversarial training
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),              # 28x28 → 14x14
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),            # 14x14 → 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),           # 7x7 → 3x3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, CLIP_corr, fmriTest_latent):
        """
        Matlab equivalent: stimPred_gan = GAN(CLIP_corr, fmriTest_latent)
        """
        # Combine inputs
        combined_input = torch.cat([CLIP_corr, fmriTest_latent], dim=1)
        
        # Generate result: stimPred_gan
        stimPred_gan = self.generator(combined_input)
        return stimPred_gan