class Diffusion_Decoder(nn.Module):
    def __init__(self, correlation_dim=512, output_shape=(20, 20)):
        super().__init__()
        self.output_shape = output_shape
        
        # U-Net untuk diffusion process
        self.unet = UNet2D(
            in_channels=1,
            condition_dim=correlation_dim,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            norm_num_groups=8
        )
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
    
    def forward(self, CLIP_corr, X_test_lat, num_inference_steps=50):
        """
        Matlab equivalent: Y_result = Diffusion(CLIP_corr, X_test)
        """
        batch_size = X_test_lat.size(0)
        
        # Initialize dengan pure noise
        noise = torch.randn(batch_size, 1, *self.output_shape, device=X_test_lat.device)
        
        # Combine correlation dengan test latent
        condition = CLIP_corr + 0.3 * X_test_lat  # Weighted combination
        
        # Denoising process
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.unet(noise, t, condition)
            
            # Remove noise step
            noise = self.scheduler.step(noise_pred, t, noise).prev_sample
        
        # Final result
        Y_result = torch.sigmoid(noise)  # Normalize to [0,1]
        return Y_result