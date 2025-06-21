class MiyawakaTrainer:
    def __init__(self):
        # Initialize models
        self.fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512)
        self.shape_encoder = Shape_Encoder(latent_dim=512)
        self.clip_correlation = CLIP_Correlation(latent_dim=512)
        self.diffusion_decoder = Diffusion_Decoder(correlation_dim=512, output_shape=(28, 28))
        self.gan_decoder = GAN_Decoder(correlation_dim=512, output_shape=(28, 28))
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            list(self.fmri_encoder.parameters()) + list(self.shape_encoder.parameters()),
            lr=1e-4
        )
        self.correlation_optimizer = torch.optim.Adam(self.clip_correlation.parameters(), lr=1e-4)
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion_decoder.parameters(), lr=1e-4)
        self.gan_optimizer = torch.optim.Adam(self.gan_decoder.parameters(), lr=1e-4)
    
    def train_phase1_encoders(self, train_loader, epochs=100):
        """Phase 1: Train latent encoders"""
        for epoch in range(epochs):
            total_loss = 0
            for fmri_batch, stim_batch in train_loader:
                # Forward pass
                fmri_latent = self.fmri_encoder(fmri_batch)
                stim_latent = self.shape_encoder(stim_batch)
                
                # Contrastive loss
                corr_loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                
                # Backward pass
                self.encoder_optimizer.zero_grad()
                corr_loss.backward()
                self.encoder_optimizer.step()
                
                total_loss += corr_loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_phase2_correlation(self, train_loader, epochs=50):
        """Phase 2: Train correlation learning"""
        for epoch in range(epochs):
            total_loss = 0
            for fmri_batch, stim_batch in train_loader:
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri_batch)
                    stim_latent = self.shape_encoder(stim_batch)
                
                # Train correlation
                CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)
                corr_loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                
                self.correlation_optimizer.zero_grad()
                corr_loss.backward()
                self.correlation_optimizer.step()
                
                total_loss += corr_loss.item()
            
            print(f"Correlation Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    def train_phase3_decoders(self, train_loader, test_loader, epochs=100):
        """Phase 3: Train generative decoders"""
        for epoch in range(epochs):
            # Train diffusion
            diff_loss_total = 0
            for fmri_batch, stim_batch in train_loader:
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri_batch)
                    stim_latent = self.shape_encoder(stim_batch)
                    CLIP_corr = self.clip_correlation(fmri_latent, stim_latent)
                
                # Get test fMRI latents (simulate testing)
                fmriTest_latent = fmri_latent  # In practice, use separate test data
                
                # Train diffusion
                stimPred_diff = self.diffusion_decoder(CLIP_corr, fmriTest_latent)
                diff_loss = F.mse_loss(stimPred_diff, stim_batch)
                
                self.diffusion_optimizer.zero_grad()
                diff_loss.backward()
                self.diffusion_optimizer.step()
                
                diff_loss_total += diff_loss.item()
            
            print(f"Diffusion Epoch {epoch}, Loss: {diff_loss_total/len(train_loader):.4f}")