def train_latent_encoders(fmriTrn, stimTrn):
    """Train fMRI_encoder dan Shape_encoder"""
    
    for epoch in range(epochs):
        for fmri_batch, stim_batch in dataloader:
            # Encode ke latent space
            fmri_latent = fmri_encoder(fmri_batch)     # 967 → 512
            stim_latent = shape_encoder(stim_batch)    # 784 → 512
            
            # CLIP correlation loss
            corr_loss = clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
            
            # Reconstruction losses (optional)
            recon_loss_fmri = F.mse_loss(fmri_decoder(fmri_latent), fmri_batch)
            recon_loss_stim = F.mse_loss(shape_decoder(stim_latent), stim_batch)
            
            # Combined loss
            total_loss = corr_loss + 0.1 * (recon_loss_fmri + recon_loss_stim)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()