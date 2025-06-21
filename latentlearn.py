def train_latent_encoders(fmri_data, shape_data):
    """Train X_encoder dan Y_encoder"""
    
    for epoch in range(epochs):
        for fmri_batch, shape_batch in dataloader:
            # Encode ke latent space
            X_lat = fmri_encoder(fmri_batch)
            Y_lat = shape_encoder(shape_batch)
            
            # CLIP correlation loss
            corr_loss = clip_correlation.compute_contrastive_loss(X_lat, Y_lat)
            
            # Reconstruction losses (optional)
            recon_loss_x = F.mse_loss(fmri_decoder(X_lat), fmri_batch)
            recon_loss_y = F.mse_loss(shape_decoder(Y_lat), shape_batch)
            
            # Combined loss
            total_loss = corr_loss + 0.1 * (recon_loss_x + recon_loss_y)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()