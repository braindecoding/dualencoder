def train_generative_backend(CLIP_corr, X_test, Y_test, backend='diffusion'):
    """Train Diffusion atau GAN decoder"""
    
    if backend == 'diffusion':
        for epoch in range(diffusion_epochs):
            # Diffusion training
            Y_pred = diffusion_decoder(CLIP_corr, X_test)
            diff_loss = F.mse_loss(Y_pred, Y_test)
            
            diff_optimizer.zero_grad()
            diff_loss.backward()
            diff_optimizer.step()
    
    elif backend == 'gan':
        for epoch in range(gan_epochs):
            # GAN training (alternating G and D)
            Y_pred = gan_decoder(CLIP_corr, X_test)
            
            # Generator loss
            gen_loss = gan_generator_loss(Y_pred, Y_test)
            
            # Discriminator loss  
            disc_loss = gan_discriminator_loss(Y_pred, Y_test)
            
            # Optimize alternately
            optimize_generator(gen_loss)
            optimize_discriminator(disc_loss)