def train_generative_backend(CLIP_corr, fmriTest_latent, stimTest, backend='diffusion'):
    """Train Diffusion atau GAN decoder"""
    
    if backend == 'diffusion':
        for epoch in range(diffusion_epochs):
            # Diffusion training
            stimPred_diff = diffusion_decoder(CLIP_corr, fmriTest_latent)
            diff_loss = F.mse_loss(stimPred_diff, stimTest)
            
            diff_optimizer.zero_grad()
            diff_loss.backward()
            diff_optimizer.step()
    
    elif backend == 'gan':
        for epoch in range(gan_epochs):
            # GAN training (alternating G and D)
            stimPred_gan = gan_decoder(CLIP_corr, fmriTest_latent)
            
            # Generator loss
            gen_loss = gan_generator_loss(stimPred_gan, stimTest)
            
            # Discriminator loss  
            disc_loss = gan_discriminator_loss(stimPred_gan, stimTest)
            
            # Optimize alternately
            optimize_generator(gen_loss)
            optimize_discriminator(disc_loss)