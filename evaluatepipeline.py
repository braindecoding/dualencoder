def evaluate_model(trainer, test_loader):
    """Evaluate both diffusion and GAN on test set"""
    trainer.fmri_encoder.eval()
    trainer.shape_encoder.eval()
    trainer.clip_correlation.eval()
    trainer.diffusion_decoder.eval()
    trainer.gan_decoder.eval()
    
    all_stimPred_diff = []
    all_stimPred_gan = []
    all_stimTest = []
    
    with torch.no_grad():
        for fmriTest_batch, stimTest_batch in test_loader:
            # Encode test fMRI
            fmriTest_latent = trainer.fmri_encoder(fmriTest_batch)
            
            # Get correlation (using trained correlation from training data)
            # Note: In practice, you need to store CLIP_corr from training phase
            CLIP_corr = get_stored_correlation(fmriTest_latent)  # Implementation needed
            
            # Generate predictions
            stimPred_diff = trainer.diffusion_decoder(CLIP_corr, fmriTest_latent)
            stimPred_gan = trainer.gan_decoder(CLIP_corr, fmriTest_latent)
            
            all_stimPred_diff.append(stimPred_diff)
            all_stimPred_gan.append(stimPred_gan)
            all_stimTest.append(stimTest_batch)
    
    # Concatenate all results
    stimPred_diff = torch.cat(all_stimPred_diff, dim=0)
    stimPred_gan = torch.cat(all_stimPred_gan, dim=0)
    stimTest = torch.cat(all_stimTest, dim=0)
    
    # Evaluate
    accuracy_diff = evaluate_decoding_performance(stimPred_diff, stimTest)
    accuracy_gan = evaluate_decoding_performance(stimPred_gan, stimTest)
    
    return accuracy_diff, accuracy_gan