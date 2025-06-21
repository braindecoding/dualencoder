def evaluate_decoding_performance(stimPred, stimTest):
    """Comprehensive evaluation untuk Miyawaki shapes"""
    
    metrics = {
        # Pixel-level accuracy
        'mse': F.mse_loss(stimPred, stimTest),
        'ssim': structural_similarity(stimPred, stimTest),
        'psnr': peak_signal_noise_ratio(stimPred, stimTest),
        
        # Perceptual metrics
        'lpips': lpips_distance(stimPred, stimTest),
        'clip_similarity': clip_cosine_similarity(stimPred, stimTest),
        
        # Shape-specific metrics untuk Miyawaki
        'shape_accuracy': shape_classification_accuracy(stimPred, stimTest),
        'contour_similarity': contour_matching_score(stimPred, stimTest),
        'edge_detection_score': edge_preservation_metric(stimPred, stimTest),
        
        # Correlation metrics
        'latent_correlation': correlation_in_latent_space(stimPred, stimTest),
        'semantic_preservation': semantic_consistency_score(stimPred, stimTest)
    }
    
    return metrics

# Specific evaluation functions
def evaluate_diffusion_performance():
    metrics_diff = evaluate_decoding_performance(stimPred_diff, stimTest)
    return metrics_diff

def evaluate_gan_performance():
    metrics_gan = evaluate_decoding_performance(stimPred_gan, stimTest)
    return metrics_gan