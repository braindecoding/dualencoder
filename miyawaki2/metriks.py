#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for Miyawaki Dataset
Includes pixel-level, perceptual, and shape-specific metrics
"""

import torch
import torch.nn.functional as F
import numpy as np

# Optional advanced metrics
try:
    from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics")
    TORCHMETRICS_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Install with: pip install scikit-image")
    SKIMAGE_AVAILABLE = False

def evaluate_decoding_performance(stimPred, stimTest):
    """Comprehensive evaluation untuk Miyawaki shapes"""
    
    metrics = {
        # Pixel-level accuracy
        'mse': F.mse_loss(stimPred, stimTest).item(),
    }

    # Add SSIM if available
    if TORCHMETRICS_AVAILABLE:
        ssim_metric = StructuralSimilarityIndexMeasure()
        psnr_metric = PeakSignalNoiseRatio()
        metrics['ssim'] = ssim_metric(stimPred, stimTest).item()
        metrics['psnr'] = psnr_metric(stimPred, stimTest).item()
    elif SKIMAGE_AVAILABLE:
        # Fallback to scikit-image SSIM
        pred_np = stimPred.detach().cpu().numpy()
        test_np = stimTest.detach().cpu().numpy()
        ssim_scores = []
        for i in range(pred_np.shape[0]):
            ssim_score = structural_similarity(pred_np[i, 0], test_np[i, 0], data_range=1.0)
            ssim_scores.append(ssim_score)
        metrics['ssim'] = np.mean(ssim_scores)

    # Add LPIPS if available
    if LPIPS_AVAILABLE:
        lpips_metric = lpips.LPIPS(net='alex')
        # Convert to 3-channel for LPIPS
        pred_3ch = stimPred.repeat(1, 3, 1, 1)
        test_3ch = stimTest.repeat(1, 3, 1, 1)
        metrics['lpips'] = lpips_metric(pred_3ch, test_3ch).mean().item()

    # Basic correlation metrics
    metrics['pixel_correlation'] = compute_pixel_correlation(stimPred, stimTest)
    metrics['cosine_similarity'] = compute_cosine_similarity(stimPred, stimTest)
    
    return metrics

def compute_pixel_correlation(pred, target):
    """Compute pixel-wise correlation between predictions and targets"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
    return correlation.item() if not torch.isnan(correlation) else 0.0

def compute_cosine_similarity(pred, target):
    """Compute cosine similarity between flattened predictions and targets"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    cosine_sim = F.cosine_similarity(pred_flat.unsqueeze(0), target_flat.unsqueeze(0))
    return cosine_sim.item()

# Specific evaluation functions
def evaluate_diffusion_performance(stimPred_diff, stimTest):
    """Evaluate diffusion decoder performance"""
    metrics_diff = evaluate_decoding_performance(stimPred_diff, stimTest)
    return metrics_diff

def evaluate_gan_performance(stimPred_gan, stimTest):
    """Evaluate GAN decoder performance"""
    metrics_gan = evaluate_decoding_performance(stimPred_gan, stimTest)
    return metrics_gan

def compare_decoders(stimPred_diff, stimPred_gan, stimTest):
    """Compare performance of diffusion vs GAN decoders"""
    diff_metrics = evaluate_diffusion_performance(stimPred_diff, stimTest)
    gan_metrics = evaluate_gan_performance(stimPred_gan, stimTest)

    comparison = {
        'diffusion': diff_metrics,
        'gan': gan_metrics,
        'winner': {}
    }

    # Determine winner for each metric
    for metric in diff_metrics.keys():
        if metric in ['mse', 'lpips']:  # Lower is better
            comparison['winner'][metric] = 'diffusion' if diff_metrics[metric] < gan_metrics[metric] else 'gan'
        else:  # Higher is better
            comparison['winner'][metric] = 'diffusion' if diff_metrics[metric] > gan_metrics[metric] else 'gan'

    return comparison