#!/usr/bin/env python3
"""
Comprehensive Performance Metrics
Calculate all standard metrics: PixCorr, SSIM, MSE, Inception Distance, CLIP Similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from quick_fix_training import QuickFixTrainer
import warnings
warnings.filterwarnings("ignore")

# Try to import additional libraries for advanced metrics
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. CLIP similarity will be skipped.")

try:
    from torchvision.models import inception_v3
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    print("⚠️ Inception model not available. Inception Distance will be skipped.")

class ComprehensiveMetrics:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load CLIP model if available
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            print("✅ CLIP model loaded")
        
        # Load Inception model if available
        if INCEPTION_AVAILABLE:
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
            self.inception_model.eval()
            print("✅ Inception model loaded")
    
    def pixel_correlation(self, pred_images, target_images):
        """Calculate pixel-wise correlation (PixCorr)"""
        correlations = []
        
        for pred, target in zip(pred_images, target_images):
            # Flatten images
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            
            # Calculate Pearson correlation
            if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
                corr, _ = pearsonr(pred_flat, target_flat)
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        return {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'individual': correlations
        }
    
    def structural_similarity(self, pred_images, target_images):
        """Calculate Structural Similarity Index (SSIM)"""
        ssim_scores = []
        
        for pred, target in zip(pred_images, target_images):
            # Ensure images are in correct format
            if len(pred.shape) == 3:
                pred = np.mean(pred, axis=2)
            if len(target.shape) == 3:
                target = np.mean(target, axis=2)
            
            # Calculate SSIM
            score = ssim(target, pred, data_range=pred.max() - pred.min())
            ssim_scores.append(score)
        
        return {
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores),
            'individual': ssim_scores
        }
    
    def mean_squared_error(self, pred_images, target_images):
        """Calculate Mean Squared Error (MSE)"""
        mse_scores = []
        
        for pred, target in zip(pred_images, target_images):
            # Normalize to [0, 1]
            pred_norm = pred.astype(np.float32) / 255.0
            target_norm = target.astype(np.float32) / 255.0
            
            # Calculate MSE
            mse = np.mean((pred_norm - target_norm) ** 2)
            mse_scores.append(mse)
        
        return {
            'mean': np.mean(mse_scores),
            'std': np.std(mse_scores),
            'individual': mse_scores
        }
    
    def peak_signal_noise_ratio(self, pred_images, target_images):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
        psnr_scores = []
        
        for pred, target in zip(pred_images, target_images):
            # Normalize to [0, 1]
            pred_norm = pred.astype(np.float32) / 255.0
            target_norm = target.astype(np.float32) / 255.0
            
            # Calculate MSE
            mse = np.mean((pred_norm - target_norm) ** 2)
            
            # Calculate PSNR
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            psnr_scores.append(psnr)
        
        return {
            'mean': np.mean([p for p in psnr_scores if p != float('inf')]),
            'std': np.std([p for p in psnr_scores if p != float('inf')]),
            'individual': psnr_scores
        }
    
    def clip_similarity(self, pred_images, target_images):
        """Calculate CLIP similarity"""
        if not CLIP_AVAILABLE:
            return {'mean': 0.0, 'std': 0.0, 'individual': [], 'note': 'CLIP not available'}
        
        similarities = []
        
        with torch.no_grad():
            for pred, target in zip(pred_images, target_images):
                # Convert to PIL Images
                pred_pil = Image.fromarray(pred.astype(np.uint8))
                target_pil = Image.fromarray(target.astype(np.uint8))
                
                # Convert to RGB if grayscale
                if pred_pil.mode != 'RGB':
                    pred_pil = pred_pil.convert('RGB')
                if target_pil.mode != 'RGB':
                    target_pil = target_pil.convert('RGB')
                
                # Preprocess for CLIP
                pred_tensor = self.clip_preprocess(pred_pil).unsqueeze(0).to(self.device)
                target_tensor = self.clip_preprocess(target_pil).unsqueeze(0).to(self.device)
                
                # Get CLIP features
                pred_features = self.clip_model.encode_image(pred_tensor)
                target_features = self.clip_model.encode_image(target_tensor)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(pred_features, target_features).item()
                similarities.append(similarity)
        
        return {
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'individual': similarities
        }
    
    def inception_distance(self, pred_images, target_images):
        """Calculate Inception Distance (simplified FID)"""
        if not INCEPTION_AVAILABLE:
            return {'mean': 0.0, 'std': 0.0, 'individual': [], 'note': 'Inception not available'}
        
        def get_inception_features(images):
            features = []
            with torch.no_grad():
                for img in images:
                    # Convert to RGB and resize to 299x299 for Inception
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis=2)
                    
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_resized = img_pil.resize((299, 299))
                    
                    # Convert to tensor
                    img_tensor = torch.FloatTensor(np.array(img_resized)).permute(2, 0, 1)
                    img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0
                    
                    # Get Inception features
                    feat = self.inception_model(img_tensor)
                    features.append(feat.cpu().numpy().flatten())
            
            return np.array(features)
        
        # Get features
        pred_features = get_inception_features(pred_images)
        target_features = get_inception_features(target_images)
        
        # Calculate distances
        distances = []
        for pred_feat, target_feat in zip(pred_features, target_features):
            dist = np.linalg.norm(pred_feat - target_feat)
            distances.append(dist)
        
        return {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'individual': distances
        }
    
    def binary_specific_metrics(self, pred_images, target_images):
        """Calculate binary-specific metrics"""
        metrics = {
            'binary_accuracy': [],
            'dice_coefficient': [],
            'jaccard_index': [],
            'edge_similarity': []
        }
        
        for pred, target in zip(pred_images, target_images):
            # Binarize images
            pred_binary = (pred > 128).astype(np.uint8)
            target_binary = (target > 128).astype(np.uint8)
            
            # Binary accuracy
            accuracy = np.mean(pred_binary == target_binary)
            metrics['binary_accuracy'].append(accuracy)
            
            # Dice coefficient
            intersection = np.sum(pred_binary * target_binary)
            dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
            metrics['dice_coefficient'].append(dice)
            
            # Jaccard index (IoU)
            union = np.sum((pred_binary + target_binary) > 0)
            jaccard = intersection / (union + 1e-8)
            metrics['jaccard_index'].append(jaccard)
            
            # Edge similarity
            pred_edges = self.detect_edges(pred_binary)
            target_edges = self.detect_edges(target_binary)
            edge_sim = np.mean(pred_edges == target_edges)
            metrics['edge_similarity'].append(edge_sim)
        
        # Calculate statistics
        result = {}
        for metric_name, values in metrics.items():
            result[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'individual': values
            }
        
        return result
    
    def detect_edges(self, image):
        """Simple edge detection using gradient"""
        grad_x = np.abs(np.diff(image, axis=1))
        grad_y = np.abs(np.diff(image, axis=0))
        
        edges = np.zeros_like(image)
        edges[:, :-1] += grad_x
        edges[:-1, :] += grad_y
        
        return (edges > 0).astype(np.uint8)

def load_generated_and_target_images():
    """Load generated images and corresponding targets"""
    print("📊 Loading generated and target images...")
    
    # Load embeddings to get target images
    embeddings_path = "miyawaki4_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = QuickFixTrainer(device=device)
    
    model_path = 'quickfix_binary_generator.pth'
    trainer.model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Generate predictions for test set
    test_fmri = embeddings_data['test']['fmri_embeddings']
    test_images = embeddings_data['test']['original_images']
    
    generated_images = []
    target_images = []
    
    print(f"🎨 Generating {len(test_fmri)} predictions...")
    
    for i, (fmri, orig_img) in enumerate(zip(test_fmri, test_images)):
        # Generate prediction
        fmri_tensor = torch.FloatTensor(fmri).to(device)
        pred_img, _ = trainer.generate_binary_pattern(fmri_tensor)
        generated_images.append(np.array(pred_img))
        
        # Process target image
        if len(orig_img.shape) == 3 and orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
        if len(orig_img.shape) == 3:
            orig_img = np.mean(orig_img, axis=2)
        
        target_binary = (orig_img > 0.5).astype(np.uint8) * 255
        target_images.append(target_binary)
        
        print(f"   Generated {i+1}/{len(test_fmri)}")
    
    return generated_images, target_images

def calculate_all_metrics():
    """Calculate all performance metrics"""
    print("🔍 COMPREHENSIVE PERFORMANCE METRICS CALCULATION")
    print("=" * 70)
    
    # Load images
    generated_images, target_images = load_generated_and_target_images()
    
    # Initialize metrics calculator
    metrics_calc = ComprehensiveMetrics()
    
    print(f"\n📊 Calculating metrics for {len(generated_images)} image pairs...")
    
    # Calculate all metrics
    results = {}
    
    print("🔍 1. Pixel Correlation (PixCorr)...")
    results['pixel_correlation'] = metrics_calc.pixel_correlation(generated_images, target_images)
    
    print("🔍 2. Structural Similarity (SSIM)...")
    results['ssim'] = metrics_calc.structural_similarity(generated_images, target_images)
    
    print("🔍 3. Mean Squared Error (MSE)...")
    results['mse'] = metrics_calc.mean_squared_error(generated_images, target_images)
    
    print("🔍 4. Peak Signal-to-Noise Ratio (PSNR)...")
    results['psnr'] = metrics_calc.peak_signal_noise_ratio(generated_images, target_images)
    
    print("🔍 5. CLIP Similarity...")
    results['clip_similarity'] = metrics_calc.clip_similarity(generated_images, target_images)
    
    print("🔍 6. Inception Distance...")
    results['inception_distance'] = metrics_calc.inception_distance(generated_images, target_images)
    
    print("🔍 7. Binary-Specific Metrics...")
    results['binary_metrics'] = metrics_calc.binary_specific_metrics(generated_images, target_images)
    
    return results

def create_metrics_report(results):
    """Create comprehensive metrics report"""
    print("\n📋 COMPREHENSIVE PERFORMANCE METRICS REPORT")
    print("=" * 70)
    
    print("🎯 STANDARD COMPUTER VISION METRICS:")
    print("-" * 50)
    
    # Pixel Correlation
    pixcorr = results['pixel_correlation']
    print(f"📊 Pixel Correlation (PixCorr):")
    print(f"   Mean: {pixcorr['mean']:.4f} ± {pixcorr['std']:.4f}")
    print(f"   Range: [{min(pixcorr['individual']):.4f}, {max(pixcorr['individual']):.4f}]")
    
    # SSIM
    ssim_res = results['ssim']
    print(f"\n📊 Structural Similarity (SSIM):")
    print(f"   Mean: {ssim_res['mean']:.4f} ± {ssim_res['std']:.4f}")
    print(f"   Range: [{min(ssim_res['individual']):.4f}, {max(ssim_res['individual']):.4f}]")
    
    # MSE
    mse_res = results['mse']
    print(f"\n📊 Mean Squared Error (MSE):")
    print(f"   Mean: {mse_res['mean']:.6f} ± {mse_res['std']:.6f}")
    print(f"   Range: [{min(mse_res['individual']):.6f}, {max(mse_res['individual']):.6f}]")
    
    # PSNR
    psnr_res = results['psnr']
    print(f"\n📊 Peak Signal-to-Noise Ratio (PSNR):")
    print(f"   Mean: {psnr_res['mean']:.2f} ± {psnr_res['std']:.2f} dB")
    
    print("\n🤖 DEEP LEARNING METRICS:")
    print("-" * 50)
    
    # CLIP Similarity
    clip_res = results['clip_similarity']
    if 'note' not in clip_res:
        print(f"📊 CLIP Similarity:")
        print(f"   Mean: {clip_res['mean']:.4f} ± {clip_res['std']:.4f}")
        print(f"   Range: [{min(clip_res['individual']):.4f}, {max(clip_res['individual']):.4f}]")
    else:
        print(f"📊 CLIP Similarity: {clip_res['note']}")
    
    # Inception Distance
    inception_res = results['inception_distance']
    if 'note' not in inception_res:
        print(f"\n📊 Inception Distance:")
        print(f"   Mean: {inception_res['mean']:.4f} ± {inception_res['std']:.4f}")
    else:
        print(f"\n📊 Inception Distance: {inception_res['note']}")
    
    print("\n🎯 BINARY-SPECIFIC METRICS:")
    print("-" * 50)
    
    binary_res = results['binary_metrics']
    for metric_name, metric_data in binary_res.items():
        display_name = metric_name.replace('_', ' ').title()
        print(f"📊 {display_name}:")
        print(f"   Mean: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
    
    return results

if __name__ == "__main__":
    # Calculate all metrics
    results = calculate_all_metrics()
    
    # Create report
    final_results = create_metrics_report(results)
    
    # Save results
    with open('comprehensive_metrics_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n✅ METRICS CALCULATION COMPLETE!")
    print(f"📁 Results saved: comprehensive_metrics_results.pkl")
    print(f"🎉 All standard performance metrics calculated!")
