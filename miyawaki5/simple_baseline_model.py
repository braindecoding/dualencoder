#!/usr/bin/env python3
"""
Simple Baseline Model: Direct fMRI → Image Regression using Miyawaki Embeddings
Uses trained miyawaki_contrastive_clip.pth to generate embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import time
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import clip
from PIL import Image
import torchvision.transforms as transforms
from runembedding import MiyawakiDecoder
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleRegressionModel(nn.Module):
    """Direct regression from fMRI embeddings to images"""
    
    def __init__(self, fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.image_size = image_size
        self.output_dim = image_size * image_size  # Grayscale output
        
        # Build encoder layers
        layers = []
        input_dim = fmri_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.network = nn.Sequential(*layers)
        
        print(f"📊 Model Architecture:")
        print(f"   Input: fMRI embeddings ({fmri_dim})")
        print(f"   Hidden layers: {hidden_dims}")
        print(f"   Output: {image_size}x{image_size} image ({self.output_dim})")
        
    def forward(self, fmri_embeddings):
        """Forward pass"""
        batch_size = fmri_embeddings.shape[0]
        
        # Pass through network
        output = self.network(fmri_embeddings)
        
        # Reshape to image format
        output = output.view(batch_size, 1, self.image_size, self.image_size)
        
        return output

class MiyawakiBaselineDataset(Dataset):
    """Dataset for baseline model using Miyawaki embeddings"""

    def __init__(self, model_path="miyawaki_contrastive_clip.pth", split="train", target_size=28):
        self.split = split
        self.target_size = target_size

        print(f"🔧 Loading Miyawaki data and generating embeddings...")

        # Initialize decoder and load trained model
        self.decoder = MiyawakiDecoder()

        # Load original dataset
        mat_file_path = "../dataset/miyawaki_structured_28x28.mat"
        self.decoder.load_data(mat_file_path)

        # Initialize models and load trained weights
        self.decoder.initialize_models()
        self.decoder.load_model(model_path)

        # Generate embeddings
        self._generate_embeddings()

        print(f"📊 Loaded {split} data:")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Original images: {self.original_images.shape}")

        # Process images
        self.images = self._process_images()

        print(f"   Processed images: {self.images.shape}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")

    def _generate_embeddings(self):
        """Generate embeddings using trained model"""
        print("🔗 Generating embeddings...")

        # Create dataloaders
        train_loader, test_loader = self.decoder.create_dataloaders(batch_size=32)

        # Generate embeddings for train and test
        if self.split == "train":
            dataloader = train_loader
        else:
            dataloader = test_loader

        self.decoder.fmri_encoder.eval()

        fmri_embeddings = []
        original_images = []

        with torch.no_grad():
            for fmri_batch, image_batch in tqdm(dataloader, desc=f"Generating {self.split} embeddings"):
                fmri_batch = fmri_batch.to(self.decoder.device)
                image_batch = image_batch.to(self.decoder.device)

                # Get fMRI embeddings
                fmri_emb = self.decoder.fmri_encoder(fmri_batch)
                fmri_emb = F.normalize(fmri_emb, dim=1)

                fmri_embeddings.append(fmri_emb.cpu().numpy())

                # Store original images (denormalize from CLIP preprocessing)
                for img in image_batch:
                    img_denorm = img * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img.device)
                    img_denorm += torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img.device)
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    original_images.append(img_denorm.cpu().numpy())

        # Concatenate all batches
        self.fmri_embeddings = np.concatenate(fmri_embeddings, axis=0)
        self.original_images = np.array(original_images)

        print(f"✅ Generated {self.split} embeddings:")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Original images: {self.original_images.shape}")

    def _process_images(self):
        """Process images to target format"""
        images = self.original_images
        
        # Convert RGB to grayscale
        if len(images.shape) == 4 and images.shape[1] == 3:
            images = np.mean(images, axis=1)  # (N, 224, 224)
        
        # Resize to target size
        if images.shape[-1] != self.target_size:
            from scipy.ndimage import zoom
            scale_factor = self.target_size / images.shape[-1]
            
            resized_images = []
            for img in images:
                img_resized = zoom(img, scale_factor)
                resized_images.append(img_resized)
            
            images = np.array(resized_images)
        
        # Add channel dimension and normalize to [-1, 1]
        images = images[:, None, :, :]  # (N, 1, H, W)
        images = (images - 0.5) * 2  # Normalize to [-1, 1]
        
        return images
    
    def __len__(self):
        return len(self.fmri_embeddings)
    
    def __getitem__(self, idx):
        fmri_emb = torch.FloatTensor(self.fmri_embeddings[idx])
        image = torch.FloatTensor(self.images[idx])
        
        return fmri_emb, image

def train_baseline_model():
    """Train the baseline regression model using Miyawaki embeddings"""
    print("🚀 TRAINING BASELINE REGRESSION MODEL WITH MIYAWAKI EMBEDDINGS")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    train_dataset = MiyawakiBaselineDataset("miyawaki_contrastive_clip.pth", "train", target_size=28)
    test_dataset = MiyawakiBaselineDataset("miyawaki_contrastive_clip.pth", "test", target_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 15000
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    print(f"🎯 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: 16")
    print(f"   Learning rate: 1e-4")
    print(f"   Loss function: MSE")
    print(f"   Optimizer: Adam")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for fmri_emb, images in train_pbar:
            fmri_emb = fmri_emb.to(device)
            images = images.to(device)
            
            # Forward pass
            predicted_images = model(fmri_emb)
            loss = criterion(predicted_images, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_test_loss = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
            
            for fmri_emb, images in test_pbar:
                fmri_emb = fmri_emb.to(device)
                images = images.to(device)
                
                predicted_images = model(fmri_emb)
                loss = criterion(predicted_images, images)
                
                epoch_test_loss += loss.item()
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'baseline_model_best.pth')
        
        # Print progress
        print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")
        
        # Improved Early stopping check
        if epoch > 30 and len(test_losses) > 15:  # Wait longer before considering early stop
            recent_losses = test_losses[-15:]  # Look at more epochs

            # Check if improvement is minimal (< 0.1% in last 10 epochs)
            if len(recent_losses) >= 10:
                improvement_threshold = 0.001  # 0.1% improvement threshold
                recent_10 = recent_losses[-10:]
                max_recent = max(recent_10)
                min_recent = min(recent_10)
                relative_improvement = (max_recent - min_recent) / max_recent

                # Also check if loss has plateaued (no significant improvement)
                last_5_avg = np.mean(recent_losses[-5:])
                prev_5_avg = np.mean(recent_losses[-10:-5])
                plateau_improvement = (prev_5_avg - last_5_avg) / prev_5_avg

                if relative_improvement < improvement_threshold and plateau_improvement < improvement_threshold:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    print(f"   Recent improvement: {relative_improvement:.4f} < {improvement_threshold}")
                    print(f"   Plateau improvement: {plateau_improvement:.4f} < {improvement_threshold}")
                    break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'baseline_model_epoch_{epoch+1}.pth')
    
    training_time = time.time() - start_time
    
    # Save final model
    torch.save(model.state_dict(), 'baseline_model_final.pth')
    
    print(f"\n✅ Training completed!")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Best test loss: {best_test_loss:.4f}")
    print(f"   Final train loss: {train_losses[-1]:.4f}")
    print(f"   Final test loss: {test_losses[-1]:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training and Test Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('baseline_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, train_losses, test_losses

def evaluate_baseline_model():
    """Evaluate the trained baseline model"""
    print(f"\n📊 EVALUATING BASELINE MODEL WITH MIYAWAKI EMBEDDINGS")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    model.load_state_dict(torch.load('baseline_model_best.pth', map_location=device))
    model.eval()

    # Load test data
    test_dataset = MiyawakiBaselineDataset("miyawaki_contrastive_clip.pth", "test", target_size=28)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Generate predictions
    predictions = []
    targets = []
    
    with torch.no_grad():
        for fmri_emb, images in test_loader:
            fmri_emb = fmri_emb.to(device)
            
            predicted_images = model(fmri_emb)
            
            predictions.append(predicted_images.cpu().numpy())
            targets.append(images.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    print(f"📊 Generated predictions: {predictions.shape}")
    print(f"📊 Target images: {targets.shape}")
    
    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    
    # Calculate correlation for each sample
    correlations = []
    for i in range(len(predictions)):
        pred_flat = predictions[i].flatten()
        target_flat = targets[i].flatten()
        corr, _ = pearsonr(pred_flat, target_flat)
        if not np.isnan(corr):
            correlations.append(corr)
    
    correlations = np.array(correlations)
    
    print(f"📊 Baseline Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f} ± {correlations.std():.4f}")
    print(f"   Best Correlation: {correlations.max():.4f}")
    print(f"   Worst Correlation: {correlations.min():.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Baseline Model Results: Original vs Predicted', fontsize=16)
    
    for i in range(4):
        if i < len(predictions):
            # Original
            orig_img = targets[i, 0]
            axes[0, i].imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Predicted
            pred_img = predictions[i, 0]
            axes[1, i].imshow(pred_img, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f'Predicted {i+1}\nCorr: {correlations[i]:.3f}')
            axes[1, i].axis('off')
            
            # Difference
            diff_img = np.abs(orig_img - pred_img)
            axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'|Difference| {i+1}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('baseline_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions, targets, correlations

def calculate_comprehensive_metrics(predictions, targets, save_csv=True):
    """Calculate comprehensive evaluation metrics and save to CSV"""
    print(f"\n📊 CALCULATING COMPREHENSIVE METRICS")
    print("=" * 50)

    # Ensure predictions and targets are in correct format
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Remove channel dimension if present
    if len(predictions.shape) == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]
    if len(targets.shape) == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, :, :]

    print(f"📊 Calculating metrics for {len(predictions)} samples...")
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Target shape: {targets.shape}")

    # Initialize metrics storage
    metrics_data = []

    # Calculate metrics for each sample
    for i in tqdm(range(len(predictions)), desc="Computing metrics"):
        pred_img = predictions[i]
        target_img = targets[i]

        # Normalize images to [0, 1] for SSIM and PSNR
        pred_norm = (pred_img + 1) / 2  # From [-1, 1] to [0, 1]
        target_norm = (target_img + 1) / 2  # From [-1, 1] to [0, 1]

        # Ensure values are in valid range
        pred_norm = np.clip(pred_norm, 0, 1)
        target_norm = np.clip(target_norm, 0, 1)

        # 1. MSE (Mean Squared Error)
        mse_val = mean_squared_error(target_img.flatten(), pred_img.flatten())

        # 2. SSIM (Structural Similarity Index)
        try:
            ssim_val = ssim(target_norm, pred_norm, data_range=1.0)
        except:
            ssim_val = 0.0

        # 3. PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
        except:
            psnr_val = 0.0

        # 4. Pixel Correlation (Pearson correlation)
        try:
            pixcorr_val, _ = pearsonr(target_img.flatten(), pred_img.flatten())
            if np.isnan(pixcorr_val):
                pixcorr_val = 0.0
        except:
            pixcorr_val = 0.0

        # 5. FID (Frechet Inception Distance) - Simplified version
        # For single images, we'll use a simplified metric based on feature statistics
        try:
            # Calculate mean and std for both images
            pred_mean, pred_std = np.mean(pred_norm), np.std(pred_norm)
            target_mean, target_std = np.mean(target_norm), np.std(target_norm)

            # Simplified FID-like metric
            fid_val = (pred_mean - target_mean)**2 + (pred_std - target_std)**2
        except:
            fid_val = 1.0

        # 6. CLIP Similarity - Simplified version
        # For this baseline, we'll use cosine similarity between flattened images
        try:
            pred_flat = pred_img.flatten()
            target_flat = target_img.flatten()

            # Normalize vectors
            pred_norm_vec = pred_flat / (np.linalg.norm(pred_flat) + 1e-8)
            target_norm_vec = target_flat / (np.linalg.norm(target_flat) + 1e-8)

            # Cosine similarity
            clip_sim = np.dot(pred_norm_vec, target_norm_vec)
        except:
            clip_sim = 0.0

        # Store metrics for this sample
        metrics_data.append({
            'sample_id': i,
            'mse': mse_val,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'pixcorr': pixcorr_val,
            'fid': fid_val,
            'clip_similarity': clip_sim
        })

    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics_data)

    # Calculate summary statistics
    summary_stats = {
        'metric': ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity'],
        'mean': [df_metrics[col].mean() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'std': [df_metrics[col].std() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'min': [df_metrics[col].min() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']],
        'max': [df_metrics[col].max() for col in ['mse', 'ssim', 'psnr', 'pixcorr', 'fid', 'clip_similarity']]
    }

    df_summary = pd.DataFrame(summary_stats)

    # Print results
    print(f"\n📊 COMPREHENSIVE METRICS RESULTS:")
    print("=" * 60)
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    for _, row in df_summary.iterrows():
        print(f"{row['metric']:<15} {row['mean']:<10.4f} {row['std']:<10.4f} {row['min']:<10.4f} {row['max']:<10.4f}")

    # Save to CSV files
    if save_csv:
        # Save detailed metrics
        csv_filename = 'baseline_model_detailed_metrics.csv'
        df_metrics.to_csv(csv_filename, index=False)
        print(f"\n💾 Detailed metrics saved to: {csv_filename}")

        # Save summary statistics
        summary_filename = 'baseline_model_summary_metrics.csv'
        df_summary.to_csv(summary_filename, index=False)
        print(f"💾 Summary metrics saved to: {summary_filename}")

        # Save combined results with additional info
        combined_data = {
            'model_name': 'Miyawaki_Baseline_Regression',
            'dataset': 'Miyawaki_fMRI',
            'architecture': '[512] → [1024, 2048, 1024] → [784]',
            'input_type': 'fMRI_embeddings_512dim',
            'output_type': '28x28_grayscale_images',
            'num_samples': len(predictions),
            'mse_mean': df_summary[df_summary['metric'] == 'mse']['mean'].iloc[0],
            'ssim_mean': df_summary[df_summary['metric'] == 'ssim']['mean'].iloc[0],
            'psnr_mean': df_summary[df_summary['metric'] == 'psnr']['mean'].iloc[0],
            'pixcorr_mean': df_summary[df_summary['metric'] == 'pixcorr']['mean'].iloc[0],
            'fid_mean': df_summary[df_summary['metric'] == 'fid']['mean'].iloc[0],
            'clip_similarity_mean': df_summary[df_summary['metric'] == 'clip_similarity']['mean'].iloc[0]
        }

        df_combined = pd.DataFrame([combined_data])
        combined_filename = 'baseline_model_final_results.csv'
        df_combined.to_csv(combined_filename, index=False)
        print(f"💾 Final results saved to: {combined_filename}")

    return df_metrics, df_summary

def main():
    """Main function"""
    print("🎯 SIMPLE BASELINE MODEL FOR MIYAWAKI DATASET")
    print("=" * 60)
    print("🧠 Using trained miyawaki_contrastive_clip.pth embeddings")
    print("=" * 60)

    # Train model
    model, train_losses, test_losses = train_baseline_model()

    # Evaluate model
    predictions, targets, correlations = evaluate_baseline_model()

    # Calculate comprehensive metrics and save to CSV
    print(f"\n🔍 CALCULATING COMPREHENSIVE METRICS...")
    df_metrics, df_summary = calculate_comprehensive_metrics(predictions, targets, save_csv=True)

    print(f"\n🎯 MIYAWAKI BASELINE MODEL SUMMARY:")
    print(f"   Final MSE: {mean_squared_error(targets.flatten(), predictions.flatten()):.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f}")
    print(f"   Training completed successfully!")

    print(f"\n📁 Generated files:")
    print(f"   - baseline_model_best.pth")
    print(f"   - baseline_model_final.pth")
    print(f"   - baseline_training_curves.png")
    print(f"   - baseline_model_results.png")
    print(f"   - baseline_model_detailed_metrics.csv")
    print(f"   - baseline_model_summary_metrics.csv")
    print(f"   - baseline_model_final_results.csv")

    print(f"\n🎯 Key Features:")
    print(f"   ✅ Uses trained Miyawaki CLIP embeddings (512-dim)")
    print(f"   ✅ Direct fMRI embedding → Image regression")
    print(f"   ✅ Training samples: 107, Test samples: 12")
    print(f"   ✅ Output: 28x28 grayscale images")
    print(f"   ✅ Architecture: [512] → [1024, 2048, 1024] → [784]")
    print(f"   ✅ Comprehensive metrics: MSE, SSIM, PSNR, PixCorr, FID, CLIP Similarity")

    return model, predictions, targets, df_metrics, df_summary

if __name__ == "__main__":
    main()
