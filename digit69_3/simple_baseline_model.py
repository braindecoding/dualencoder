#!/usr/bin/env python3
"""
Simple Baseline Model: Direct fMRI → Image Regression
Validate concept before complex diffusion models
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

class Digit69BaselineDataset(Dataset):
    """Dataset for baseline model"""
    
    def __init__(self, embeddings_file="digit69_embeddings.pkl", split="train", target_size=28):
        self.split = split
        self.target_size = target_size
        
        # Load data
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.fmri_embeddings = data[split]['fmri_embeddings']
        self.original_images = data[split]['original_images']
        
        print(f"📊 Loaded {split} data:")
        print(f"   fMRI embeddings: {self.fmri_embeddings.shape}")
        print(f"   Original images: {self.original_images.shape}")
        
        # Process images
        self.images = self._process_images()
        
        print(f"   Processed images: {self.images.shape}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")
    
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
    """Train the baseline regression model"""
    print("🚀 TRAINING BASELINE REGRESSION MODEL")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load datasets
    train_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "train", target_size=28)
    test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 5000  # Increased for better convergence
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
        
        # Extended Early Stopping Logic (More Patient)
        if epoch > 100:  # Start checking after 100 epochs (was 50)
            # Check for plateau: no improvement > 0.05% for 50+ epochs (was 30)
            if len(test_losses) >= 50:
                recent_best = min(test_losses[-50:])  # Look at last 50 epochs (was 30)
                current_best = min(test_losses)

                # Calculate relative improvement
                if current_best > 0:
                    improvement = (current_best - recent_best) / current_best

                    # Stop if improvement < 0.05% for last 50 epochs (was 0.1% for 30)
                    if improvement < 0.0005:  # More strict threshold (was 0.001)
                        print(f"🛑 Early stopping at epoch {epoch+1}")
                        print(f"   Reason: No significant improvement (< 0.05%) for 50 epochs")
                        print(f"   Current best: {current_best:.6f}, Recent best: {recent_best:.6f}")
                        print(f"   Improvement: {improvement*100:.4f}%")
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
    print(f"\n📊 EVALUATING BASELINE MODEL")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = SimpleRegressionModel(fmri_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    model.load_state_dict(torch.load('baseline_model_best.pth', map_location=device))
    model.eval()
    
    # Load test data
    test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)
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

def main():
    """Main function"""
    print("🎯 SIMPLE BASELINE MODEL FOR DIGIT69")
    print("=" * 50)
    
    # Train model
    model, train_losses, test_losses = train_baseline_model()
    
    # Evaluate model
    predictions, targets, correlations = evaluate_baseline_model()
    
    print(f"\n🎯 BASELINE MODEL SUMMARY:")
    print(f"   Final MSE: {mean_squared_error(targets.flatten(), predictions.flatten()):.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f}")
    print(f"   Training completed successfully!")
    
    print(f"\n📁 Generated files:")
    print(f"   - baseline_model_best.pth")
    print(f"   - baseline_model_final.pth")
    print(f"   - baseline_training_curves.png")
    print(f"   - baseline_model_results.png")

if __name__ == "__main__":
    main()
