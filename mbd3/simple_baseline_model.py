#!/usr/bin/env python3
"""
Simple Baseline Model: Direct EEG Embeddings → Image Regression
Using MBD3 EEG embeddings for brain-to-image reconstruction
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
    """Direct regression from EEG embeddings to images"""

    def __init__(self, eeg_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]):
        super().__init__()

        self.eeg_dim = eeg_dim
        self.image_size = image_size
        self.output_dim = image_size * image_size  # Grayscale output

        # Build encoder layers
        layers = []
        input_dim = eeg_dim

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
        print(f"   Input: EEG embeddings ({eeg_dim})")
        print(f"   Hidden layers: {hidden_dims}")
        print(f"   Output: {image_size}x{image_size} image ({self.output_dim})")

    def forward(self, eeg_embeddings):
        """Forward pass"""
        batch_size = eeg_embeddings.shape[0]

        # Pass through network
        output = self.network(eeg_embeddings)

        # Reshape to image format
        output = output.view(batch_size, 1, self.image_size, self.image_size)

        return output

class EEGBaselineDataset(Dataset):
    """Dataset for EEG baseline model using MBD3 embeddings"""

    def __init__(self, embeddings_file="eeg_embeddings_enhanced_20250622_123559.pkl",
                 data_splits_file="explicit_eeg_data_splits.pkl", split="train", target_size=28):
        self.split = split
        self.target_size = target_size

        # Load EEG embeddings
        with open(embeddings_file, 'rb') as f:
            emb_data = pickle.load(f)

        # Load original data splits for stimulus images
        with open(data_splits_file, 'rb') as f:
            split_data = pickle.load(f)

        # Get split indices
        split_indices = emb_data['split_indices'][split]

        # Extract data for this split
        self.eeg_embeddings = emb_data['embeddings'][split_indices]
        self.labels = emb_data['labels'][split_indices]

        # Get stimulus images for this split
        if split == 'train':
            self.original_images = split_data['training']['stimTrn']
        elif split == 'val':
            self.original_images = split_data['validation']['stimVal']
        else:  # test
            self.original_images = split_data['test']['stimTest']

        print(f"📊 Loaded {split} data:")
        print(f"   EEG embeddings: {self.eeg_embeddings.shape}")
        print(f"   Labels: {self.labels.shape}")
        print(f"   Original images: {len(self.original_images)} PIL images")

        # Process images
        self.images = self._process_images()

        print(f"   Processed images: {self.images.shape}")
        print(f"   Image range: [{self.images.min():.3f}, {self.images.max():.3f}]")
    
    def _process_images(self):
        """Process PIL images to target format"""
        import numpy as np
        from PIL import Image

        processed_images = []

        for pil_img in self.original_images:
            # Convert PIL to numpy array
            img_array = np.array(pil_img)

            # Convert to grayscale if RGB
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)  # RGB to grayscale

            # Resize to target size
            if img_array.shape[0] != self.target_size:
                pil_resized = Image.fromarray(img_array.astype(np.uint8)).resize(
                    (self.target_size, self.target_size), Image.LANCZOS)
                img_array = np.array(pil_resized)

            # Normalize to [-1, 1]
            img_array = img_array.astype(np.float32) / 255.0  # [0, 1]
            img_array = (img_array - 0.5) * 2  # [-1, 1]

            processed_images.append(img_array)

        # Stack and add channel dimension
        images = np.array(processed_images)  # (N, H, W)
        images = images[:, None, :, :]  # (N, 1, H, W)

        return images
    
    def __len__(self):
        return len(self.eeg_embeddings)
    
    def __getitem__(self, idx):
        eeg_emb = torch.FloatTensor(self.eeg_embeddings[idx])
        image = torch.FloatTensor(self.images[idx])
        label = self.labels[idx]

        return eeg_emb, image, label

def train_baseline_model():
    """Train the baseline regression model"""
    print("🚀 TRAINING BASELINE REGRESSION MODEL")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load datasets
    train_dataset = EEGBaselineDataset(split="train", target_size=28)
    test_dataset = EEGBaselineDataset(split="test", target_size=28)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = SimpleRegressionModel(eeg_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 5
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
        
        for eeg_emb, images, labels in train_pbar:
            eeg_emb = eeg_emb.to(device)
            images = images.to(device)

            # Forward pass
            predicted_images = model(eeg_emb)
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
            
            for eeg_emb, images, labels in test_pbar:
                eeg_emb = eeg_emb.to(device)
                images = images.to(device)

                predicted_images = model(eeg_emb)
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
        
        # Early stopping check
        if epoch > 20 and len(test_losses) > 10:
            recent_losses = test_losses[-10:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                print(f"🛑 Early stopping at epoch {epoch+1}")
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
    model = SimpleRegressionModel(eeg_dim=512, image_size=28, hidden_dims=[1024, 2048, 1024]).to(device)
    model.load_state_dict(torch.load('baseline_model_best.pth', map_location=device))
    model.eval()

    # Load test data
    test_dataset = EEGBaselineDataset(split="test", target_size=28)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Generate predictions
    predictions = []
    targets = []
    
    with torch.no_grad():
        for eeg_emb, images, labels in test_loader:
            eeg_emb = eeg_emb.to(device)

            predicted_images = model(eeg_emb)
            
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
    print("🎯 SIMPLE BASELINE MODEL FOR EEG EMBEDDINGS")
    print("=" * 60)
    print("Using MBD3 EEG embeddings for brain-to-image reconstruction")
    print("=" * 60)

    # Train model
    _, train_losses, test_losses = train_baseline_model()

    # Evaluate model
    predictions, targets, correlations = evaluate_baseline_model()

    print(f"\n🎯 EEG BASELINE MODEL SUMMARY:")
    print(f"   Final MSE: {mean_squared_error(targets.flatten(), predictions.flatten()):.4f}")
    print(f"   Mean Correlation: {correlations.mean():.4f}")
    print(f"   Training completed successfully!")

    print(f"\n📁 Generated files:")
    print(f"   - baseline_model_best.pth")
    print(f"   - baseline_model_final.pth")
    print(f"   - baseline_training_curves.png")
    print(f"   - baseline_model_results.png")

    print(f"\n🧠 EEG-to-Image Reconstruction Results:")
    print(f"   Input: 512-dim EEG embeddings (from enhanced transformer)")
    print(f"   Output: 28x28 grayscale digit images")
    print(f"   Performance: {correlations.mean():.4f} mean correlation")
    print(f"   🚀 Ready for advanced applications!")

if __name__ == "__main__":
    main()
