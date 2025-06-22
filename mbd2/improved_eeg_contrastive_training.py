#!/usr/bin/env python3
"""
Improved EEG Contrastive Learning Training
Train EEG Transformer Encoder using contrastive learning with CLIP image embeddings
With explicit CLIP model specification and better training strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
import warnings
warnings.filterwarnings("ignore")

from eeg_transformer_encoder import EEGToEmbeddingModel

class EarlyStopping:
    """
    Early stopping to prevent overfitting during long training
    """
    def __init__(self, patience=30, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class ImprovedContrastiveLoss(nn.Module):
    """
    Improved contrastive loss with temperature scaling and normalization
    """
    def __init__(self, temperature=0.07, use_cosine_similarity=True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        
    def forward(self, eeg_embeddings, image_embeddings):
        """
        Compute contrastive loss between EEG and image embeddings
        
        Args:
            eeg_embeddings: [batch, embed_dim] - EEG embeddings
            image_embeddings: [batch, embed_dim] - CLIP image embeddings
            
        Returns:
            loss: contrastive loss value
        """
        batch_size = eeg_embeddings.size(0)
        
        # Normalize embeddings to unit sphere
        eeg_embeddings = F.normalize(eeg_embeddings, dim=-1, p=2)
        image_embeddings = F.normalize(image_embeddings, dim=-1, p=2)
        
        if self.use_cosine_similarity:
            # Cosine similarity matrix
            similarity_matrix = torch.matmul(eeg_embeddings, image_embeddings.T) / self.temperature
        else:
            # Dot product similarity
            similarity_matrix = torch.matmul(eeg_embeddings, image_embeddings.T) / self.temperature
        
        # Labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=eeg_embeddings.device)
        
        # Compute cross-entropy loss for both directions
        loss_eeg_to_image = F.cross_entropy(similarity_matrix, labels)
        loss_image_to_eeg = F.cross_entropy(similarity_matrix.T, labels)
        
        # Symmetric loss
        total_loss = (loss_eeg_to_image + loss_image_to_eeg) / 2
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            pred_eeg_to_img = torch.argmax(similarity_matrix, dim=1)
            pred_img_to_eeg = torch.argmax(similarity_matrix.T, dim=1)
            acc_eeg_to_img = (pred_eeg_to_img == labels).float().mean()
            acc_img_to_eeg = (pred_img_to_eeg == labels).float().mean()
            avg_accuracy = (acc_eeg_to_img + acc_img_to_eeg) / 2
        
        return total_loss, avg_accuracy.item()

class EEGImageDataset(Dataset):
    """
    Dataset for EEG-Image pairs with improved image handling
    """
    def __init__(self, eeg_signals, labels, stimulus_images_path="dataset/datasets/MindbigdataStimuli"):
        self.eeg_signals = eeg_signals
        self.labels = labels
        self.stimulus_images_path = stimulus_images_path
        
        # Load stimulus images
        self.stimulus_images = {}
        for digit in range(10):
            img_path = f"{stimulus_images_path}/{digit}.jpg"
            try:
                img = Image.open(img_path).convert('RGB')
                self.stimulus_images[digit] = img
                print(f"✅ Loaded stimulus image for digit {digit}: {img.size}")
            except Exception as e:
                print(f"⚠️  Could not load image for digit {digit}: {e}")
                # Create dummy white image if file not found
                self.stimulus_images[digit] = Image.new('RGB', (224, 224), color='white')
        
        print(f"📊 EEG-Image Dataset created:")
        print(f"   EEG signals: {len(eeg_signals)}")
        print(f"   Unique labels: {len(set(labels))}")
        print(f"   Stimulus images loaded: {len(self.stimulus_images)}")
    
    def __len__(self):
        return len(self.eeg_signals)
    
    def __getitem__(self, idx):
        eeg_signal = torch.FloatTensor(self.eeg_signals[idx])
        label = self.labels[idx]
        image = self.stimulus_images[label]
        
        return eeg_signal, image, label

def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL Images
    """
    eeg_signals = torch.stack([item[0] for item in batch])
    images = [item[1] for item in batch]  # Keep as list of PIL Images
    labels = torch.tensor([item[2] for item in batch])
    
    return eeg_signals, images, labels

def load_preprocessed_data_for_training():
    """
    Load correctly preprocessed EEG data for contrastive training
    """
    print("📂 Loading correctly preprocessed EEG data for contrastive training...")
    
    with open('mbd2/correctly_preprocessed_eeg_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    processed_data = dataset['correctly_processed_eeg_data']
    
    # Create multi-electrode training dataset
    electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    eeg_signals = []
    labels = []
    
    for digit in range(10):  # Digits 0-9
        print(f"   Processing digit {digit}...")
        
        # Find minimum samples across all electrodes for this digit
        min_samples = min(len(processed_data[electrode][digit]) 
                         for electrode in electrodes 
                         if digit in processed_data[electrode])
        
        # Use up to 400 samples per digit for training (increased from 300)
        num_samples = min(min_samples, 400)
        
        for sample_idx in range(num_samples):
            # Stack signals from all 14 electrodes
            multi_electrode_signal = []
            for electrode in electrodes:
                signal = processed_data[electrode][digit][sample_idx]
                multi_electrode_signal.append(signal)
            
            eeg_signals.append(np.stack(multi_electrode_signal))  # Shape: (14, 256)
            labels.append(digit)
    
    eeg_signals = np.array(eeg_signals)
    labels = np.array(labels)
    
    print(f"✅ Training dataset created:")
    print(f"   EEG signals shape: {eeg_signals.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Samples per digit: {len(labels) // 10}")
    
    return eeg_signals, labels

def train_eeg_encoder_improved(eeg_encoder, dataloader, clip_model, clip_preprocess, device,
                              num_epochs=300, lr=1e-4, warmup_epochs=30, use_early_stopping=True):
    """
    Improved training function with better monitoring and optimization
    """
    print(f"🚀 Starting Improved EEG Contrastive Training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Training samples: {len(dataloader.dataset)}")
    
    # Initialize optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': eeg_encoder.encoder.parameters(), 'lr': lr},
        {'params': eeg_encoder.embedding_adapter.parameters(), 'lr': lr * 2}  # Higher LR for adapter
    ], weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Improved contrastive loss
    contrastive_loss_fn = ImprovedContrastiveLoss(temperature=0.07).to(device)

    # Early stopping
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)
        print(f"   Early stopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    # Training history
    train_losses = []
    train_accuracies = []
    
    eeg_encoder.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (eeg_signals, images, labels) in enumerate(progress_bar):
            # Move to device
            eeg_signals = eeg_signals.to(device)
            
            # Preprocess images for CLIP
            processed_images = []
            for img in images:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Generate EEG embeddings (TRAINABLE)
            eeg_embeddings = eeg_encoder(eeg_signals)
            
            # Get CLIP embeddings for corresponding images (FROZEN)
            with torch.no_grad():
                clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Contrastive loss with accuracy
            loss, accuracy = contrastive_loss_fn(eeg_embeddings, clip_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.3f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}',
                'Avg Acc': f'{epoch_accuracy/num_batches:.3f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Update learning rate
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}, "
              f"Accuracy: {avg_epoch_accuracy:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(avg_epoch_loss, eeg_encoder):
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best loss: {early_stopping.best_loss:.4f}")
                print(f"   No improvement for {early_stopping.patience} epochs")
                early_stopping.restore_weights(eeg_encoder)
                print(f"   Restored best weights")
                break

        # Save checkpoint every 50 epochs (less frequent for 300 epochs)
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': eeg_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'accuracy': avg_epoch_accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies
            }
            torch.save(checkpoint, f'improved_eeg_contrastive_encoder_epoch_{epoch+1}.pth')
            print(f"✅ Checkpoint saved: improved_eeg_contrastive_encoder_epoch_{epoch+1}.pth")

    # Final training summary
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs completed: {len(train_losses)}")
    print(f"   Final loss: {train_losses[-1]:.4f}")
    print(f"   Final accuracy: {train_accuracies[-1]:.3f}")
    if early_stopping is not None:
        print(f"   Best loss achieved: {early_stopping.best_loss:.4f}")
        print(f"   Early stopping used: {'Yes' if early_stopping.counter >= early_stopping.patience else 'No'}")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': eeg_encoder.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'config': {
            'n_channels': 14,
            'seq_len': 256,
            'd_model': 128,
            'embedding_dim': 512,
            'encoder_type': eeg_encoder.encoder_type,
            'num_epochs': num_epochs,
            'final_loss': avg_epoch_loss,
            'final_accuracy': avg_epoch_accuracy
        }
    }
    torch.save(final_checkpoint, 'improved_eeg_contrastive_encoder_final.pth')
    print(f"✅ Final model saved: improved_eeg_contrastive_encoder_final.pth")
    
    return train_losses, train_accuracies

def main():
    """
    Main function for improved EEG contrastive training
    """
    print("🧠 IMPROVED EEG CONTRASTIVE LEARNING TRAINING")
    print("=" * 70)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load CLIP model with explicit specification
    print(f"\n📥 Loading Pre-trained CLIP Model...")
    print(f"   🎯 Model: ViT-B/32 (Vision Transformer Base)")
    print(f"   🎯 Architecture: 12 transformer layers, 768 hidden dim")
    print(f"   🎯 Image patches: 32x32 pixels")
    print(f"   🎯 Pre-training: 400M image-text pairs")
    print(f"   🎯 Embedding dimension: 512")
    print(f"   🎯 Publisher: OpenAI")
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()  # Keep CLIP frozen - NO TRAINING
    
    # Explicitly freeze all CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in clip_model.parameters())
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    
    print(f"✅ CLIP model loaded and frozen")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Status: 100% FROZEN (no training)")
    
    # Load preprocessed EEG data
    eeg_signals, labels = load_preprocessed_data_for_training()
    
    # Create dataset and dataloader
    dataset = EEGImageDataset(eeg_signals, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    # Initialize EEG encoder
    print(f"\n🔧 Initializing EEG Encoder...")
    eeg_encoder = EEGToEmbeddingModel(
        n_channels=14,
        seq_len=256,
        d_model=128,
        embedding_dim=512,  # Match CLIP embedding dimension
        encoder_type='single',
        nhead=8,
        num_layers=6,
        patch_size=16,
        dropout=0.1
    ).to(device)
    
    eeg_total_params = sum(p.numel() for p in eeg_encoder.parameters())
    eeg_trainable_params = sum(p.numel() for p in eeg_encoder.parameters() if p.requires_grad)
    
    print(f"✅ EEG Encoder initialized")
    print(f"   Total parameters: {eeg_total_params:,}")
    print(f"   Trainable parameters: {eeg_trainable_params:,}")
    print(f"   Status: 100% TRAINABLE")
    
    # Train EEG encoder with improved strategy
    print(f"\n🚀 Starting Extended Contrastive Training...")
    train_losses, train_accuracies = train_eeg_encoder_improved(
        eeg_encoder=eeg_encoder,
        dataloader=dataloader,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
        num_epochs=300,  # Extended to 300 epochs
        lr=1e-4,
        warmup_epochs=30,  # Longer warmup
        use_early_stopping=True  # Enable early stopping
    )
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_eeg_contrastive_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Improved EEG Contrastive Training completed successfully!")
    print(f"✅ Final Loss: {train_losses[-1]:.4f}")
    print(f"✅ Final Accuracy: {train_accuracies[-1]:.3f}")
    
    return {
        'eeg_encoder': eeg_encoder,
        'clip_model': clip_model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }

if __name__ == "__main__":
    results = main()
