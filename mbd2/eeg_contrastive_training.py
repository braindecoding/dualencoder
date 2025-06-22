#!/usr/bin/env python3
"""
EEG Contrastive Learning Training
Train EEG Transformer Encoder using contrastive learning with CLIP image embeddings
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

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for EEG-Image alignment
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, eeg_embeddings, image_embeddings):
        """
        Compute contrastive loss between EEG and image embeddings
        
        Args:
            eeg_embeddings: [batch, embed_dim] - EEG embeddings
            image_embeddings: [batch, embed_dim] - CLIP image embeddings
            
        Returns:
            loss: contrastive loss value
        """
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(eeg_embeddings, image_embeddings.T) / self.temperature
        
        # Labels for positive pairs (diagonal elements)
        batch_size = eeg_embeddings.size(0)
        labels = torch.arange(batch_size).to(eeg_embeddings.device)
        
        # Compute cross-entropy loss for both directions
        loss_eeg_to_image = F.cross_entropy(similarity_matrix, labels)
        loss_image_to_eeg = F.cross_entropy(similarity_matrix.T, labels)
        
        # Average both losses
        loss = (loss_eeg_to_image + loss_image_to_eeg) / 2
        
        return loss

class EEGImageDataset(Dataset):
    """
    Dataset for EEG-Image pairs
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
            except:
                print(f"Warning: Could not load image for digit {digit}")
                # Create dummy image if file not found
                self.stimulus_images[digit] = Image.new('RGB', (28, 28), color='white')
        
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
        
        # Use up to 300 samples per digit for training
        num_samples = min(min_samples, 300)
        
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

def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL Images
    """
    eeg_signals = torch.stack([item[0] for item in batch])
    images = [item[1] for item in batch]  # Keep as list of PIL Images
    labels = torch.tensor([item[2] for item in batch])

    return eeg_signals, images, labels

def train_eeg_encoder(eeg_encoder, dataloader, clip_model, device, num_epochs=50, lr=1e-4):
    """
    Train EEG encoder using contrastive learning with CLIP
    """
    print(f"🚀 Starting EEG Contrastive Training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Training samples: {len(dataloader.dataset)}")
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(eeg_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07).to(device)
    
    # Training history
    train_losses = []
    
    # CLIP preprocessing
    _, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    eeg_encoder.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
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
            
            # Generate EEG embeddings
            eeg_embeddings = eeg_encoder(eeg_signals)
            
            # Get CLIP embeddings for corresponding images
            with torch.no_grad():
                clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Contrastive loss
            loss = contrastive_loss_fn(eeg_embeddings, clip_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}'
            })
        
        # Update learning rate
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': eeg_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'train_losses': train_losses
            }
            torch.save(checkpoint, f'eeg_contrastive_encoder_epoch_{epoch+1}.pth')
            print(f"✅ Checkpoint saved: eeg_contrastive_encoder_epoch_{epoch+1}.pth")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': eeg_encoder.state_dict(),
        'train_losses': train_losses,
        'config': {
            'n_channels': 14,
            'seq_len': 256,
            'd_model': 128,
            'embedding_dim': 128,
            'encoder_type': eeg_encoder.encoder_type
        }
    }
    torch.save(final_checkpoint, 'eeg_contrastive_encoder_final.pth')
    print(f"✅ Final model saved: eeg_contrastive_encoder_final.pth")
    
    return train_losses

def evaluate_eeg_encoder(eeg_encoder, dataloader, clip_model, device):
    """
    Evaluate trained EEG encoder
    """
    print(f"📊 Evaluating EEG Encoder...")
    
    eeg_encoder.eval()
    _, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    all_eeg_embeddings = []
    all_clip_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for eeg_signals, images, labels in tqdm(dataloader, desc="Evaluating"):
            eeg_signals = eeg_signals.to(device)
            
            # Preprocess images for CLIP
            processed_images = []
            for img in images:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Generate embeddings
            eeg_embeddings = eeg_encoder(eeg_signals)
            clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Normalize embeddings
            eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
            clip_embeddings = F.normalize(clip_embeddings, dim=-1)
            
            all_eeg_embeddings.append(eeg_embeddings.cpu())
            all_clip_embeddings.append(clip_embeddings.cpu())
            all_labels.extend(labels.numpy())
    
    # Concatenate all embeddings
    all_eeg_embeddings = torch.cat(all_eeg_embeddings, dim=0)
    all_clip_embeddings = torch.cat(all_clip_embeddings, dim=0)
    all_labels = np.array(all_labels)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_eeg_embeddings, all_clip_embeddings.T)
    
    # Compute retrieval accuracy (top-1)
    _, top1_indices = torch.topk(similarity_matrix, k=1, dim=1)
    top1_accuracy = (top1_indices.squeeze() == torch.arange(len(all_labels))).float().mean()
    
    # Compute retrieval accuracy (top-5)
    _, top5_indices = torch.topk(similarity_matrix, k=5, dim=1)
    top5_accuracy = 0
    for i in range(len(all_labels)):
        if i in top5_indices[i]:
            top5_accuracy += 1
    top5_accuracy /= len(all_labels)
    
    print(f"✅ Evaluation Results:")
    print(f"   Top-1 Retrieval Accuracy: {top1_accuracy:.3f}")
    print(f"   Top-5 Retrieval Accuracy: {top5_accuracy:.3f}")
    
    return {
        'top1_accuracy': top1_accuracy.item(),
        'top5_accuracy': top5_accuracy,
        'eeg_embeddings': all_eeg_embeddings,
        'clip_embeddings': all_clip_embeddings,
        'labels': all_labels
    }

def main():
    """
    Main function for EEG contrastive training
    """
    print("🧠 EEG CONTRASTIVE LEARNING TRAINING")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load CLIP model (specify pre-trained model explicitly)
    print(f"\n📥 Loading CLIP model...")
    print(f"   Model: ViT-B/32 (Vision Transformer Base, 32x32 patches)")
    print(f"   Pre-trained: OpenAI CLIP")
    print(f"   Embedding dimension: 512")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()  # Keep CLIP frozen - NO TRAINING

    # Freeze all CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    print(f"✅ CLIP model loaded and frozen")
    print(f"   Total parameters: {sum(p.numel() for p in clip_model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in clip_model.parameters() if p.requires_grad):,}")
    
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
    
    # Train EEG encoder
    print(f"\n🚀 Starting Contrastive Training...")
    train_losses = train_eeg_encoder(
        eeg_encoder=eeg_encoder,
        dataloader=dataloader,
        clip_model=clip_model,
        device=device,
        num_epochs=50,
        lr=1e-4
    )
    
    # Evaluate trained encoder
    print(f"\n📊 Evaluating Trained Encoder...")
    eval_results = evaluate_eeg_encoder(eeg_encoder, dataloader, clip_model, device)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('EEG Contrastive Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.grid(True)
    plt.savefig('eeg_contrastive_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 EEG Contrastive Training completed successfully!")
    print(f"✅ Final Top-1 Accuracy: {eval_results['top1_accuracy']:.3f}")
    print(f"✅ Final Top-5 Accuracy: {eval_results['top5_accuracy']:.3f}")
    
    return {
        'eeg_encoder': eeg_encoder,
        'train_losses': train_losses,
        'eval_results': eval_results
    }

if __name__ == "__main__":
    results = main()
