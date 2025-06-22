#!/usr/bin/env python3
"""
EEG Contrastive Learning Training with Explicit Data Variables
Using eegTrn, stimTrn, eegVal, stimVal, eegTest, stimTest format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
import warnings
warnings.filterwarnings("ignore")

from eeg_transformer_encoder import EEGToEmbeddingModel

class ContrastiveLoss(nn.Module):
    """Contrastive loss for EEG-image alignment"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, eeg_embeddings, image_embeddings):
        batch_size = eeg_embeddings.size(0)
        
        # Normalize embeddings
        eeg_norm = F.normalize(eeg_embeddings, dim=-1, p=2)
        img_norm = F.normalize(image_embeddings, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity = torch.matmul(eeg_norm, img_norm.T) / self.temperature
        
        # Labels for positive pairs
        labels = torch.arange(batch_size, device=eeg_embeddings.device)
        
        # Contrastive loss (both directions)
        loss_eeg2img = F.cross_entropy(similarity, labels)
        loss_img2eeg = F.cross_entropy(similarity.T, labels)
        
        total_loss = (loss_eeg2img + loss_img2eeg) / 2
        
        # Compute accuracy
        with torch.no_grad():
            pred_eeg2img = torch.argmax(similarity, dim=1)
            pred_img2eeg = torch.argmax(similarity.T, dim=1)
            acc_eeg2img = (pred_eeg2img == labels).float().mean()
            acc_img2eeg = (pred_img2eeg == labels).float().mean()
            avg_accuracy = (acc_eeg2img + acc_img2eeg) / 2
        
        return total_loss, avg_accuracy.item()

def load_explicit_data():
    """
    Load explicit data splits
    Returns: eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest
    """
    print("📂 Loading explicit EEG data splits...")
    
    with open('mbd2/explicit_eeg_data_splits.pkl', 'rb') as f:
        explicit_data = pickle.load(f)
    
    # Extract training data
    eegTrn = explicit_data['training']['eegTrn']
    stimTrn = explicit_data['training']['stimTrn']
    labelsTrn = explicit_data['training']['labelsTrn']
    
    # Extract validation data
    eegVal = explicit_data['validation']['eegVal']
    stimVal = explicit_data['validation']['stimVal']
    labelsVal = explicit_data['validation']['labelsVal']
    
    # Extract test data
    eegTest = explicit_data['test']['eegTest']
    stimTest = explicit_data['test']['stimTest']
    labelsTest = explicit_data['test']['labelsTest']
    
    print(f"✅ Explicit data loaded:")
    print(f"   Training:   eegTrn {eegTrn.shape}, stimTrn {len(stimTrn)}, labelsTrn {labelsTrn.shape}")
    print(f"   Validation: eegVal {eegVal.shape}, stimVal {len(stimVal)}, labelsVal {labelsVal.shape}")
    print(f"   Test:       eegTest {eegTest.shape}, stimTest {len(stimTest)}, labelsTest {labelsTest.shape}")
    
    return eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest

def create_batches(eeg_data, stim_data, labels, batch_size, shuffle=True):
    """
    Create batches from explicit data arrays
    """
    num_samples = len(eeg_data)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_eeg = eeg_data[batch_indices]
        batch_stim = [stim_data[i] for i in batch_indices]
        batch_labels = labels[batch_indices]
        
        yield batch_eeg, batch_stim, batch_labels

def evaluate_explicit(model, eegData, stimData, labelsData, clip_model, clip_preprocess, device, 
                     split_name, batch_size=32):
    """
    Evaluate model using explicit data arrays
    """
    model.eval()
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07).to(device)
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    print(f"   Evaluating {split_name}...")
    
    with torch.no_grad():
        for batch_eeg, batch_stim, batch_labels in create_batches(eegData, stimData, labelsData, 
                                                                 batch_size, shuffle=False):
            # Move EEG to device
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Preprocess images for CLIP
            processed_images = []
            for img in batch_stim:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Forward pass
            eeg_embeddings = model(batch_eeg)
            clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Compute loss and accuracy
            loss, accuracy = contrastive_loss_fn(eeg_embeddings, clip_embeddings)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    print(f"   {split_name} Results: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
    
    return avg_loss, avg_accuracy

def train_with_explicit_data(eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, 
                           clip_model, clip_preprocess, device, num_epochs=200, 
                           batch_size=32, lr=1e-4):
    """
    Train EEG encoder using explicit data variables
    """
    print(f"🚀 Starting EEG Contrastive Training with Explicit Data...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Training samples: {len(eegTrn)}")
    print(f"   Validation samples: {len(eegVal)}")
    
    # Initialize EEG encoder
    eeg_encoder = EEGToEmbeddingModel(
        n_channels=14,
        seq_len=256,
        d_model=128,
        embedding_dim=512,
        encoder_type='single',
        nhead=8,
        num_layers=6,
        patch_size=16,
        dropout=0.1
    ).to(device)
    
    print(f"✅ EEG Encoder initialized with {sum(p.numel() for p in eeg_encoder.parameters()):,} parameters")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': eeg_encoder.encoder.parameters(), 'lr': lr},
        {'params': eeg_encoder.embedding_adapter.parameters(), 'lr': lr * 2}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07).to(device)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0
    patience_counter = 0
    patience = 25
    
    for epoch in range(num_epochs):
        # TRAINING PHASE
        eeg_encoder.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        num_train_batches = 0
        
        # Create training batches
        train_progress = tqdm(
            create_batches(eegTrn, stimTrn, labelsTrn, batch_size, shuffle=True),
            desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]",
            total=len(eegTrn) // batch_size
        )
        
        for batch_eeg, batch_stim, batch_labels in train_progress:
            # Move EEG to device
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Preprocess images for CLIP
            processed_images = []
            for img in batch_stim:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Forward pass
            eeg_embeddings = eeg_encoder(batch_eeg)
            
            with torch.no_grad():
                clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Compute loss
            loss, accuracy = contrastive_loss_fn(eeg_embeddings, clip_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy
            num_train_batches += 1
            
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.3f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # VALIDATION PHASE
        val_loss, val_accuracy = evaluate_explicit(
            eeg_encoder, eegVal, stimVal, labelsVal, 
            clip_model, clip_preprocess, device, "validation", batch_size
        )
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_accuracy = epoch_train_accuracy / num_train_batches
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.3f}")
        
        # Early stopping and best model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': eeg_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'train_accuracy': avg_train_accuracy,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_accuracy
            }, 'explicit_eeg_contrastive_encoder_best.pth')
            
            print(f"   ✅ New best validation accuracy: {best_val_accuracy:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_accuracy:.3f}")
            break
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': eeg_encoder.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, f'explicit_eeg_contrastive_encoder_epoch_{epoch+1}.pth')
    
    # Final model save
    torch.save({
        'model_state_dict': eeg_encoder.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'config': {
            'n_channels': 14,
            'seq_len': 256,
            'd_model': 128,
            'embedding_dim': 512,
            'encoder_type': 'single',
            'num_epochs': len(train_losses),
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1],
            'best_val_accuracy': best_val_accuracy
        }
    }, 'explicit_eeg_contrastive_encoder_final.pth')
    
    return eeg_encoder, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }

def main():
    """
    Main function using explicit data format
    """
    print("🧠 EEG CONTRASTIVE LEARNING WITH EXPLICIT DATA")
    print("=" * 70)
    print("Format: eegTrn, stimTrn, eegVal, stimVal, eegTest, stimTest")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load CLIP model
    print(f"\n📥 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print(f"✅ CLIP model loaded and frozen")
    
    # Load explicit data
    eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest = load_explicit_data()
    
    # Train model
    eeg_encoder, training_results = train_with_explicit_data(
        eegTrn, stimTrn, labelsTrn, 
        eegVal, stimVal, labelsVal,
        clip_model, clip_preprocess, device,
        num_epochs=200,
        batch_size=32,
        lr=1e-4
    )
    
    # Load best model for final test evaluation
    print(f"\n🔍 Loading best model for final test evaluation...")
    checkpoint = torch.load('explicit_eeg_contrastive_encoder_best.pth')
    eeg_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation using explicit test data
    print(f"\n🎯 FINAL TEST EVALUATION (eegTest, stimTest, labelsTest):")
    test_loss, test_accuracy = evaluate_explicit(
        eeg_encoder, eegTest, stimTest, labelsTest,
        clip_model, clip_preprocess, device, "test", batch_size=32
    )
    
    print(f"\n🏆 FINAL RESULTS SUMMARY:")
    print(f"   Best Validation Accuracy: {training_results['best_val_accuracy']:.3f} ({training_results['best_val_accuracy']*100:.1f}%)")
    print(f"   Final Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_results['train_losses'], label='Train Loss')
    plt.plot(training_results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_results['train_accuracies'], label='Train Accuracy')
    plt.plot(training_results['val_accuracies'], label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label=f'Test Accuracy ({test_accuracy:.3f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('explicit_eeg_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'eeg_encoder': eeg_encoder,
        'training_results': training_results,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

if __name__ == "__main__":
    results = main()
