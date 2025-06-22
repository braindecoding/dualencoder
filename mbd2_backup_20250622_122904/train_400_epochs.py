#!/usr/bin/env python3
"""
Extended EEG Contrastive Learning Training - 400 Epochs
Optimized training script with extended epochs and improved scheduling
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
import time
warnings.filterwarnings("ignore")

from eeg_transformer_encoder import EEGToEmbeddingModel
from explicit_eeg_contrastive_training import (
    ContrastiveLoss, load_explicit_data, create_batches, evaluate_explicit
)

def train_extended_epochs():
    """
    Extended training with 400 epochs and optimizations
    """
    print("🧠 EXTENDED EEG CONTRASTIVE LEARNING - 400 EPOCHS")
    print("=" * 70)
    print("🚀 Optimizations:")
    print("   • 400 epochs with warm restarts")
    print("   • Increased patience (50 epochs)")
    print("   • Cosine annealing with warm restarts")
    print("   • Enhanced monitoring and logging")
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
    
    # Training configuration
    num_epochs = 400
    batch_size = 32
    lr = 1e-4
    patience = 50
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Patience: {patience}")
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
    
    # Enhanced optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': eeg_encoder.encoder.parameters(), 'lr': lr},
        {'params': eeg_encoder.embedding_adapter.parameters(), 'lr': lr * 2}
    ], weight_decay=1e-4)
    
    # Cosine annealing with warm restarts for longer training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # Loss function
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07).to(device)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_val_accuracy = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting Extended Training...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
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
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.3f}',
                'LR': f'{current_lr:.2e}'
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
        current_lr = scheduler.get_last_lr()[0]
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        learning_rates.append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s, Total: {total_time/3600:.1f}h):")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.3f}")
        print(f"   LR: {current_lr:.2e}")
        
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
                'best_val_accuracy': best_val_accuracy,
                'total_epochs': num_epochs,
                'training_time_hours': total_time / 3600
            }, 'mbd2/eeg_contrastive_400epochs_best.pth')
            
            print(f"   ✅ New best validation accuracy: {best_val_accuracy:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_accuracy:.3f}")
            print(f"   Total training time: {total_time/3600:.1f} hours")
            break
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': eeg_encoder.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'learning_rates': learning_rates,
                'training_time_hours': total_time / 3600
            }, f'mbd2/eeg_contrastive_400epochs_checkpoint_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    # Final model save
    total_training_time = time.time() - start_time
    torch.save({
        'model_state_dict': eeg_encoder.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy,
        'total_training_time_hours': total_training_time / 3600,
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
    }, 'mbd2/eeg_contrastive_400epochs_final.pth')
    
    print(f"\n🏁 Training completed!")
    print(f"   Total time: {total_training_time/3600:.1f} hours")
    print(f"   Best validation accuracy: {best_val_accuracy:.3f}")
    
    return eeg_encoder, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy,
        'training_time_hours': total_training_time / 3600
    }

def evaluate_final_model():
    """
    Load best model and evaluate on test set
    """
    print(f"\n🔍 FINAL MODEL EVALUATION")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # Load test data
    _, _, _, _, _, _, eegTest, stimTest, labelsTest = load_explicit_data()

    # Load best model
    print(f"📥 Loading best model...")
    checkpoint = torch.load('mbd2/eeg_contrastive_400epochs_best.pth')

    eeg_encoder = EEGToEmbeddingModel(
        n_channels=14, seq_len=256, d_model=128, embedding_dim=512,
        encoder_type='single', nhead=8, num_layers=6, patch_size=16, dropout=0.1
    ).to(device)

    eeg_encoder.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Best validation accuracy: {checkpoint['best_val_accuracy']:.3f}")
    print(f"   Training time: {checkpoint.get('training_time_hours', 0):.1f} hours")

    # Final test evaluation
    print(f"\n🎯 FINAL TEST EVALUATION:")
    test_loss, test_accuracy = evaluate_explicit(
        eeg_encoder, eegTest, stimTest, labelsTest,
        clip_model, clip_preprocess, device, "test", batch_size=32
    )

    print(f"\n🏆 FINAL RESULTS SUMMARY:")
    print(f"   Best Validation Accuracy: {checkpoint['best_val_accuracy']:.3f} ({checkpoint['best_val_accuracy']*100:.1f}%)")
    print(f"   Final Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"   Test Loss: {test_loss:.4f}")

    return test_accuracy, test_loss

if __name__ == "__main__":
    # Run extended training
    model, results = train_extended_epochs()

    # Evaluate final model
    test_acc, test_loss = evaluate_final_model()

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(results['train_losses'], label='Train Loss', alpha=0.7)
    plt.plot(results['val_losses'], label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (400 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(results['train_accuracies'], label='Train Accuracy', alpha=0.7)
    plt.plot(results['val_accuracies'], label='Validation Accuracy', alpha=0.7)
    plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Test Accuracy ({test_acc:.3f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (400 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(results['learning_rates'], label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Cosine Annealing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('mbd2/eeg_training_400epochs_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📊 Training curves saved as 'mbd2/eeg_training_400epochs_curves.png'")
    print(f"🎉 Extended training with 400 epochs completed!")
