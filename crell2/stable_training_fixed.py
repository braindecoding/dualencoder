#!/usr/bin/env python3
"""
Stable Training with Fixed Parameters
Fix NaN loss issue and improve convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel

class StableContrastiveLoss(nn.Module):
    """Stable contrastive loss with proper temperature and gradient clipping"""
    
    def __init__(self, temperature=0.1):  # Higher temperature for stability
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size).to(embeddings.device)
        
        # Stable InfoNCE loss
        exp_sim = torch.exp(similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0])  # Numerical stability
        
        # Positive pairs
        pos_mask = mask
        pos_exp = exp_sim * pos_mask
        pos_sum = torch.sum(pos_exp, dim=1, keepdim=True)
        
        # All pairs (excluding diagonal)
        all_sum = torch.sum(exp_sim, dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
        
        # Avoid division by zero
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        all_sum = torch.clamp(all_sum, min=1e-8)
        
        # InfoNCE loss
        loss = -torch.log(pos_sum / all_sum)
        
        # Only compute for samples with positive pairs
        valid_mask = (pos_sum.squeeze() > 1e-8)
        if valid_mask.sum() > 0:
            loss = torch.mean(loss[valid_mask])
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(embeddings.device)
        
        return loss

class CrellEEGDataset(Dataset):
    """Dataset for Crell EEG-to-Letter training"""
    
    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]

def load_crell_data():
    """Load Crell dataset"""
    print("📂 Loading Crell dataset...")
    
    with open('crell_processed_data_correct.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return data

def stable_training():
    """Stable training with fixed parameters"""
    print("🚀 STABLE TRAINING - FIXED PARAMETERS")
    print("=" * 70)
    print("🎯 Fixes: Stable temperature, gradient clipping, proper loss")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load data
    data = load_crell_data()
    
    # Create datasets
    train_dataset = CrellEEGDataset(
        data['training']['eeg'],
        data['training']['labels']
    )
    
    val_dataset = CrellEEGDataset(
        data['validation']['eeg'],
        data['validation']['labels']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = EnhancedEEGToEmbeddingModel(
        n_channels=64,
        seq_len=500,
        d_model=256,
        embedding_dim=512,
        nhead=8,
        num_layers=8,
        patch_size=25,
        dropout=0.1
    ).to(device)
    
    # Add stable classifier
    model.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)  # 10 letters
    ).to(device)
    
    # Load checkpoint if available
    try:
        checkpoint = torch.load('advanced_eeg_checkpoint_50.pth', map_location=device)
        model_state = checkpoint['model_state_dict']
        current_model_state = model.state_dict()
        
        # Load only matching keys
        filtered_state = {k: v for k, v in model_state.items() if k in current_model_state and current_model_state[k].shape == v.shape}
        current_model_state.update(filtered_state)
        model.load_state_dict(current_model_state)
        
        start_epoch = 51  # Continue from epoch 51
        print(f"✅ Loaded checkpoint, resuming from epoch {start_epoch}")
    except:
        start_epoch = 0
        print("⚠️ No checkpoint found, starting from scratch")
    
    # Stable loss functions
    contrastive_loss = StableContrastiveLoss(temperature=0.1)  # Stable temperature
    classification_loss = nn.CrossEntropyLoss()
    
    # Conservative optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Training parameters
    num_epochs = 200  # Reduced for stability
    best_val_accuracy = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n🎯 Stable Training Configuration:")
    print(f"   Start epoch: {start_epoch}")
    print(f"   Total epochs: {num_epochs}")
    print(f"   Batch size: 16")
    print(f"   Learning rate: 1e-4 (conservative)")
    print(f"   Temperature: 0.1 (stable)")
    print(f"   Gradient clipping: 1.0")
    print(f"   Scheduler: StepLR (decay every 50 epochs)")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_cont_loss = 0
        total_class_loss = 0
        
        for eeg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(eeg)
            
            # Contrastive loss
            cont_loss = contrastive_loss(embeddings, labels)
            
            # Classification loss
            logits = model.classifier(embeddings)
            class_loss = classification_loss(logits, labels)
            
            # Combined loss with conservative weighting
            loss = 0.7 * cont_loss + 0.3 * class_loss
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_cont_loss += cont_loss.item()
            total_class_loss += class_loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg = eeg.to(device)
                labels = labels.to(device)
                
                # Forward pass
                embeddings = model(eeg)
                
                # Losses
                cont_loss = contrastive_loss(embeddings, labels)
                logits = model.classifier(embeddings)
                class_loss = classification_loss(logits, labels)
                
                loss = 0.7 * cont_loss + 0.3 * class_loss
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
                # Accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        # Calculate averages
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"📈 Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train={avg_train_loss:.4f} (C:{total_cont_loss/len(train_loader):.4f}, Cl:{total_class_loss/len(train_loader):.4f}), "
                  f"Val={avg_val_loss:.4f}, Acc={val_accuracy:.3f} ({val_accuracy*100:.1f}%), "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'train_loss': avg_train_loss
            }, 'stable_eeg_model_best.pth')
            print(f"   ✅ New best model saved! Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'train_loss': avg_train_loss
            }, f'stable_eeg_checkpoint_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    training_time = time.time() - start_time
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'train_loss': avg_train_loss
    }, 'stable_eeg_model_final.pth')
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    epochs_range = range(start_epoch, num_epochs)
    plt.plot(epochs_range, train_losses, label='Training Loss', alpha=0.8)
    plt.plot(epochs_range, val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stable Training: Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Stable Training: Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Best accuracy progress
    plt.subplot(1, 3, 3)
    best_acc_so_far = np.maximum.accumulate(val_accuracies)
    plt.plot(epochs_range, best_acc_so_far, label='Best Accuracy So Far', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Best Accuracy')
    plt.title('Best Accuracy Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stable_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Stable training completed!")
    print(f"   Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"   Best validation accuracy: {best_val_accuracy:.3f} ({best_val_accuracy*100:.1f}%)")
    print(f"   Final validation accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
    print(f"   Accuracy improvement: {(val_accuracy - val_accuracies[0] if val_accuracies else 0):.3f}")
    
    print(f"\n📁 Generated files:")
    print(f"   - stable_eeg_model_best.pth")
    print(f"   - stable_eeg_model_final.pth")
    print(f"   - stable_training_results.png")
    
    print(f"\n🎯 Issues Fixed:")
    print(f"   ✅ Stable temperature (0.1) prevents NaN loss")
    print(f"   ✅ Gradient clipping (1.0) prevents explosion")
    print(f"   ✅ Conservative learning rate (1e-4)")
    print(f"   ✅ Proper numerical stability in loss computation")
    print(f"   ✅ NaN detection and handling")
    
    return model, train_losses, val_losses, val_accuracies

if __name__ == "__main__":
    model, train_losses, val_losses, val_accuracies = stable_training()
