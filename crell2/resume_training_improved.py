#!/usr/bin/env python3
"""
Resume Training with Improved Parameters
Fix issues: Early stopping, High inter-letter similarity, Low accuracy
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
from sklearn.metrics import accuracy_score
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel

class ImprovedContrastiveLoss(nn.Module):
    """Improved contrastive loss with lower temperature for better separation"""
    
    def __init__(self, temperature=0.05, margin=2.0):  # Lower temperature, higher margin
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
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
        
        # InfoNCE loss for better separation
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs
        pos_mask = mask
        pos_exp = exp_sim * pos_mask
        pos_sum = torch.sum(pos_exp, dim=1, keepdim=True)
        
        # All pairs (positive + negative)
        all_sum = torch.sum(exp_sim, dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
        
        # InfoNCE loss
        loss = -torch.log(pos_sum / (all_sum + 1e-8))
        loss = torch.mean(loss[pos_sum.squeeze() > 0])  # Only compute for samples with positive pairs
        
        return loss

class CrellEEGDataset(Dataset):
    """Dataset for Crell EEG-to-Letter training"""
    
    def __init__(self, eeg_data, labels, images=None):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.images = torch.FloatTensor(images) if images is not None else None
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]
        
        if self.images is not None:
            image = self.images[idx]
            return eeg, label, image
        else:
            return eeg, label

def load_crell_data():
    """Load Crell dataset"""
    print("📂 Loading Crell dataset...")
    
    with open('crell_processed_data_correct.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_checkpoint_and_resume():
    """Load checkpoint and resume training with improved parameters"""
    print("🔄 Loading checkpoint for resume training...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load data
    data = load_crell_data()
    
    # Create datasets
    train_dataset = CrellEEGDataset(
        data['training']['eeg'],
        data['training']['labels'],
        data['training']['images']
    )
    
    val_dataset = CrellEEGDataset(
        data['validation']['eeg'],
        data['validation']['labels']
    )
    
    # Create data loaders with smaller batch size for better gradients
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced from 16
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = EnhancedEEGToEmbeddingModel(
        n_channels=64,
        seq_len=500,
        d_model=256,
        embedding_dim=512,
        nhead=8,
        num_layers=8,
        patch_size=25,
        dropout=0.2  # Increased dropout for better generalization
    ).to(device)
    
    # Add improved classifier
    model.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),  # Higher dropout
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)  # 10 letters
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('advanced_eeg_checkpoint_50.pth', map_location=device)

    # Load model state dict with flexibility for new classifier
    model_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()

    # Load only matching keys (exclude classifier if not present)
    filtered_state = {k: v for k, v in model_state.items() if k in current_model_state and current_model_state[k].shape == v.shape}
    current_model_state.update(filtered_state)
    model.load_state_dict(current_model_state)

    print(f"✅ Loaded {len(filtered_state)} layers from checkpoint")
    
    # Improved loss functions
    contrastive_loss = ImprovedContrastiveLoss(temperature=0.03, margin=2.0)  # Lower temp
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # Improved optimizer with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)  # Lower LR, higher weight decay
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        except:
            print("⚠️ Could not load optimizer state, using fresh optimizer")
    
    # Improved scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resuming from epoch {start_epoch}")
    
    return model, optimizer, scheduler, train_loader, val_loader, start_epoch, device, contrastive_loss, classification_loss

def train_epoch_improved(model, train_loader, optimizer, contrastive_loss, classification_loss, device):
    """Improved training epoch with better loss weighting"""
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_classification = 0
    
    for eeg, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        eeg = eeg.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(eeg)
        
        # Contrastive loss (primary)
        cont_loss = contrastive_loss(embeddings, labels)
        
        # Classification loss (auxiliary)
        if hasattr(model, 'classifier'):
            logits = model.classifier(embeddings)
            class_loss = classification_loss(logits, labels)
        else:
            class_loss = torch.tensor(0.0).to(device)
        
        # Improved loss weighting: More emphasis on contrastive learning
        loss = 0.8 * cont_loss + 0.2 * class_loss  # Changed from 1.0 + 0.1
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_contrastive += cont_loss.item()
        total_classification += class_loss.item()
    
    return total_loss / len(train_loader), total_contrastive / len(train_loader), total_classification / len(train_loader)

def validate_epoch_improved(model, val_loader, contrastive_loss, classification_loss, device):
    """Improved validation with accuracy calculation"""
    model.eval()
    total_loss = 0
    total_contrastive = 0
    total_classification = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for eeg, labels in tqdm(val_loader, desc="Validating", leave=False):
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(eeg)
            
            # Contrastive loss
            cont_loss = contrastive_loss(embeddings, labels)
            
            # Classification loss and accuracy
            if hasattr(model, 'classifier'):
                logits = model.classifier(embeddings)
                class_loss = classification_loss(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
            else:
                class_loss = torch.tensor(0.0).to(device)
            
            # Combined loss
            loss = 0.8 * cont_loss + 0.2 * class_loss
            
            total_loss += loss.item()
            total_contrastive += cont_loss.item()
            total_classification += class_loss.item()
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return (total_loss / len(val_loader), 
            total_contrastive / len(val_loader), 
            total_classification / len(val_loader),
            accuracy)

def resume_training_improved():
    """Main function to resume training with improvements"""
    print("🚀 RESUMING TRAINING WITH IMPROVED PARAMETERS")
    print("=" * 80)
    print("🎯 Fixes: Lower temperature, better loss weighting, improved optimizer")
    print("=" * 80)
    
    # Load checkpoint and setup
    model, optimizer, scheduler, train_loader, val_loader, start_epoch, device, contrastive_loss, classification_loss = load_checkpoint_and_resume()
    
    # Training parameters
    num_epochs = 400
    best_val_accuracy = 0
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    print(f"\n🎯 Improved Training Configuration:")
    print(f"   Resume from epoch: {start_epoch}")
    print(f"   Total epochs: {num_epochs}")
    print(f"   Batch size: 8 (reduced for better gradients)")
    print(f"   Learning rate: 5e-5 (reduced)")
    print(f"   Contrastive temperature: 0.03 (lower for better separation)")
    print(f"   Loss weighting: 80% contrastive + 20% classification")
    print(f"   Dropout: 0.2 model + 0.3 classifier (increased)")
    print(f"   Scheduler: CosineAnnealingWarmRestarts")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Train
        train_loss, train_cont, train_class = train_epoch_improved(
            model, train_loader, optimizer, contrastive_loss, classification_loss, device
        )
        
        # Validate
        val_loss, val_cont, val_class, val_accuracy = validate_epoch_improved(
            model, val_loader, contrastive_loss, classification_loss, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"📈 Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train={train_loss:.4f} (C:{train_cont:.4f}, Cl:{train_class:.4f}), "
                  f"Val={val_loss:.4f} (C:{val_cont:.4f}, Cl:{val_class:.4f}), "
                  f"Acc={val_accuracy:.3f}, LR={current_lr:.2e}")
        
        # Save best model based on accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss
            }, 'improved_eeg_model_best.pth')
            print(f"   ✅ New best model saved! Accuracy: {val_accuracy:.3f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss
            }, f'improved_eeg_checkpoint_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    training_time = time.time() - start_time
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'train_loss': train_loss
    }, 'improved_eeg_model_final.pth')
    
    # Plot improved training curves
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(range(start_epoch, num_epochs), train_losses, label='Training Loss', alpha=0.8)
    plt.plot(range(start_epoch, num_epochs), val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved Training: Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curve
    plt.subplot(2, 2, 2)
    plt.plot(range(start_epoch, num_epochs), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Improved Training: Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate
    plt.subplot(2, 2, 3)
    plt.plot(range(start_epoch, num_epochs), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Warm Restarts)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Best accuracy over time
    plt.subplot(2, 2, 4)
    best_acc_so_far = np.maximum.accumulate(val_accuracies)
    plt.plot(range(start_epoch, num_epochs), best_acc_so_far, label='Best Accuracy So Far', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Best Accuracy')
    plt.title('Best Accuracy Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Improved training completed!")
    print(f"   Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"   Best validation accuracy: {best_val_accuracy:.3f} ({best_val_accuracy*100:.1f}%)")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Final validation accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
    print(f"   Accuracy improvement: {(val_accuracy - val_accuracies[0]):.3f}")
    
    print(f"\n📁 Generated files:")
    print(f"   - improved_eeg_model_best.pth")
    print(f"   - improved_eeg_model_final.pth")
    print(f"   - improved_training_results.png")
    
    print(f"\n🎯 Issues Fixed:")
    print(f"   ✅ Training resumed from epoch {start_epoch} to {num_epochs}")
    print(f"   ✅ Lower temperature (0.03) for better letter separation")
    print(f"   ✅ Improved loss weighting (80% contrastive + 20% classification)")
    print(f"   ✅ Better optimizer settings and regularization")
    print(f"   ✅ Accuracy tracking and best model saving")
    
    return model, train_losses, val_losses, val_accuracies

if __name__ == "__main__":
    model, train_losses, val_losses, val_accuracies = resume_training_improved()
