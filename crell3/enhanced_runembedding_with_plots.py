#!/usr/bin/env python3
"""
Enhanced Runembedding with Comprehensive Plotting
Adds test loss tracking and comprehensive visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import clip
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing components (assuming they exist)
try:
    from runembedding import MiyawakiDecoder, MiyawakiDataset, ContrastiveLoss
    print("✅ Imported existing components")
except ImportError:
    print("⚠️ Could not import existing components, will need to define them")

def enhanced_train_with_validation(decoder, train_loader, test_loader, epochs=100, lr=1e-3):
    """Enhanced training with validation tracking"""
    print("🚀 Starting enhanced training with validation tracking...")
    
    optimizer = optim.Adam(decoder.fmri_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Track both training and validation metrics
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # TRAINING PHASE
        decoder.fmri_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        train_batches = 0
        
        for batch_idx, (fmri, images) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            fmri = fmri.to(decoder.device)
            images = images.to(decoder.device)
            
            # Forward pass
            fmri_emb = decoder.fmri_encoder(fmri)
            
            with torch.no_grad():
                image_emb = decoder.clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
            
            # Compute loss
            loss = criterion(fmri_emb, image_emb)
            
            # Compute accuracy (cosine similarity > threshold)
            similarities = F.cosine_similarity(fmri_emb, image_emb, dim=1)
            accuracy = (similarities > 0.0).float().mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += accuracy.item()
            train_batches += 1
        
        # VALIDATION PHASE
        decoder.fmri_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for fmri, images in tqdm(test_loader, desc=f"Val Epoch {epoch+1}"):
                fmri = fmri.to(decoder.device)
                images = images.to(decoder.device)
                
                # Forward pass
                fmri_emb = decoder.fmri_encoder(fmri)
                image_emb = decoder.clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
                
                # Compute loss and accuracy
                loss = criterion(fmri_emb, image_emb)
                similarities = F.cosine_similarity(fmri_emb, image_emb, dim=1)
                accuracy = (similarities > 0.0).float().mean()
                
                epoch_val_loss += loss.item()
                epoch_val_acc += accuracy.item()
                val_batches += 1
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate averages
        avg_train_loss = epoch_train_loss / train_batches
        avg_val_loss = epoch_val_loss / val_batches
        avg_train_acc = epoch_train_acc / train_batches
        avg_val_acc = epoch_val_acc / val_batches
        
        # Store metrics
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_accuracies'].append(avg_train_acc)
        history['val_accuracies'].append(avg_val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                  f"Train Acc={avg_train_acc:.3f}, Val Acc={avg_val_acc:.3f}, LR={current_lr:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'fmri_encoder_state_dict': decoder.fmri_encoder.state_dict(),
                'scaler': decoder.scaler,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history
            }, 'miyawaki_contrastive_clip_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    return history

def plot_comprehensive_training_results(history, save_plots=True):
    """Create comprehensive training plots"""
    print("📊 Creating comprehensive training plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Miyawaki Training Results - Train vs Validation', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', alpha=0.8, color='blue')
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.8, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Contrastive Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_accuracies'], label='Training Accuracy', alpha=0.8, color='blue')
    axes[0, 1].plot(epochs, history['val_accuracies'], label='Validation Accuracy', alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    axes[0, 2].plot(epochs, history['learning_rates'], alpha=0.8, color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Loss difference (overfitting detection)
    loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
    axes[1, 0].plot(epochs, loss_diff, alpha=0.8, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val Loss - Train Loss')
    axes[1, 0].set_title('Overfitting Detection')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Accuracy difference
    acc_diff = np.array(history['val_accuracies']) - np.array(history['train_accuracies'])
    axes[1, 1].plot(epochs, acc_diff, alpha=0.8, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Acc - Train Acc')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training summary
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    best_val_loss = min(history['val_losses'])
    best_val_acc = max(history['val_accuracies'])
    
    summary_text = f"""Training Summary:

Final Results:
  Train Loss: {final_train_loss:.4f}
  Val Loss: {final_val_loss:.4f}
  Train Acc: {final_train_acc:.3f}
  Val Acc: {final_val_acc:.3f}

Best Performance:
  Best Val Loss: {best_val_loss:.4f}
  Best Val Acc: {best_val_acc:.3f}

Training Info:
  Total Epochs: {len(epochs)}
  Final LR: {history['learning_rates'][-1]:.2e}
  
Overfitting Analysis:
  Loss Gap: {final_val_loss - final_train_loss:.4f}
  Acc Gap: {final_val_acc - final_train_acc:.3f}"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Training Summary')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('miyawaki_enhanced_training_results.png', dpi=300, bbox_inches='tight')
        print("📊 Plot saved: miyawaki_enhanced_training_results.png")
    
    plt.show()
    
    return fig

def run_enhanced_training():
    """Run enhanced training with comprehensive plotting"""
    print("🚀 ENHANCED MIYAWAKI TRAINING WITH COMPREHENSIVE PLOTS")
    print("=" * 70)
    
    try:
        # Initialize decoder (assuming MiyawakiDecoder exists)
        decoder = MiyawakiDecoder()
        
        # Load data
        mat_file_path = "../dataset/miyawaki_structured_28x28.mat"
        decoder.load_data(mat_file_path)
        
        # Initialize models
        decoder.initialize_models()
        
        # Create dataloaders
        train_loader, test_loader = decoder.create_dataloaders(batch_size=32)
        
        print(f"📊 Dataset loaded:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        
        # Run enhanced training
        history = enhanced_train_with_validation(
            decoder, train_loader, test_loader, 
            epochs=100, lr=1e-3
        )
        
        # Create comprehensive plots
        plot_comprehensive_training_results(history, save_plots=True)
        
        # Save final model
        torch.save({
            'fmri_encoder_state_dict': decoder.fmri_encoder.state_dict(),
            'scaler': decoder.scaler,
            'history': history
        }, 'miyawaki_contrastive_clip_enhanced.pth')
        
        print("✅ Enhanced training completed!")
        print("📊 Files generated:")
        print("   - miyawaki_enhanced_training_results.png")
        print("   - miyawaki_contrastive_clip_best.pth")
        print("   - miyawaki_contrastive_clip_enhanced.pth")
        
        return decoder, history
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure runembedding.py components are available")
        return None, None

def main():
    """Main function"""
    return run_enhanced_training()

if __name__ == "__main__":
    main()
