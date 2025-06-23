#!/usr/bin/env python3
"""
Enhanced Miyawaki Training with Comprehensive Plotting
Adds test loss tracking and comprehensive visualization for Miyawaki dataset
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
from datetime import datetime
warnings.filterwarnings('ignore')

# Import existing Miyawaki components
from runembedding import MiyawakiDecoder, MiyawakiDataset, ContrastiveLoss

def enhanced_miyawaki_train_with_validation(decoder, train_loader, test_loader, epochs=100, lr=1e-3):
    """Enhanced Miyawaki training with validation tracking"""
    print("🧠 Starting enhanced Miyawaki training with validation tracking...")
    print(f"📊 Dataset: Miyawaki fMRI → Visual stimulus reconstruction")
    print(f"🎯 Task: Contrastive learning between fMRI and CLIP embeddings")
    
    optimizer = optim.Adam(decoder.fmri_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Track both training and validation metrics
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_similarities': [],
        'val_similarities': [],
        'train_top1_acc': [],
        'val_top1_acc': [],
        'train_top5_acc': [],
        'val_top5_acc': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_val_top1 = 0.0
    patience = 25
    patience_counter = 0
    
    print(f"🚀 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Patience: {patience}")
    print(f"   Optimizer: Adam with weight decay 1e-4")
    print(f"   Scheduler: CosineAnnealingLR")
    
    for epoch in range(epochs):
        # TRAINING PHASE
        decoder.fmri_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_sim = 0.0
        train_batches = 0
        
        train_fmri_embs = []
        train_image_embs = []
        
        for batch_idx, (fmri, images) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            fmri = fmri.to(decoder.device)
            images = images.to(decoder.device)
            
            # Forward pass
            fmri_emb = decoder.fmri_encoder(fmri)
            
            with torch.no_grad():
                image_emb = decoder.clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
            
            # Normalize fMRI embeddings
            fmri_emb = F.normalize(fmri_emb, dim=-1)
            
            # Compute loss
            loss = criterion(fmri_emb, image_emb)
            
            # Compute similarity
            similarities = F.cosine_similarity(fmri_emb, image_emb, dim=1)
            avg_similarity = similarities.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_sim += avg_similarity.item()
            train_batches += 1
            
            # Store embeddings for retrieval metrics
            train_fmri_embs.append(fmri_emb.detach().cpu())
            train_image_embs.append(image_emb.detach().cpu())
        
        # Calculate training retrieval metrics
        train_fmri_all = torch.cat(train_fmri_embs, dim=0)
        train_image_all = torch.cat(train_image_embs, dim=0)
        train_sim_matrix = torch.matmul(train_fmri_all, train_image_all.T)
        
        train_top1_acc = calculate_top_k_accuracy(train_sim_matrix, k=1)
        train_top5_acc = calculate_top_k_accuracy(train_sim_matrix, k=5)
        
        # VALIDATION PHASE
        decoder.fmri_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_sim = 0.0
        val_batches = 0
        
        val_fmri_embs = []
        val_image_embs = []
        
        with torch.no_grad():
            for fmri, images in tqdm(test_loader, desc=f"Val Epoch {epoch+1}"):
                fmri = fmri.to(decoder.device)
                images = images.to(decoder.device)
                
                # Forward pass
                fmri_emb = decoder.fmri_encoder(fmri)
                image_emb = decoder.clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
                fmri_emb = F.normalize(fmri_emb, dim=-1)
                
                # Compute loss and similarity
                loss = criterion(fmri_emb, image_emb)
                similarities = F.cosine_similarity(fmri_emb, image_emb, dim=1)
                avg_similarity = similarities.mean()
                
                epoch_val_loss += loss.item()
                epoch_val_sim += avg_similarity.item()
                val_batches += 1
                
                # Store embeddings for retrieval metrics
                val_fmri_embs.append(fmri_emb.cpu())
                val_image_embs.append(image_emb.cpu())
        
        # Calculate validation retrieval metrics
        val_fmri_all = torch.cat(val_fmri_embs, dim=0)
        val_image_all = torch.cat(val_image_embs, dim=0)
        val_sim_matrix = torch.matmul(val_fmri_all, val_image_all.T)
        
        val_top1_acc = calculate_top_k_accuracy(val_sim_matrix, k=1)
        val_top5_acc = calculate_top_k_accuracy(val_sim_matrix, k=5)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate averages
        avg_train_loss = epoch_train_loss / train_batches
        avg_val_loss = epoch_val_loss / val_batches
        avg_train_sim = epoch_train_sim / train_batches
        avg_val_sim = epoch_val_sim / val_batches
        
        # Store metrics
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_similarities'].append(avg_train_sim)
        history['val_similarities'].append(avg_val_sim)
        history['train_top1_acc'].append(train_top1_acc)
        history['val_top1_acc'].append(val_top1_acc)
        history['train_top5_acc'].append(train_top5_acc)
        history['val_top5_acc'].append(val_top5_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"           Train Sim={avg_train_sim:.3f}, Val Sim={avg_val_sim:.3f}")
            print(f"           Train Top1={train_top1_acc:.3f}, Val Top1={val_top1_acc:.3f}")
            print(f"           Train Top5={train_top5_acc:.3f}, Val Top5={val_top5_acc:.3f}, LR={current_lr:.6f}")
        
        # Early stopping based on validation top-1 accuracy
        if val_top1_acc > best_val_top1:
            best_val_top1 = val_top1_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'fmri_encoder_state_dict': decoder.fmri_encoder.state_dict(),
                'scaler': decoder.scaler,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_top1': best_val_top1,
                'history': history
            }, 'miyawaki_contrastive_clip_best.pth')
            print(f"   💾 New best model saved! Val Top1: {val_top1_acc:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation Top-1 accuracy: {best_val_top1:.3f}")
            break
    
    return history

def calculate_top_k_accuracy(similarity_matrix, k=1):
    """Calculate top-k retrieval accuracy"""
    n_samples = similarity_matrix.shape[0]
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    correct_retrieval = 0
    for i in range(n_samples):
        if i in top_k_indices[i]:
            correct_retrieval += 1
    
    accuracy = correct_retrieval / n_samples
    return accuracy

def plot_comprehensive_miyawaki_results(history, save_plots=True):
    """Create comprehensive Miyawaki training plots"""
    print("📊 Creating comprehensive Miyawaki training plots...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Enhanced Miyawaki Training Results - fMRI to Visual Stimulus Reconstruction', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', alpha=0.8, color='blue')
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.8, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Contrastive Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Similarity curves
    axes[0, 1].plot(epochs, history['train_similarities'], label='Training Similarity', alpha=0.8, color='blue')
    axes[0, 1].plot(epochs, history['val_similarities'], label='Validation Similarity', alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Training vs Validation Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top-1 Accuracy
    axes[0, 2].plot(epochs, history['train_top1_acc'], label='Training Top-1', alpha=0.8, color='blue')
    axes[0, 2].plot(epochs, history['val_top1_acc'], label='Validation Top-1', alpha=0.8, color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Top-1 Accuracy')
    axes[0, 2].set_title('Top-1 Retrieval Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Top-5 Accuracy
    axes[1, 0].plot(epochs, history['train_top5_acc'], label='Training Top-5', alpha=0.8, color='blue')
    axes[1, 0].plot(epochs, history['val_top5_acc'], label='Validation Top-5', alpha=0.8, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy')
    axes[1, 0].set_title('Top-5 Retrieval Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning rate schedule
    axes[1, 1].plot(epochs, history['learning_rates'], alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Overfitting detection (loss difference)
    loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
    axes[1, 2].plot(epochs, loss_diff, alpha=0.8, color='orange')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Val Loss - Train Loss')
    axes[1, 2].set_title('Overfitting Detection')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Accuracy comparison
    axes[2, 0].plot(epochs, history['val_top1_acc'], label='Top-1 Accuracy', alpha=0.8, color='red')
    axes[2, 0].plot(epochs, history['val_top5_acc'], label='Top-5 Accuracy', alpha=0.8, color='blue')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Validation Accuracy')
    axes[2, 0].set_title('Validation Retrieval Performance')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Training summary
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    final_train_sim = history['train_similarities'][-1]
    final_val_sim = history['val_similarities'][-1]
    best_val_top1 = max(history['val_top1_acc'])
    best_val_top5 = max(history['val_top5_acc'])
    
    summary_text = f"""Miyawaki Training Summary:

Dataset: fMRI → Visual Stimulus
Task: Contrastive Learning (fMRI ↔ CLIP)

Final Results:
  Train Loss: {final_train_loss:.4f}
  Val Loss: {final_val_loss:.4f}
  Train Similarity: {final_train_sim:.3f}
  Val Similarity: {final_val_sim:.3f}

Best Performance:
  Best Val Top-1: {best_val_top1:.3f}
  Best Val Top-5: {best_val_top5:.3f}

Training Info:
  Total Epochs: {len(epochs)}
  Final LR: {history['learning_rates'][-1]:.2e}
  
Generalization:
  Loss Gap: {final_val_loss - final_train_loss:.4f}
  Sim Gap: {final_val_sim - final_train_sim:.3f}

Architecture:
  fMRI Encoder: 967 → 512 dims
  Target: CLIP ViT-B/32 embeddings
  Loss: Contrastive (temp=0.07)"""
    
    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Training Summary')
    
    # Plot 9: Performance progression
    val_top1_best_so_far = []
    current_best = 0
    for acc in history['val_top1_acc']:
        if acc > current_best:
            current_best = acc
        val_top1_best_so_far.append(current_best)
    
    axes[2, 2].plot(epochs, history['val_top1_acc'], alpha=0.6, label='Val Top-1 Accuracy')
    axes[2, 2].plot(epochs, val_top1_best_so_far, linewidth=2, label='Best So Far')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Top-1 Accuracy')
    axes[2, 2].set_title('Performance Progression')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'miyawaki_enhanced_training_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_filename}")
    
    plt.show()
    
    return fig

def run_enhanced_miyawaki_training():
    """Run enhanced Miyawaki training with comprehensive plotting"""
    print("🧠 ENHANCED MIYAWAKI TRAINING WITH COMPREHENSIVE PLOTS")
    print("=" * 70)
    print("Dataset: Miyawaki fMRI → Visual stimulus reconstruction")
    print("Task: Contrastive learning between fMRI signals and CLIP embeddings")
    print("=" * 70)
    
    try:
        # Initialize Miyawaki decoder
        decoder = MiyawakiDecoder()
        
        # Load Miyawaki dataset
        mat_file_path = "../dataset/miyawaki_structured_28x28.mat"
        print(f"📂 Loading Miyawaki dataset from: {mat_file_path}")
        decoder.load_data(mat_file_path)
        
        # Initialize models (fMRI encoder + CLIP)
        decoder.initialize_models()
        
        # Create dataloaders
        train_loader, test_loader = decoder.create_dataloaders(batch_size=32)
        
        print(f"📊 Miyawaki dataset loaded:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        print(f"   fMRI dimensions: 967 voxels")
        print(f"   Visual stimuli: 28x28 grayscale images")
        
        # Run enhanced training with validation tracking
        history = enhanced_miyawaki_train_with_validation(
            decoder, train_loader, test_loader, 
            epochs=100, lr=1e-3
        )
        
        # Create comprehensive plots
        plot_comprehensive_miyawaki_results(history, save_plots=True)
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'miyawaki_contrastive_clip_enhanced_{timestamp}.pth'
        torch.save({
            'fmri_encoder_state_dict': decoder.fmri_encoder.state_dict(),
            'scaler': decoder.scaler,
            'history': history
        }, final_model_path)
        
        print("✅ Enhanced Miyawaki training completed!")
        print("📊 Files generated:")
        print(f"   - miyawaki_enhanced_training_results_{timestamp}.png")
        print(f"   - miyawaki_contrastive_clip_best.pth")
        print(f"   - {final_model_path}")
        
        # Print final performance
        best_val_top1 = max(history['val_top1_acc'])
        best_val_top5 = max(history['val_top5_acc'])
        print(f"\n🏆 FINAL PERFORMANCE:")
        print(f"   Best Validation Top-1 Accuracy: {best_val_top1:.3f} ({best_val_top1*100:.1f}%)")
        print(f"   Best Validation Top-5 Accuracy: {best_val_top5:.3f} ({best_val_top5*100:.1f}%)")
        
        return decoder, history
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure:")
        print("   1. Miyawaki dataset is available at ../dataset/miyawaki_structured_28x28.mat")
        print("   2. runembedding.py components are available")
        print("   3. CLIP is properly installed")
        return None, None

def main():
    """Main function"""
    return run_enhanced_miyawaki_training()

if __name__ == "__main__":
    main()
