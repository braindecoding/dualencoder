#!/usr/bin/env python3
"""
Enhanced Training for MBD2 - MindBigData EEG Dataset
Advanced EEG-to-Image reconstruction with improved architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import clip
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel
from advanced_loss_functions import AdaptiveTemperatureContrastiveLoss, WarmupCosineScheduler

def load_mbd2_data():
    """Load MBD2 explicit data splits"""
    print("📂 Loading MBD2 explicit data splits...")
    
    with open('explicit_eeg_data_splits.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Extract all data splits
    eegTrn = data['training']['eegTrn']
    stimTrn = data['training']['stimTrn']
    labelsTrn = data['training']['labelsTrn']
    
    eegVal = data['validation']['eegVal']
    stimVal = data['validation']['stimVal']
    labelsVal = data['validation']['labelsVal']
    
    eegTest = data['test']['eegTest']
    stimTest = data['test']['stimTest']
    labelsTest = data['test']['labelsTest']
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training: {len(eegTrn)} samples")
    print(f"   Validation: {len(eegVal)} samples")
    print(f"   Test: {len(eegTest)} samples")
    print(f"   EEG shape: {eegTrn.shape}")
    print(f"   Labels: {set(labelsTrn)} (digits 0-9)")
    
    return eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest

def create_enhanced_model(device):
    """Create enhanced EEG model with improved architecture"""
    model = EnhancedEEGToEmbeddingModel(
        n_channels=14,
        seq_len=256,
        d_model=512,  # Increased from 256
        embedding_dim=512,
        nhead=16,     # Increased from 8
        num_layers=12, # Increased from 8
        patch_size=16,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🤖 Enhanced Model Created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def evaluate_model(model, eeg_data, stim_data, labels, clip_model, clip_preprocess, device, phase_name="validation", batch_size=32):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(eeg_data), batch_size):
            batch_eeg = eeg_data[i:i+batch_size]
            batch_stim = stim_data[i:i+batch_size]
            
            # Convert to tensors
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Process images for CLIP
            processed_images = []
            for img in batch_stim:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Forward pass
            eeg_embeddings = model(batch_eeg)
            clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Normalize embeddings
            eeg_embeddings = nn.functional.normalize(eeg_embeddings, dim=1)
            clip_embeddings = nn.functional.normalize(clip_embeddings, dim=1)
            
            # Compute similarity and accuracy
            similarities = torch.cosine_similarity(eeg_embeddings, clip_embeddings, dim=1)
            accuracy = (similarities > 0.0).float().mean()
            
            # Simple loss (negative cosine similarity)
            loss = -similarities.mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_accuracy

def train_enhanced_model():
    """Train enhanced MBD2 model"""
    print("🚀 ENHANCED MBD2 TRAINING")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_epochs': 200,
        'batch_size': 16,  # Reduced for larger model
        'learning_rate': 5e-5,  # Reduced for stability
        'weight_decay': 1e-4,
        'patience': 30,
        'gradient_clip': 1.0,
        'warmup_epochs': 20,
        'save_frequency': 25
    }
    
    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest = load_mbd2_data()
    
    # Load CLIP model
    print(f"\n📥 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print(f"✅ CLIP model loaded and frozen")
    
    # Create enhanced model
    model = create_enhanced_model(device)
    
    # Create loss function
    loss_fn = AdaptiveTemperatureContrastiveLoss(
        initial_temperature=0.07,
        learn_temperature=True
    ).to(device)
    
    # Create optimizer with differential learning rates
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'embedding_projection' in name or 'final_norm' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['learning_rate']},
        {'params': head_params, 'lr': config['learning_rate'] * 2},
        {'params': loss_fn.parameters(), 'lr': config['learning_rate'] * 5}
    ], weight_decay=config['weight_decay'])
    
    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config['warmup_epochs'],
        max_epochs=config['num_epochs'],
        base_lr=config['learning_rate'],
        min_lr=config['learning_rate'] * 0.01
    )
    
    # Training history
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'learning_rates': [],
        'temperatures': []
    }
    
    best_val_accuracy = 0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting Enhanced Training...")
    print(f"   Target: Improve upon 8.0% test accuracy")
    print(f"   Strategy: Larger model + adaptive temperature + differential LR")
    
    for epoch in range(config['num_epochs']):
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # TRAINING PHASE
        model.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        epoch_temperature = 0
        num_train_batches = 0
        
        # Create training batches
        indices = np.arange(len(eegTrn))
        np.random.shuffle(indices)
        
        train_pbar = tqdm(range(0, len(indices), config['batch_size']), 
                         desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for i in train_pbar:
            batch_indices = indices[i:i+config['batch_size']]
            batch_eeg = eegTrn[batch_indices]
            batch_stim = [stimTrn[idx] for idx in batch_indices]
            
            # Move EEG to device
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Process images for CLIP
            processed_images = []
            for img in batch_stim:
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Forward pass
            eeg_embeddings = model(batch_eeg)
            
            with torch.no_grad():
                clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Compute loss with adaptive temperature
            loss, accuracy, temperature = loss_fn(eeg_embeddings, clip_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy.item()
            epoch_temperature += temperature
            num_train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy.item():.3f}",
                'Temp': f"{temperature:.3f}"
            })
        
        # VALIDATION PHASE
        val_loss, val_accuracy = evaluate_model(
            model, eegVal, stimVal, labelsVal,
            clip_model, clip_preprocess, device, "validation", config['batch_size']
        )
        
        # Record metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_accuracy = epoch_train_accuracy / num_train_batches
        avg_temperature = epoch_temperature / num_train_batches
        
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(avg_train_accuracy)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_accuracy)
        history['learning_rates'].append(current_lr)
        history['temperatures'].append(avg_temperature)
        
        # Early stopping and best model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'config': config,
                'history': history
            }, 'enhanced_mbd2_model_best.pth')
            
            print(f"✅ New best model saved! Val accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.3f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.3f}, LR={current_lr:.2e}, Temp={avg_temperature:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'config': config,
                'history': history
            }, f'enhanced_mbd2_checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    print(f"\n🔍 Final Evaluation on Test Set...")
    model.load_state_dict(torch.load('enhanced_mbd2_model_best.pth')['model_state_dict'])
    test_loss, test_accuracy = evaluate_model(
        model, eegTest, stimTest, labelsTest,
        clip_model, clip_preprocess, device, "test", config['batch_size']
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time
    }, 'enhanced_mbd2_model_final.pth')
    
    # Create training visualization
    plot_training_results(history, best_val_accuracy, test_accuracy, training_time)
    
    print(f"\n🏆 ENHANCED TRAINING COMPLETED!")
    print(f"   Training time: {training_time/60:.1f} minutes")
    print(f"   Best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.1f}%)")
    print(f"   Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"   Epochs trained: {len(history['train_losses'])}")
    
    # Compare with previous results
    previous_best = 0.08  # 8.0% from README
    improvement = (test_accuracy - previous_best) / previous_best * 100
    print(f"\n📊 IMPROVEMENT ANALYSIS:")
    print(f"   Previous best: {previous_best:.3f} (8.0%)")
    print(f"   Current result: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    if improvement > 0:
        print(f"   Improvement: +{improvement:.1f}% 🎉")
    else:
        print(f"   Change: {improvement:.1f}%")
    
    return model, history, best_val_accuracy, test_accuracy

def plot_training_results(history, best_val_acc, test_acc, training_time):
    """Plot training results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced MBD2 Training Results', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', alpha=0.8)
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_accuracies'], label='Training Accuracy', alpha=0.8)
    axes[0, 1].plot(epochs, history['val_accuracies'], label='Validation Accuracy', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    axes[0, 2].plot(epochs, history['learning_rates'], alpha=0.8, color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Temperature evolution
    axes[1, 0].plot(epochs, history['temperatures'], alpha=0.8, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('Adaptive Temperature')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Performance summary
    summary_text = f"""Enhanced MBD2 Training Summary:

Architecture:
  • Model: Enhanced EEG Transformer
  • Dimensions: 512d, 16 heads, 12 layers
  • Parameters: ~30M (estimated)

Training Results:
  • Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)
  • Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)
  • Training Time: {training_time/60:.1f} minutes
  • Epochs: {len(history['train_losses'])}

Improvements:
  • Larger model capacity
  • Adaptive temperature learning
  • Differential learning rates
  • Warmup + cosine scheduling

Target: Beat 8.0% baseline"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Training Summary')
    
    # Plot 6: Accuracy improvement over time
    val_accs = history['val_accuracies']
    best_so_far = []
    current_best = 0
    for acc in val_accs:
        if acc > current_best:
            current_best = acc
        best_so_far.append(current_best)
    
    axes[1, 2].plot(epochs, val_accs, alpha=0.6, label='Validation Accuracy')
    axes[1, 2].plot(epochs, best_so_far, linewidth=2, label='Best So Far')
    axes[1, 2].axhline(y=0.08, color='red', linestyle='--', alpha=0.7, label='Previous Best (8.0%)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy Improvement Progress')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'enhanced_mbd2_training_results_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training plot saved: {plot_filename}")

def main():
    """Main function"""
    print("🧠 ENHANCED MBD2 TRAINING - MINDBIGDATA EEG DATASET")
    print("=" * 70)
    print("Goal: Improve upon 8.0% test accuracy with enhanced architecture")
    print("Strategy: Larger model + adaptive temperature + differential learning rates")
    print("=" * 70)
    
    # Run enhanced training
    model, history, best_val_acc, test_acc = train_enhanced_model()
    
    return model, history, best_val_acc, test_acc

if __name__ == "__main__":
    main()
