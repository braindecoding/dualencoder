#!/usr/bin/env python3
"""
Advanced EEG Contrastive Learning Training
Combines all improvements:
- Enhanced transformer architecture
- Data augmentation
- Advanced loss functions
- Improved learning rate scheduling
- Better monitoring and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
import warnings
import time
import pickle
from datetime import datetime
warnings.filterwarnings("ignore")

# Import our enhanced components
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel
from data_augmentation import EEGAugmentation, AugmentedDataLoader
from advanced_loss_functions import (
    AdaptiveTemperatureContrastiveLoss, 
    FocalContrastiveLoss,
    MultiScaleContrastiveLoss,
    WarmupCosineScheduler
)
from explicit_eeg_contrastive_training import load_explicit_data, evaluate_explicit

class AdvancedTrainingConfig:
    """
    Configuration for advanced training
    """
    def __init__(self):
        # Model architecture
        self.n_channels = 14
        self.seq_len = 256
        self.d_model = 256  # Enhanced from 128
        self.embedding_dim = 512
        self.nhead = 8
        self.num_layers = 8  # Enhanced from 6
        self.patch_size = 16
        self.dropout = 0.1
        
        # Training parameters
        self.num_epochs = 300
        self.batch_size = 32
        self.base_lr = 5e-5  # Lower learning rate
        self.min_lr = 1e-6
        self.warmup_epochs = 20
        self.patience = 60  # Increased patience
        self.weight_decay = 1e-4
        self.gradient_clip = 1.0
        
        # Loss function
        self.loss_type = 'adaptive_temperature'  # 'adaptive_temperature', 'focal', 'multiscale'
        self.temperature = 0.05  # Lower temperature
        
        # Data augmentation
        self.use_augmentation = True
        self.augment_prob = 0.8
        self.noise_std = 0.03
        self.temporal_shift_range = 8
        self.electrode_dropout_prob = 0.15
        
        # Monitoring
        self.save_frequency = 50
        self.eval_frequency = 5
        
    def __str__(self):
        return f"""Advanced Training Configuration:
        Model: Enhanced Transformer (d_model={self.d_model}, layers={self.num_layers})
        Training: {self.num_epochs} epochs, lr={self.base_lr}, batch_size={self.batch_size}
        Loss: {self.loss_type}, temperature={self.temperature}
        Augmentation: {'Enabled' if self.use_augmentation else 'Disabled'}
        """

def create_advanced_model(config, device):
    """
    Create enhanced EEG model
    """
    model = EnhancedEEGToEmbeddingModel(
        n_channels=config.n_channels,
        seq_len=config.seq_len,
        d_model=config.d_model,
        embedding_dim=config.embedding_dim,
        nhead=config.nhead,
        num_layers=config.num_layers,
        patch_size=config.patch_size,
        dropout=config.dropout
    ).to(device)
    
    return model

def create_advanced_loss(config, device):
    """
    Create advanced loss function
    """
    if config.loss_type == 'adaptive_temperature':
        return AdaptiveTemperatureContrastiveLoss(
            initial_temperature=config.temperature,
            learn_temperature=True
        ).to(device)
    elif config.loss_type == 'focal':
        return FocalContrastiveLoss(
            temperature=config.temperature,
            alpha=1.0,
            gamma=2.0
        ).to(device)
    elif config.loss_type == 'multiscale':
        return MultiScaleContrastiveLoss(
            embedding_dims=[256, 512, 1024],
            temperature=config.temperature
        ).to(device)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")

def create_advanced_optimizer_and_scheduler(model, config):
    """
    Create advanced optimizer and scheduler
    """
    # Different learning rates for different parts
    param_groups = [
        {'params': model.encoder.spatial_projection.parameters(), 'lr': config.base_lr * 0.5},
        {'params': model.encoder.patch_embed.parameters(), 'lr': config.base_lr * 0.8},
        {'params': model.encoder.transformer.parameters(), 'lr': config.base_lr},
        {'params': model.encoder.embedding_projector.parameters(), 'lr': config.base_lr * 1.5},
        {'params': model.embedding_adapter.parameters(), 'lr': config.base_lr * 2.0}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.num_epochs,
        base_lr=config.base_lr,
        min_lr=config.min_lr
    )
    
    return optimizer, scheduler

def advanced_training_loop(config):
    """
    Main advanced training loop
    """
    print("🚀 ADVANCED EEG CONTRASTIVE LEARNING")
    print("=" * 70)
    print(config)
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load CLIP model
    print(f"\n📥 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print(f"✅ CLIP model loaded and frozen")
    
    # Load data
    print(f"\n📊 Loading data...")
    eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest = load_explicit_data()
    
    print(f"   Training: {len(eegTrn)} samples")
    print(f"   Validation: {len(eegVal)} samples")
    print(f"   Test: {len(eegTest)} samples")
    
    # Create enhanced model
    print(f"\n🧠 Creating enhanced model...")
    model = create_advanced_model(config, device)
    
    # Create advanced loss function
    print(f"\n🎯 Creating advanced loss function...")
    loss_fn = create_advanced_loss(config, device)
    print(f"   Loss type: {config.loss_type}")
    
    # Create optimizer and scheduler
    print(f"\n⚙️  Creating optimizer and scheduler...")
    optimizer, scheduler = create_advanced_optimizer_and_scheduler(model, config)
    print(f"   Optimizer: AdamW with differential learning rates")
    print(f"   Scheduler: Warmup + Cosine Annealing")
    
    # Create data augmentation
    if config.use_augmentation:
        print(f"\n🔄 Creating data augmentation...")
        augmenter = EEGAugmentation(
            noise_std=config.noise_std,
            temporal_shift_range=config.temporal_shift_range,
            electrode_dropout_prob=config.electrode_dropout_prob,
            augment_prob=config.augment_prob
        )
        print(f"   Augmentation probability: {config.augment_prob}")
    else:
        augmenter = None
    
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
    
    print(f"\n🚀 Starting Advanced Training...")
    print(f"   Target epochs: {config.num_epochs}")
    print(f"   Early stopping patience: {config.patience}")
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # TRAINING PHASE
        model.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        num_train_batches = 0
        epoch_temperature = 0
        
        # Create training batches with augmentation
        if config.use_augmentation:
            train_loader = AugmentedDataLoader(
                eegTrn, stimTrn, labelsTrn, augmenter, 
                batch_size=config.batch_size, shuffle=True
            )
        else:
            # Use original data loader
            from explicit_eeg_contrastive_training import create_batches
            train_batches = create_batches(eegTrn, stimTrn, labelsTrn, config.batch_size, shuffle=True)
            train_loader = train_batches
        
        # Training progress bar
        if config.use_augmentation:
            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{config.num_epochs} [TRAIN]",
                total=len(train_loader)
            )
        else:
            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{config.num_epochs} [TRAIN]",
                total=len(eegTrn) // config.batch_size
            )
        
        for batch_data in train_progress:
            if config.use_augmentation:
                batch_eeg, batch_stim, batch_labels = batch_data
            else:
                batch_eeg, batch_stim, batch_labels = batch_data
            
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
            
            with torch.no_grad():
                clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Compute loss
            if config.loss_type == 'adaptive_temperature':
                loss, accuracy, temperature = loss_fn(eeg_embeddings, clip_embeddings)
                epoch_temperature += temperature
            else:
                loss, accuracy = loss_fn(eeg_embeddings, clip_embeddings)
                epoch_temperature += config.temperature
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy.item()
            num_train_batches += 1
            
            # Update progress bar
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy.item():.3f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # VALIDATION PHASE (every eval_frequency epochs)
        if (epoch + 1) % config.eval_frequency == 0:
            val_loss, val_accuracy = evaluate_explicit(
                model, eegVal, stimVal, labelsVal,
                clip_model, clip_preprocess, device, "validation", config.batch_size
            )
        else:
            val_loss, val_accuracy = history['val_losses'][-1] if history['val_losses'] else 0, \
                                   history['val_accuracies'][-1] if history['val_accuracies'] else 0
        
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
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} ({epoch_time:.1f}s, Total: {total_time/3600:.1f}h):")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.3f}")
        print(f"   LR: {current_lr:.2e}, Temperature: {avg_temperature:.4f}")
        
        # Early stopping and best model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'history': history,
                'best_val_accuracy': best_val_accuracy,
                'training_time_hours': total_time / 3600
            }, 'mbd2/advanced_eeg_model_best.pth')
            
            print(f"   ✅ New best validation accuracy: {best_val_accuracy:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_accuracy:.3f}")
            break
        
        # Save checkpoint
        if (epoch + 1) % config.save_frequency == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'history': history,
                'config': config.__dict__
            }, f'mbd2/advanced_eeg_checkpoint_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    # Final evaluation
    print(f"\n🎯 FINAL EVALUATION:")
    
    # Load best model
    checkpoint = torch.load('mbd2/advanced_eeg_model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_accuracy = evaluate_explicit(
        model, eegTest, stimTest, labelsTest,
        clip_model, clip_preprocess, device, "test", config.batch_size
    )
    
    total_training_time = time.time() - start_time
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.3f} ({best_val_accuracy*100:.1f}%)")
    print(f"   Final Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print(f"   Training Time: {total_training_time/3600:.1f} hours")
    print(f"   Total Epochs: {len(history['train_losses'])}")
    
    return model, history, best_val_accuracy, test_accuracy

def main():
    """
    Main function to run advanced training
    """
    # Create configuration
    config = AdvancedTrainingConfig()
    
    # Run training
    model, history, best_val_acc, test_acc = advanced_training_loop(config)
    
    # Plot results
    plot_advanced_results(history, best_val_acc, test_acc)
    
    return model, history

def plot_advanced_results(history, best_val_acc, test_acc):
    """
    Plot comprehensive training results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Advanced EEG Training Results\nBest Val: {best_val_acc:.3f}, Test: {test_acc:.3f}', fontsize=16)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_losses'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_accuracies'], label='Train Accuracy', alpha=0.7)
    axes[0, 1].plot(epochs, history['val_accuracies'], label='Validation Accuracy', alpha=0.7)
    axes[0, 1].axhline(y=test_acc, color='red', linestyle='--', label=f'Test Accuracy ({test_acc:.3f})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[0, 2].plot(epochs, history['learning_rates'], label='Learning Rate', color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Temperature plot
    axes[1, 0].plot(epochs, history['temperatures'], label='Temperature', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('Temperature Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy improvement
    val_improvement = np.array(history['val_accuracies']) - history['val_accuracies'][0]
    axes[1, 1].plot(epochs, val_improvement * 100, label='Val Accuracy Improvement (%)', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('Validation Accuracy Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Training efficiency
    train_val_gap = np.array(history['train_accuracies']) - np.array(history['val_accuracies'])
    axes[1, 2].plot(epochs, train_val_gap, label='Train-Val Gap', color='red')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy Gap')
    axes[1, 2].set_title('Overfitting Monitor (Train-Val Gap)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'mbd2/advanced_training_results_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Training results plot saved as: {plot_filename}")

if __name__ == "__main__":
    main()
