#!/usr/bin/env python3
"""
Training Monitor for 400 Epochs EEG Training
Real-time monitoring and visualization of training progress
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime, timedelta

def monitor_training_progress():
    """
    Monitor training progress by checking saved checkpoints
    """
    print("📊 EEG TRAINING MONITOR - 400 EPOCHS")
    print("=" * 50)
    
    checkpoint_files = [
        'mbd2/eeg_contrastive_400epochs_best.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_100.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_200.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_300.pth',
        'mbd2/eeg_contrastive_400epochs_final.pth',
        'mbd2/advanced_eeg_model_best.pth',
        'mbd2/advanced_eeg_checkpoint_50.pth',
        'mbd2/advanced_eeg_checkpoint_100.pth'
    ]
    
    print("🔍 Checking for training checkpoints...")
    
    # Check for different model types
    best_model_paths = [
        'mbd2/advanced_eeg_model_best.pth',
        'mbd2/eeg_contrastive_400epochs_best.pth'
    ]

    best_model_path = None
    for path in best_model_paths:
        if os.path.exists(path):
            best_model_path = path
            break
    
    if best_model_path and os.path.exists(best_model_path):
        print(f"✅ Found best model checkpoint: {best_model_path}")

        # Load and display current best results
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        print(f"\n📈 CURRENT BEST RESULTS:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Best Validation Accuracy: {checkpoint['best_val_accuracy']:.3f} ({checkpoint['best_val_accuracy']*100:.1f}%)")

        # Handle different checkpoint formats
        if 'train_accuracy' in checkpoint:
            print(f"   Train Accuracy: {checkpoint['train_accuracy']:.3f}")
        if 'val_loss' in checkpoint:
            print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")

        if 'training_time_hours' in checkpoint:
            print(f"   Training Time: {checkpoint['training_time_hours']:.1f} hours")
        
        # Check if training is complete
        if 'total_epochs' in checkpoint:
            progress = (checkpoint['epoch'] / checkpoint['total_epochs']) * 100
            print(f"   Progress: {progress:.1f}% ({checkpoint['epoch']}/{checkpoint['total_epochs']} epochs)")
        
    else:
        print("❌ No best model checkpoint found")
        print("   Training may not have started yet or no improvement found")
    
    # Check for other checkpoints
    print(f"\n📁 Checkpoint Status:")
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            size_mb = os.path.getsize(checkpoint_file) / 1024 / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_file))
            print(f"   ✅ {checkpoint_file} ({size_mb:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {checkpoint_file} - Not found")

def plot_training_progress():
    """
    Plot current training progress from available checkpoints
    """
    print(f"\n📊 PLOTTING TRAINING PROGRESS")
    print("=" * 50)
    
    # Try to load the most recent checkpoint with training history
    checkpoint_files = [
        'mbd2/advanced_eeg_model_best.pth',
        'mbd2/eeg_contrastive_400epochs_final.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_300.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_200.pth',
        'mbd2/eeg_contrastive_400epochs_checkpoint_100.pth'
    ]
    
    checkpoint_data = None
    loaded_file = None
    
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            try:
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                loaded_file = checkpoint_file
                print(f"✅ Loaded training history from: {checkpoint_file}")
                break
            except Exception as e:
                print(f"❌ Failed to load {checkpoint_file}: {e}")
    
    if checkpoint_data is None:
        print("❌ No checkpoint with training history found")
        return
    
    # Extract training metrics
    train_losses = checkpoint_data.get('train_losses', [])
    train_accuracies = checkpoint_data.get('train_accuracies', [])
    val_losses = checkpoint_data.get('val_losses', [])
    val_accuracies = checkpoint_data.get('val_accuracies', [])
    learning_rates = checkpoint_data.get('learning_rates', [])
    
    if not train_losses:
        print("❌ No training history found in checkpoint")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    print(f"📈 Plotting {len(train_losses)} epochs of training data...")
    
    # Create comprehensive training plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'EEG Training Progress - 400 Epochs\nLoaded from: {loaded_file}', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', alpha=0.7, color='blue')
    axes[0, 0].plot(epochs, val_losses, label='Validation Loss', alpha=0.7, color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_accuracies, label='Train Accuracy', alpha=0.7, color='blue')
    axes[0, 1].plot(epochs, val_accuracies, label='Validation Accuracy', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if learning_rates:
        axes[1, 0].plot(epochs, learning_rates, label='Learning Rate', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule')
    
    # Performance summary
    if val_accuracies:
        best_val_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_val_acc) + 1
        current_val_acc = val_accuracies[-1]
        
        summary_text = f"""Training Summary:
        
Current Epoch: {len(epochs)}
Best Val Accuracy: {best_val_acc:.3f} (Epoch {best_epoch})
Current Val Accuracy: {current_val_acc:.3f}
Current Train Accuracy: {train_accuracies[-1]:.3f}

Improvement: {((current_val_acc - val_accuracies[0]) * 100):.1f}%
Best Improvement: {((best_val_acc - val_accuracies[0]) * 100):.1f}%"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'mbd2/training_progress_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Training progress plot saved as: {plot_filename}")
    
    # Print summary statistics
    if val_accuracies:
        print(f"\n📈 TRAINING STATISTICS:")
        print(f"   Epochs completed: {len(epochs)}")
        print(f"   Best validation accuracy: {max(val_accuracies):.3f} (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")
        print(f"   Current validation accuracy: {val_accuracies[-1]:.3f}")
        print(f"   Accuracy improvement: {((val_accuracies[-1] - val_accuracies[0]) * 100):.1f}%")
        print(f"   Current train/val gap: {(train_accuracies[-1] - val_accuracies[-1]):.3f}")

def estimate_completion_time():
    """
    Estimate training completion time based on current progress
    """
    print(f"\n⏱️  TRAINING TIME ESTIMATION")
    print("=" * 50)
    
    # Check for different model types
    best_model_paths = [
        'mbd2/advanced_eeg_model_best.pth',
        'mbd2/eeg_contrastive_400epochs_best.pth'
    ]

    best_model_path = None
    for path in best_model_paths:
        if os.path.exists(path):
            best_model_path = path
            break

    if not best_model_path:
        print("❌ No checkpoint found for time estimation")
        return
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    
    current_epoch = checkpoint['epoch']
    total_epochs = checkpoint.get('total_epochs', 400)
    training_time_hours = checkpoint.get('training_time_hours', 0)
    
    if training_time_hours > 0 and current_epoch > 0:
        time_per_epoch = training_time_hours / current_epoch
        remaining_epochs = total_epochs - current_epoch
        estimated_remaining_hours = remaining_epochs * time_per_epoch
        
        print(f"📊 Time Analysis:")
        print(f"   Current epoch: {current_epoch}/{total_epochs}")
        print(f"   Time elapsed: {training_time_hours:.1f} hours")
        print(f"   Time per epoch: {time_per_epoch*60:.1f} minutes")
        print(f"   Estimated remaining: {estimated_remaining_hours:.1f} hours")
        
        if estimated_remaining_hours > 0:
            completion_time = datetime.now() + timedelta(hours=estimated_remaining_hours)
            print(f"   Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ Insufficient data for time estimation")

def main():
    """
    Main monitoring function
    """
    print("🔍 EEG TRAINING MONITOR")
    print("=" * 50)
    
    # Monitor current progress
    monitor_training_progress()
    
    # Plot training curves
    plot_training_progress()
    
    # Estimate completion time
    estimate_completion_time()
    
    print(f"\n✅ Monitoring complete!")

if __name__ == "__main__":
    main()
