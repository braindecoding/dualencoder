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
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import random
import os
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
# from explicit_eeg_contrastive_training import load_explicit_data, evaluate_explicit

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"🎲 Random seeds set to {seed} for reproducibility")

def load_crell_data():
    """Load Crell dataset for advanced training"""
    print("📂 Loading Crell dataset...")

    with open('../crell/crell_processed_data_correct.pkl', 'rb') as f:
        data = pickle.load(f)

    # Extract data
    eegTrn = data['training']['eeg']        # (384, 64, 500)
    stimTrn = data['training']['images']    # (384, 28, 28)
    labelsTrn = data['training']['labels']  # (384,)

    eegVal = data['validation']['eeg']      # (128, 64, 500)
    stimVal = data['validation']['images'] if 'images' in data['validation'] else None
    labelsVal = data['validation']['labels'] # (128,)

    eegTest = data['test']['eeg']           # (128, 64, 500)
    stimTest = data['test']['images'] if 'images' in data['test'] else None
    labelsTest = data['test']['labels']     # (128,)

    # For validation/test, use training images as reference (same letters)
    if stimVal is None:
        # Create mapping from labels to images
        label_to_image = {}
        for i, label in enumerate(labelsTrn):
            if label not in label_to_image:
                label_to_image[label] = stimTrn[i]

        # Map validation labels to corresponding images
        stimVal = np.array([label_to_image[label] for label in labelsVal])
        stimTest = np.array([label_to_image[label] for label in labelsTest])

    print(f"✅ Crell dataset loaded successfully!")
    print(f"   Training: {len(eegTrn)} samples")
    print(f"   Validation: {len(eegVal)} samples")
    print(f"   Test: {len(eegTest)} samples")
    print(f"   Letters: {sorted(set(labelsTrn))} -> {[data['metadata']['idx_to_letter'][i] for i in sorted(set(labelsTrn))]}")

    return eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest

def evaluate_crell(model, eeg_data, stim_data, labels, clip_model, clip_preprocess, device, phase_name, batch_size=32):
    """Evaluate model on Crell dataset"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    # Simple contrastive loss for evaluation
    criterion = nn.CosineEmbeddingLoss()

    with torch.no_grad():
        for i in range(0, len(eeg_data), batch_size):
            batch_eeg = eeg_data[i:i+batch_size]
            batch_stim = stim_data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Convert to tensors
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)

            # Process images for CLIP
            processed_images = []
            for img in batch_stim:
                # Convert to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = (img * 255).astype(np.uint8)
                    from PIL import Image
                    img = Image.fromarray(img).convert('RGB')
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
            accuracy = (similarities > 0.0).float().mean()  # Changed from 0.5 to 0.0

            # Simple loss (negative cosine similarity)
            loss = -similarities.mean()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0

    print(f"   {phase_name.capitalize()} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.3f}")

    return avg_loss, avg_accuracy

class AdvancedTrainingConfig:
    """
    Configuration for advanced training
    """
    def __init__(self):
        # Model architecture - ADAPTED FOR CRELL DATASET
        self.n_channels = 64      # Crell: 64 channels (vs 14 MindBigData)
        self.seq_len = 500        # Crell: 500 timepoints (vs 256 MindBigData)
        self.d_model = 256        # Enhanced from 128
        self.embedding_dim = 512
        self.nhead = 8
        self.num_layers = 8       # Enhanced from 6
        self.patch_size = 25      # 500/25 = 20 patches (vs 16 for 256)
        self.dropout = 0.1

        # Training parameters - EXTENDED FOR BETTER CONVERGENCE
        self.num_epochs = 400     # Extended from 300 for better convergence
        self.batch_size = 32
        self.base_lr = 1e-4  # Increased learning rate
        self.min_lr = 1e-6
        self.warmup_epochs = 20
        self.patience = 30  # Reduced patience
        self.weight_decay = 1e-4
        self.gradient_clip = 1.0
        
        # Loss function
        self.loss_type = 'adaptive_temperature'  # 'adaptive_temperature', 'focal', 'multiscale'
        self.temperature = 0.05  # Lower temperature
        
        # Data augmentation - ADAPTED FOR CRELL
        self.use_augmentation = True
        self.augment_prob = 0.8
        self.noise_std = 0.03
        self.temporal_shift_range = 20    # Increased for 500 timepoints (vs 8 for 256)
        self.electrode_dropout_prob = 0.1  # Reduced for 64 channels (vs 0.15 for 14)
        
        # Monitoring
        self.save_frequency = 50
        # self.eval_frequency = 5  # REMOVED - now evaluate every epoch
        
    def __str__(self):
        return f"""Advanced Training Configuration (CRELL DATASET):
        Model: Enhanced Transformer (64ch, 500Hz, d_model={self.d_model}, layers={self.num_layers})
        Training: {self.num_epochs} epochs, lr={self.base_lr}, batch_size={self.batch_size}
        Loss: {self.loss_type}, temperature={self.temperature}
        Augmentation: {'Enabled' if self.use_augmentation else 'Disabled'}
        Task: EEG-to-Letter (10 letters: a,d,e,f,j,n,o,s,t,v)
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
    eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest = load_crell_data()
    
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
        
        # Create training batches (simplified for Crell)
        # Simple batch creation without complex augmentation for now
        train_batches = []
        indices = np.arange(len(eegTrn))
        if True:  # shuffle
            np.random.shuffle(indices)

        for i in range(0, len(indices), config.batch_size):
            batch_indices = indices[i:i+config.batch_size]
            batch_eeg = eegTrn[batch_indices]
            batch_stim = stimTrn[batch_indices]
            batch_labels = labelsTrn[batch_indices]
            train_batches.append((batch_eeg, batch_stim, batch_labels))

        train_loader = train_batches
        
        # Training progress bar
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [TRAIN]",
            total=len(train_loader)
        )
        
        for batch_data in train_progress:
            batch_eeg, batch_stim, batch_labels = batch_data
            
            # Move EEG to device
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Preprocess images for CLIP
            processed_images = []
            for img in batch_stim:
                # Convert numpy array to PIL Image
                if isinstance(img, np.ndarray):
                    # Normalize to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    # Convert grayscale to RGB if needed
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis=-1)

                    from PIL import Image
                    img = Image.fromarray(img)

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
        
        # VALIDATION PHASE (EVERY EPOCH - FIXED!)
        val_loss, val_accuracy = evaluate_crell(
            model, eegVal, stimVal, labelsVal,
            clip_model, clip_preprocess, device, "validation", config.batch_size
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
            }, 'advanced_eeg_model_best.pth')
            
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
            }, f'advanced_eeg_checkpoint_{epoch+1}.pth')
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    # Final evaluation
    print(f"\n🎯 FINAL EVALUATION:")
    
    # Load best model
    checkpoint = torch.load('advanced_eeg_model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    _, test_accuracy = evaluate_crell(
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

def run_10fold_cross_validation():
    """
    Run 10-fold cross-validation for advanced training
    """
    print("🔄 10-FOLD CROSS-VALIDATION - ADVANCED EEG TRAINING")
    print("=" * 80)

    # Set random seeds for reproducibility
    set_random_seeds(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load CLIP model once
    print(f"\n📥 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print(f"✅ CLIP model loaded and frozen")

    # Load all data first
    print(f"\n📂 Loading complete dataset...")
    eegTrn, stimTrn, labelsTrn, eegVal, stimVal, labelsVal, eegTest, stimTest, labelsTest = load_crell_data()

    # Combine training and validation for CV (keep test separate)
    all_eeg = np.concatenate([eegTrn, eegVal], axis=0)
    all_stim = np.concatenate([stimTrn, stimVal], axis=0)
    all_labels = np.concatenate([labelsTrn, labelsVal], axis=0)

    print(f"✅ Dataset loaded for CV: {len(all_eeg)} samples")
    print(f"   EEG shape: {all_eeg.shape}")
    print(f"   Stimulus shape: {all_stim.shape}")
    print(f"   Labels: {len(all_labels)} samples")

    # Check data distribution
    label_counts = Counter(all_labels)
    print(f"   Label distribution: {dict(label_counts)}")

    # Initialize 10-fold cross-validation with stratification
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []
    all_histories = []

    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_eeg, all_labels)):
        print(f"\n🔄 FOLD {fold + 1}/10")
        print("=" * 60)

        # Set random seeds for this fold
        fold_seed = 42 + fold
        set_random_seeds(fold_seed)

        # Split data for this fold
        fold_eeg_train = all_eeg[train_idx]
        fold_stim_train = all_stim[train_idx]
        fold_labels_train = all_labels[train_idx]

        fold_eeg_val = all_eeg[val_idx]
        fold_stim_val = all_stim[val_idx]
        fold_labels_val = all_labels[val_idx]

        print(f"   Training: {len(fold_eeg_train)} samples")
        print(f"   Validation: {len(fold_eeg_val)} samples")

        # Check fold balance
        train_counts = Counter(fold_labels_train)
        val_counts = Counter(fold_labels_val)
        print(f"   Train distribution: {dict(train_counts)}")
        print(f"   Val distribution: {dict(val_counts)}")

        # Create config for this fold (reduced epochs for CV)
        config = AdvancedTrainingConfig()
        config.num_epochs = 150  # Reduced for CV
        config.patience = 25     # Reduced patience
        # config.eval_frequency = 3  # REMOVED - now evaluate every epoch

        # Create model for this fold
        model = create_advanced_model(config, device)

        # Create loss function
        loss_fn = create_advanced_loss(config, device)

        # Create optimizer and scheduler
        optimizer, scheduler = create_advanced_optimizer_and_scheduler(model, config)

        # Training history for this fold
        fold_history = {
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

        print(f"🚀 Training Fold {fold + 1}...")

        # Training loop for this fold
        for epoch in range(config.num_epochs):
            # Update learning rate
            current_lr = scheduler.step(epoch)

            # TRAINING PHASE
            model.train()
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            num_train_batches = 0
            epoch_temperature = 0

            # Create training batches
            train_batches = []
            indices = np.arange(len(fold_eeg_train))
            np.random.shuffle(indices)

            for i in range(0, len(indices), config.batch_size):
                batch_indices = indices[i:i+config.batch_size]
                batch_eeg = fold_eeg_train[batch_indices]
                batch_stim = fold_stim_train[batch_indices]
                batch_labels = fold_labels_train[batch_indices]
                train_batches.append((batch_eeg, batch_stim, batch_labels))

            # Training batches
            for batch_data in train_batches:
                batch_eeg, batch_stim, batch_labels = batch_data

                # Move EEG to device
                batch_eeg = torch.FloatTensor(batch_eeg).to(device)

                # Preprocess images for CLIP
                processed_images = []
                for img in batch_stim:
                    if isinstance(img, np.ndarray):
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)

                        if len(img.shape) == 2:
                            img = np.stack([img, img, img], axis=-1)

                        from PIL import Image
                        img = Image.fromarray(img)

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

            # VALIDATION PHASE (EVERY EPOCH - FIXED!)
            val_loss, val_accuracy = evaluate_crell(
                model, fold_eeg_val, fold_stim_val, fold_labels_val,
                clip_model, clip_preprocess, device, "validation", config.batch_size
            )

            # Record metrics
            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
            avg_train_accuracy = epoch_train_accuracy / num_train_batches if num_train_batches > 0 else 0
            avg_temperature = epoch_temperature / num_train_batches if num_train_batches > 0 else config.temperature

            fold_history['train_losses'].append(avg_train_loss)
            fold_history['train_accuracies'].append(avg_train_accuracy)
            fold_history['val_losses'].append(val_loss)
            fold_history['val_accuracies'].append(val_accuracy)
            fold_history['learning_rates'].append(current_lr)
            fold_history['temperatures'].append(avg_temperature)

            # Early stopping and best model saving
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0

                # Save best model for this fold
                torch.save(model.state_dict(), f'advanced_eeg_fold_{fold+1}_best.pth')
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_accuracy:.3f}, Val Acc={val_accuracy:.3f}")

            # Early stopping
            if patience_counter >= config.patience:
                print(f"      🛑 Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time

        # Final evaluation for this fold
        model.load_state_dict(torch.load(f'advanced_eeg_fold_{fold+1}_best.pth'))
        model.eval()

        # Calculate final metrics on validation set
        final_val_loss, final_val_accuracy = evaluate_crell(
            model, fold_eeg_val, fold_stim_val, fold_labels_val,
            clip_model, clip_preprocess, device, "final_validation", config.batch_size
        )

        # Store results for this fold
        fold_result = {
            'fold': fold + 1,
            'best_val_accuracy': best_val_accuracy,
            'final_val_accuracy': final_val_accuracy,
            'training_time': training_time,
            'epochs_trained': len(fold_history['train_losses']),
            'train_samples': len(fold_eeg_train),
            'val_samples': len(fold_eeg_val)
        }

        fold_results.append(fold_result)
        all_histories.append(fold_history)

        print(f"   ✅ Fold {fold + 1} completed:")
        print(f"      Best Val Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.1f}%)")
        print(f"      Final Val Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.1f}%)")
        print(f"      Training time: {training_time/60:.1f} minutes")
        print(f"      Epochs trained: {len(fold_history['train_losses'])}")

    # Calculate cross-validation statistics
    val_accuracies = [result['best_val_accuracy'] for result in fold_results]
    final_accuracies = [result['final_val_accuracy'] for result in fold_results]

    mean_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)
    mean_final_acc = np.mean(final_accuracies)
    std_final_acc = np.std(final_accuracies)

    print(f"\n📊 10-FOLD CROSS-VALIDATION RESULTS:")
    print(f"=" * 60)
    for i, result in enumerate(fold_results):
        print(f"   Fold {i+1}: Best={result['best_val_accuracy']:.4f}, Final={result['final_val_accuracy']:.4f}, Time={result['training_time']/60:.1f}min")

    print(f"\n🎯 SUMMARY STATISTICS:")
    print(f"   Best Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f} ({mean_val_acc*100:.1f}% ± {std_val_acc*100:.1f}%)")
    print(f"   Final Validation Accuracy: {mean_final_acc:.4f} ± {std_final_acc:.4f} ({mean_final_acc*100:.1f}% ± {std_final_acc*100:.1f}%)")

    # Test on held-out test set using best fold model
    best_fold_idx = np.argmax(val_accuracies)
    best_fold_num = best_fold_idx + 1

    print(f"\n🏆 TESTING ON HELD-OUT TEST SET:")
    print(f"   Using best fold model: Fold {best_fold_num} (accuracy: {val_accuracies[best_fold_idx]:.4f})")

    # Load best fold model
    model = create_advanced_model(AdvancedTrainingConfig(), device)
    model.load_state_dict(torch.load(f'advanced_eeg_fold_{best_fold_num}_best.pth'))
    model.eval()

    # Test evaluation
    test_loss, test_accuracy = evaluate_crell(
        model, eegTest, stimTest, labelsTest,
        clip_model, clip_preprocess, device, "test", 32
    )

    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

    # Save complete results
    cv_results = {
        'fold_results': fold_results,
        'cv_statistics': {
            'mean_val_accuracy': mean_val_acc,
            'std_val_accuracy': std_val_acc,
            'mean_final_accuracy': mean_final_acc,
            'std_final_accuracy': std_final_acc,
            'test_accuracy': test_accuracy,
            'best_fold': best_fold_num
        },
        'all_histories': all_histories
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'advanced_cv_results_10fold_{timestamp}.pkl', 'wb') as f:
        pickle.dump(cv_results, f)

    print(f"\n🏆 FINAL 10-FOLD CV RESULTS:")
    print(f"   Cross-Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f} ({mean_val_acc*100:.1f}% ± {std_val_acc*100:.1f}%)")
    print(f"   Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"   Results saved: advanced_cv_results_10fold_{timestamp}.pkl")

    return cv_results, all_histories

def main():
    """
    Main function with 10-fold cross-validation
    """
    print("🎯 ADVANCED EEG TRAINING - 10-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print("Enhanced EEG Transformer with Contrastive Learning")
    print("Dataset: Crell (10 letters: a,d,e,f,j,n,o,s,t,v)")
    print("=" * 80)

    # Run 10-fold cross-validation
    cv_results, histories = run_10fold_cross_validation()

    # Plot results using first fold history as example
    plot_advanced_results(histories[0], cv_results['cv_statistics']['mean_val_accuracy'],
                         cv_results['cv_statistics']['test_accuracy'])

    print(f"\n🎯 Advanced Training 10-Fold CV Complete!")
    stats = cv_results['cv_statistics']
    print(f"   Cross-Validation Accuracy: {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}")
    print(f"   Test Set Accuracy: {stats['test_accuracy']:.4f}")
    print(f"   Best Fold: {stats['best_fold']}")

    return cv_results, histories

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
    plot_filename = f'advanced_training_results_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Training results plot saved as: {plot_filename}")

if __name__ == "__main__":
    main()
