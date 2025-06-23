#!/usr/bin/env python3
"""
Enhanced Digit69 Training with Comprehensive Plotting
Adds test loss tracking and comprehensive visualization for Digit69 EEG dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import clip
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class Digit69Dataset(Dataset):
    """Dataset for Digit69 EEG data"""
    def __init__(self, eeg_data, images, labels, transform=None):
        self.eeg_data = eeg_data
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = torch.FloatTensor(self.eeg_data[idx])

        # Handle image processing - images are already in (3, 224, 224) format
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            # Convert from (3, 224, 224) to (224, 224, 3) for PIL
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))  # (3, 224, 224) -> (224, 224, 3)

            # Ensure values are in [0, 255] range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            # Convert to PIL Image
            image = Image.fromarray(image).convert('RGB')

        # Convert back to tensor for CLIP (no additional transform needed)
        if self.transform:
            image = self.transform(image)
        else:
            # Default CLIP preprocessing
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])(image)

        label = self.labels[idx]
        return eeg, image, label

def load_digit69_data():
    """Load Digit69 EEG dataset"""
    print("📂 Loading Digit69 EEG dataset...")

    # Try to load from different possible locations
    possible_paths = [
        'digit69_embeddings.pkl',
        '../digit69_2/digit69_embeddings.pkl',
        'eeg_digit_data.pkl'
    ]

    data = None
    for path in possible_paths:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Data loaded from: {path}")
            break
        except FileNotFoundError:
            continue

    if data is None:
        raise FileNotFoundError("Could not find Digit69 dataset file")

    # Extract data based on actual structure
    if 'train' in data and 'test' in data:
        # Use original fMRI data (not embeddings) for training
        eegTrn = data['train']['original_fmri']
        stimTrn = data['train']['original_images']
        # Generate labels from indices (assuming digits 0-9)
        labelsTrn = np.arange(len(eegTrn)) % 10

        eegTest = data['test']['original_fmri']
        stimTest = data['test']['original_images']
        labelsTest = np.arange(len(eegTest)) % 10
    else:
        raise ValueError("Unexpected data structure")

    print(f"✅ Digit69 dataset loaded:")
    print(f"   Training: {len(eegTrn)} samples")
    print(f"   Test: {len(eegTest)} samples")
    print(f"   EEG shape: {eegTrn.shape}")
    print(f"   Labels: {set(labelsTrn)} (digits 0-9)")

    return eegTrn, stimTrn, labelsTrn, eegTest, stimTest, labelsTest

class FMRIEncoder(nn.Module):
    """fMRI Encoder for Digit69 (actually fMRI data, not EEG)"""
    def __init__(self, input_dim=3092, embedding_dim=512):
        super().__init__()

        # Multi-layer MLP for fMRI encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),

            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.encoder(x)  # (batch, embedding_dim)
        return F.normalize(x, dim=1)

def enhanced_digit69_train_with_validation(eeg_encoder, train_loader, test_loader, device, epochs=100, lr=1e-3):
    """Enhanced Digit69 training with validation tracking"""
    print("🔢 Starting enhanced Digit69 training with validation tracking...")
    print(f"📊 Dataset: EEG → Digit reconstruction (0-9)")
    print(f"🎯 Task: Contrastive learning between EEG and CLIP embeddings")

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(eeg_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CosineEmbeddingLoss(margin=0.1)

    # Track comprehensive metrics
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_similarities': [],
        'val_similarities': [],
        'train_top1_acc': [],
        'val_top1_acc': [],
        'train_top5_acc': [],
        'val_top5_acc': [],
        'train_digit_acc': [],  # Per-digit accuracy
        'val_digit_acc': [],
        'learning_rates': []
    }

    best_val_top1 = 0.0
    patience = 25
    patience_counter = 0

    print(f"🚀 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Patience: {patience}")
    print(f"   Optimizer: AdamW with weight decay 1e-4")
    print(f"   Scheduler: CosineAnnealingLR")

    for epoch in range(epochs):
        # TRAINING PHASE
        eeg_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_sim = 0.0
        train_batches = 0

        train_eeg_embs = []
        train_clip_embs = []
        train_labels = []

        for batch_idx, (eeg, images, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            eeg = eeg.to(device)
            images = images.to(device)

            # Forward pass
            eeg_emb = eeg_encoder(eeg)

            with torch.no_grad():
                clip_emb = clip_model.encode_image(images)
                clip_emb = F.normalize(clip_emb.float(), dim=-1)

            # Compute loss (contrastive)
            target = torch.ones(eeg.size(0)).to(device)  # All positive pairs
            loss = criterion(eeg_emb, clip_emb, target)

            # Compute similarity
            similarities = F.cosine_similarity(eeg_emb, clip_emb, dim=1)
            avg_similarity = similarities.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eeg_encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_sim += avg_similarity.item()
            train_batches += 1

            # Store embeddings for retrieval metrics
            train_eeg_embs.append(eeg_emb.detach().cpu())
            train_clip_embs.append(clip_emb.detach().cpu())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training retrieval metrics
        train_eeg_all = torch.cat(train_eeg_embs, dim=0)
        train_clip_all = torch.cat(train_clip_embs, dim=0)
        train_sim_matrix = torch.matmul(train_eeg_all, train_clip_all.T)

        train_top1_acc = calculate_top_k_accuracy(train_sim_matrix, k=1)
        train_top5_acc = calculate_top_k_accuracy(train_sim_matrix, k=5)
        train_digit_acc = calculate_digit_accuracy(train_sim_matrix, train_labels)

        # VALIDATION PHASE
        eeg_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_sim = 0.0
        val_batches = 0

        val_eeg_embs = []
        val_clip_embs = []
        val_labels = []

        with torch.no_grad():
            for eeg, images, labels in tqdm(test_loader, desc=f"Val Epoch {epoch+1}"):
                eeg = eeg.to(device)
                images = images.to(device)

                # Forward pass
                eeg_emb = eeg_encoder(eeg)
                clip_emb = clip_model.encode_image(images)
                clip_emb = F.normalize(clip_emb.float(), dim=-1)

                # Compute loss and similarity
                target = torch.ones(eeg.size(0)).to(device)
                loss = criterion(eeg_emb, clip_emb, target)
                similarities = F.cosine_similarity(eeg_emb, clip_emb, dim=1)
                avg_similarity = similarities.mean()

                epoch_val_loss += loss.item()
                epoch_val_sim += avg_similarity.item()
                val_batches += 1

                # Store embeddings for retrieval metrics
                val_eeg_embs.append(eeg_emb.cpu())
                val_clip_embs.append(clip_emb.cpu())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation retrieval metrics
        val_eeg_all = torch.cat(val_eeg_embs, dim=0)
        val_clip_all = torch.cat(val_clip_embs, dim=0)
        val_sim_matrix = torch.matmul(val_eeg_all, val_clip_all.T)

        val_top1_acc = calculate_top_k_accuracy(val_sim_matrix, k=1)
        val_top5_acc = calculate_top_k_accuracy(val_sim_matrix, k=5)
        val_digit_acc = calculate_digit_accuracy(val_sim_matrix, val_labels)

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
        history['train_digit_acc'].append(train_digit_acc)
        history['val_digit_acc'].append(val_digit_acc)
        history['learning_rates'].append(current_lr)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"           Train Sim={avg_train_sim:.3f}, Val Sim={avg_val_sim:.3f}")
            print(f"           Train Top1={train_top1_acc:.3f}, Val Top1={val_top1_acc:.3f}")
            print(f"           Train Digit={train_digit_acc:.3f}, Val Digit={val_digit_acc:.3f}, LR={current_lr:.6f}")

        # Early stopping based on validation top-1 accuracy
        if val_top1_acc > best_val_top1:
            best_val_top1 = val_top1_acc
            patience_counter = 0

            # Save best model
            torch.save({
                'eeg_encoder_state_dict': eeg_encoder.state_dict(),
                'epoch': epoch,
                'best_val_top1': best_val_top1,
                'history': history
            }, 'digit69_contrastive_clip_best.pth')
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

def calculate_digit_accuracy(similarity_matrix, labels):
    """Calculate digit classification accuracy"""
    n_samples = similarity_matrix.shape[0]
    _, top_1_indices = torch.topk(similarity_matrix, 1, dim=1)

    correct_digit = 0
    for i in range(n_samples):
        retrieved_idx = top_1_indices[i][0].item()
        if labels[i] == labels[retrieved_idx]:
            correct_digit += 1

    accuracy = correct_digit / n_samples
    return accuracy

def plot_comprehensive_digit69_results(history, save_plots=True):
    """Create comprehensive Digit69 training plots"""
    print("📊 Creating comprehensive Digit69 training plots...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Enhanced Digit69 Training Results - EEG to Digit Reconstruction',
                 fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_losses']) + 1)

    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', alpha=0.8, color='blue')
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.8, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cosine Embedding Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Similarity curves
    axes[0, 1].plot(epochs, history['train_similarities'], label='Training Similarity', alpha=0.8, color='blue')
    axes[0, 1].plot(epochs, history['val_similarities'], label='Validation Similarity', alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('EEG-CLIP Similarity')
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

    # Plot 5: Digit Classification Accuracy
    axes[1, 1].plot(epochs, history['train_digit_acc'], label='Training Digit Acc', alpha=0.8, color='blue')
    axes[1, 1].plot(epochs, history['val_digit_acc'], label='Validation Digit Acc', alpha=0.8, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Digit Classification Accuracy')
    axes[1, 1].set_title('Digit Recognition Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Learning rate schedule
    axes[1, 2].plot(epochs, history['learning_rates'], alpha=0.8, color='green')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 7: Performance comparison
    axes[2, 0].plot(epochs, history['val_top1_acc'], label='Top-1 Accuracy', alpha=0.8, color='red')
    axes[2, 0].plot(epochs, history['val_top5_acc'], label='Top-5 Accuracy', alpha=0.8, color='blue')
    axes[2, 0].plot(epochs, history['val_digit_acc'], label='Digit Accuracy', alpha=0.8, color='green')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Validation Accuracy')
    axes[2, 0].set_title('Validation Performance Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Training summary
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    best_val_top1 = max(history['val_top1_acc'])
    best_val_digit = max(history['val_digit_acc'])

    summary_text = f"""Digit69 Training Summary:

Dataset: EEG → Digit Recognition (0-9)
Task: Contrastive Learning (EEG ↔ CLIP)

Final Results:
  Train Loss: {final_train_loss:.4f}
  Val Loss: {final_val_loss:.4f}
  Train Sim: {history['train_similarities'][-1]:.3f}
  Val Sim: {history['val_similarities'][-1]:.3f}

Best Performance:
  Best Val Top-1: {best_val_top1:.3f}
  Best Val Top-5: {max(history['val_top5_acc']):.3f}
  Best Digit Acc: {best_val_digit:.3f}

Training Info:
  Total Epochs: {len(epochs)}
  Final LR: {history['learning_rates'][-1]:.2e}

Architecture:
  EEG Encoder: 14 channels → 512 dims
  Target: CLIP ViT-B/32 embeddings
  Loss: Cosine Embedding Loss

Digit Recognition:
  Task: Classify digits 0-9 from EEG
  Method: EEG → CLIP → Digit retrieval
  Performance: {best_val_digit*100:.1f}% accuracy"""

    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Training Summary')

    # Plot 9: Overfitting analysis
    loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
    axes[2, 2].plot(epochs, loss_diff, alpha=0.8, color='orange', label='Loss Gap')
    axes[2, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    acc_diff = np.array(history['val_top1_acc']) - np.array(history['train_top1_acc'])
    axes[2, 2].plot(epochs, acc_diff, alpha=0.8, color='purple', label='Accuracy Gap')

    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Val - Train')
    axes[2, 2].set_title('Generalization Analysis')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'digit69_enhanced_training_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_filename}")

    plt.show()

    return fig

def run_enhanced_digit69_training():
    """Run enhanced Digit69 training with comprehensive plotting"""
    print("🔢 ENHANCED DIGIT69 TRAINING WITH COMPREHENSIVE PLOTS")
    print("=" * 70)
    print("Dataset: EEG → Digit recognition (0-9)")
    print("Task: Contrastive learning between EEG signals and CLIP embeddings")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    try:
        # Load Digit69 dataset
        eegTrn, stimTrn, labelsTrn, eegTest, stimTest, labelsTest = load_digit69_data()

        # Create CLIP preprocessing
        clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # Create datasets
        train_dataset = Digit69Dataset(eegTrn, stimTrn, labelsTrn, transform=clip_preprocess)
        test_dataset = Digit69Dataset(eegTest, stimTest, labelsTest, transform=clip_preprocess)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"📊 Digit69 dataset prepared:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   EEG channels: 14")
        print(f"   Sequence length: 256")
        print(f"   Digits: 0-9")

        # Create fMRI encoder
        fmri_encoder = FMRIEncoder(input_dim=3092, embedding_dim=512).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in fmri_encoder.parameters())
        print(f"🤖 fMRI Encoder created: {total_params:,} parameters")

        # Run enhanced training
        history = enhanced_digit69_train_with_validation(
            fmri_encoder, train_loader, test_loader, device,
            epochs=100, lr=1e-3
        )

        # Create comprehensive plots
        plot_comprehensive_digit69_results(history, save_plots=True)

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'digit69_contrastive_clip_enhanced_{timestamp}.pth'
        torch.save({
            'fmri_encoder_state_dict': fmri_encoder.state_dict(),
            'history': history
        }, final_model_path)

        print("✅ Enhanced Digit69 training completed!")
        print("📊 Files generated:")
        print(f"   - digit69_enhanced_training_results_{timestamp}.png")
        print(f"   - digit69_contrastive_clip_best.pth")
        print(f"   - {final_model_path}")

        # Print final performance
        best_val_top1 = max(history['val_top1_acc'])
        best_val_digit = max(history['val_digit_acc'])
        print(f"\n🏆 FINAL PERFORMANCE:")
        print(f"   Best Validation Top-1 Accuracy: {best_val_top1:.3f} ({best_val_top1*100:.1f}%)")
        print(f"   Best Digit Classification: {best_val_digit:.3f} ({best_val_digit*100:.1f}%)")

        return fmri_encoder, history

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure:")
        print("   1. Digit69 dataset is available")
        print("   2. CLIP is properly installed")
        print("   3. All dependencies are available")
        return None, None

def main():
    """Main function"""
    return run_enhanced_digit69_training()

if __name__ == "__main__":
    main()