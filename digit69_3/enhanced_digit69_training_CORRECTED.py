#!/usr/bin/env python3
"""
CORRECTED Enhanced Digit69 Training with REAL LABELS
Uses actual labels from original dataset, not artificial cycling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import clip
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class Digit69Dataset(Dataset):
    """Dataset for Digit69 with REAL labels"""
    def __init__(self, fmri_data, images, labels, transform=None):
        self.fmri_data = fmri_data
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = torch.FloatTensor(self.fmri_data[idx])
        
        # Handle image processing - reshape from 784 to 28x28
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            # Reshape from 784 to 28x28
            if len(image.shape) == 1 and image.shape[0] == 784:
                image = image.reshape(28, 28)
            
            # Convert to RGB
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            
            # Ensure values are in [0, 255] range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image).convert('RGB')
        
        # Apply CLIP preprocessing
        if self.transform:
            image = self.transform(image)
        else:
            # Default CLIP preprocessing
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])(image)
        
        label = self.labels[idx]
        return fmri, image, label

def load_digit69_data_CORRECTED():
    """Load Digit69 dataset with REAL labels"""
    print("📂 Loading Digit69 dataset with REAL labels...")
    
    try:
        # Load original .mat file
        mat_data = sio.loadmat('../dataset/digit69_28x28.mat')
        
        # Extract data with REAL labels
        fmriTrn = mat_data['fmriTrn'].astype(np.float32)
        stimTrn = mat_data['stimTrn'].astype(np.float32)
        labelsTrn = mat_data['labelTrn'].flatten().astype(np.int64)  # ✅ REAL LABELS
        
        fmriTest = mat_data['fmriTest'].astype(np.float32)
        stimTest = mat_data['stimTest'].astype(np.float32)
        labelsTest = mat_data['labelTest'].flatten().astype(np.int64)  # ✅ REAL LABELS
        
        print(f"✅ Digit69 dataset loaded with REAL labels!")
        print(f"   Training: {len(fmriTrn)} samples")
        print(f"   Test: {len(fmriTest)} samples")
        print(f"   fMRI shape: {fmriTrn.shape}")
        print(f"   Stimulus shape: {stimTrn.shape}")
        print(f"   Training labels: {np.unique(labelsTrn)} (counts: {np.bincount(labelsTrn)})")
        print(f"   Test labels: {np.unique(labelsTest)} (counts: {np.bincount(labelsTest)})")
        
        return fmriTrn, stimTrn, labelsTrn, fmriTest, stimTest, labelsTest
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

class FMRIEncoder(nn.Module):
    """fMRI Encoder for binary classification (classes 1 and 2)"""
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

def enhanced_digit69_train_CORRECTED(fmri_encoder, train_loader, test_loader, device, epochs=100, lr=1e-3):
    """Enhanced training with REAL labels and binary classification"""
    print("🔢 Starting CORRECTED Digit69 training with REAL labels...")
    print(f"📊 Dataset: fMRI → Binary classification (classes 1 and 2)")
    print(f"🎯 Task: Contrastive learning between fMRI and CLIP embeddings")
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(fmri_encoder.parameters(), lr=lr, weight_decay=1e-4)
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
        'train_class_acc': [],  # Binary classification accuracy
        'val_class_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience = 25
    patience_counter = 0
    
    print(f"🚀 Training Configuration:")
    print(f"   Task: Binary classification (2 classes)")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Patience: {patience}")
    
    for epoch in range(epochs):
        # TRAINING PHASE
        fmri_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_sim = 0.0
        train_batches = 0
        
        train_fmri_embs = []
        train_clip_embs = []
        train_labels = []
        
        for batch_idx, (fmri, images, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            fmri = fmri.to(device)
            images = images.to(device)
            
            # Forward pass
            fmri_emb = fmri_encoder(fmri)
            
            with torch.no_grad():
                clip_emb = clip_model.encode_image(images)
                clip_emb = F.normalize(clip_emb.float(), dim=-1)
            
            # Compute loss (contrastive)
            target = torch.ones(fmri.size(0)).to(device)  # All positive pairs
            loss = criterion(fmri_emb, clip_emb, target)
            
            # Compute similarity
            similarities = F.cosine_similarity(fmri_emb, clip_emb, dim=1)
            avg_similarity = similarities.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fmri_encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_sim += avg_similarity.item()
            train_batches += 1
            
            # Store embeddings for retrieval metrics
            train_fmri_embs.append(fmri_emb.detach().cpu())
            train_clip_embs.append(clip_emb.detach().cpu())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_fmri_all = torch.cat(train_fmri_embs, dim=0)
        train_clip_all = torch.cat(train_clip_embs, dim=0)
        train_sim_matrix = torch.matmul(train_fmri_all, train_clip_all.T)
        
        train_top1_acc = calculate_top_k_accuracy(train_sim_matrix, k=1)
        train_class_acc = calculate_binary_class_accuracy(train_sim_matrix, train_labels)
        
        # VALIDATION PHASE
        fmri_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_sim = 0.0
        val_batches = 0
        
        val_fmri_embs = []
        val_clip_embs = []
        val_labels = []
        
        with torch.no_grad():
            for fmri, images, labels in tqdm(test_loader, desc=f"Val Epoch {epoch+1}"):
                fmri = fmri.to(device)
                images = images.to(device)
                
                # Forward pass
                fmri_emb = fmri_encoder(fmri)
                clip_emb = clip_model.encode_image(images)
                clip_emb = F.normalize(clip_emb.float(), dim=-1)
                
                # Compute loss and similarity
                target = torch.ones(fmri.size(0)).to(device)
                loss = criterion(fmri_emb, clip_emb, target)
                similarities = F.cosine_similarity(fmri_emb, clip_emb, dim=1)
                avg_similarity = similarities.mean()
                
                epoch_val_loss += loss.item()
                epoch_val_sim += avg_similarity.item()
                val_batches += 1
                
                # Store embeddings for retrieval metrics
                val_fmri_embs.append(fmri_emb.cpu())
                val_clip_embs.append(clip_emb.cpu())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_fmri_all = torch.cat(val_fmri_embs, dim=0)
        val_clip_all = torch.cat(val_clip_embs, dim=0)
        val_sim_matrix = torch.matmul(val_fmri_all, val_clip_all.T)
        
        val_top1_acc = calculate_top_k_accuracy(val_sim_matrix, k=1)
        val_class_acc = calculate_binary_class_accuracy(val_sim_matrix, val_labels)
        
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
        history['train_class_acc'].append(train_class_acc)
        history['val_class_acc'].append(val_class_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"           Train Sim={avg_train_sim:.3f}, Val Sim={avg_val_sim:.3f}")
            print(f"           Train Top1={train_top1_acc:.3f}, Val Top1={val_top1_acc:.3f}")
            print(f"           Train Class={train_class_acc:.3f}, Val Class={val_class_acc:.3f}, LR={current_lr:.6f}")
        
        # Early stopping based on validation class accuracy
        if val_class_acc > best_val_acc:
            best_val_acc = val_class_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'fmri_encoder_state_dict': fmri_encoder.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'history': history
            }, 'digit69_CORRECTED_best.pth')
            print(f"   💾 New best model saved! Val Class Acc: {val_class_acc:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation class accuracy: {best_val_acc:.3f}")
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

def calculate_binary_class_accuracy(similarity_matrix, labels):
    """Calculate binary classification accuracy"""
    n_samples = similarity_matrix.shape[0]
    _, top_1_indices = torch.topk(similarity_matrix, 1, dim=1)
    
    correct_class = 0
    for i in range(n_samples):
        retrieved_idx = top_1_indices[i][0].item()
        if labels[i] == labels[retrieved_idx]:
            correct_class += 1
    
    accuracy = correct_class / n_samples
    return accuracy

def main():
    """Main function"""
    print("🧠 CORRECTED DIGIT69 TRAINING WITH REAL LABELS")
    print("=" * 70)
    print("Dataset: fMRI → Binary classification (classes 1 and 2)")
    print("Task: Contrastive learning with REAL labels")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    try:
        # Load dataset with REAL labels
        fmriTrn, stimTrn, labelsTrn, fmriTest, stimTest, labelsTest = load_digit69_data_CORRECTED()
        
        # Create CLIP preprocessing
        clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Create datasets
        train_dataset = Digit69Dataset(fmriTrn, stimTrn, labelsTrn, transform=clip_preprocess)
        test_dataset = Digit69Dataset(fmriTest, stimTest, labelsTest, transform=clip_preprocess)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"📊 Dataset prepared:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Task: Binary classification (2 classes)")
        
        # Create fMRI encoder
        fmri_encoder = FMRIEncoder(input_dim=3092, embedding_dim=512).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in fmri_encoder.parameters())
        print(f"🤖 fMRI Encoder created: {total_params:,} parameters")
        
        # Run corrected training
        history = enhanced_digit69_train_CORRECTED(
            fmri_encoder, train_loader, test_loader, device,
            epochs=100, lr=1e-3
        )
        
        # Print final performance
        best_val_acc = max(history['val_class_acc'])
        print(f"\n🏆 CORRECTED FINAL PERFORMANCE:")
        print(f"   Best Validation Class Accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
        print(f"   Random Baseline (2 classes): 0.500 (50.0%)")
        
        if best_val_acc > 0.5:
            improvement = (best_val_acc - 0.5) / 0.5 * 100
            print(f"   Improvement over random: +{improvement:.1f}% 🎉")
        else:
            decline = (0.5 - best_val_acc) / 0.5 * 100
            print(f"   Performance vs random: -{decline:.1f}% ⚠️")
        
        return fmri_encoder, history
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    main()
