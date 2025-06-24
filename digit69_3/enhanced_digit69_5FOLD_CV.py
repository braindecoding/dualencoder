#!/usr/bin/env python3
"""
Enhanced Digit69 Brain Reconstruction with 5-Fold Cross Validation
Comprehensive evaluation with statistical significance testing
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
from sklearn.model_selection import KFold
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

class BrainImageDataset(Dataset):
    """Dataset for brain-to-image reconstruction"""
    def __init__(self, brain_data, images, transform=None):
        self.brain_data = brain_data
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.brain_data)
    
    def __getitem__(self, idx):
        brain = torch.FloatTensor(self.brain_data[idx])
        
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
        
        return brain, image

def load_brain_image_data():
    """Load brain-image pairs for 5-fold CV"""
    print("📂 Loading brain-image pairs for 5-fold cross validation...")
    
    try:
        # Load original .mat file
        mat_data = sio.loadmat('dataset/digit69_28x28.mat')
        
        # Combine training and test data for CV
        brainTrn = mat_data['fmriTrn'].astype(np.float32)
        imagesTrn = mat_data['stimTrn'].astype(np.float32)
        
        brainTest = mat_data['fmriTest'].astype(np.float32)
        imagesTest = mat_data['stimTest'].astype(np.float32)
        
        # Combine all data for cross validation
        all_brain = np.concatenate([brainTrn, brainTest], axis=0)
        all_images = np.concatenate([imagesTrn, imagesTest], axis=0)
        
        print(f"✅ Brain-image dataset loaded for 5-fold CV!")
        print(f"   Total samples: {len(all_brain)}")
        print(f"   Brain signals: {all_brain.shape}")
        print(f"   Visual stimuli: {all_images.shape}")
        print(f"   Task: Brain → Image reconstruction via contrastive learning")
        
        return all_brain, all_images
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

class BrainEncoder(nn.Module):
    """Brain encoder for image reconstruction"""
    def __init__(self, input_dim=3092, embedding_dim=512):
        super().__init__()
        
        # Multi-layer MLP for brain encoding
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

def train_single_fold(brain_encoder, train_loader, val_loader, device, epochs=50, lr=1e-3, fold_num=1):
    """Train single fold"""
    print(f"🔄 Training Fold {fold_num}...")
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(brain_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_similarity = 0.0
    patience = 15
    patience_counter = 0
    
    fold_history = {
        'train_losses': [],
        'val_losses': [],
        'train_similarities': [],
        'val_similarities': [],
        'val_retrieval_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        # TRAINING PHASE
        brain_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_sim = 0.0
        train_batches = 0
        
        for brain, images in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1}", leave=False):
            brain = brain.to(device)
            images = images.to(device)
            
            # Forward pass
            brain_emb = brain_encoder(brain)
            
            with torch.no_grad():
                image_emb = clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
            
            # Compute contrastive loss
            similarities = F.cosine_similarity(brain_emb, image_emb, dim=1)
            loss = -similarities.mean()  # Contrastive loss
            
            avg_similarity = similarities.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain_encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_sim += avg_similarity.item()
            train_batches += 1
        
        # VALIDATION PHASE
        brain_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_sim = 0.0
        val_batches = 0
        
        val_brain_embs = []
        val_image_embs = []
        
        with torch.no_grad():
            for brain, images in val_loader:
                brain = brain.to(device)
                images = images.to(device)
                
                # Forward pass
                brain_emb = brain_encoder(brain)
                image_emb = clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
                
                # Compute loss and similarity
                similarities = F.cosine_similarity(brain_emb, image_emb, dim=1)
                loss = -similarities.mean()
                avg_similarity = similarities.mean()
                
                epoch_val_loss += loss.item()
                epoch_val_sim += avg_similarity.item()
                val_batches += 1
                
                # Store embeddings for retrieval evaluation
                val_brain_embs.append(brain_emb.cpu())
                val_image_embs.append(image_emb.cpu())
        
        # Calculate validation retrieval accuracy
        val_brain_all = torch.cat(val_brain_embs, dim=0)
        val_image_all = torch.cat(val_image_embs, dim=0)
        val_sim_matrix = torch.matmul(val_brain_all, val_image_all.T)
        val_retrieval_acc = calculate_retrieval_accuracy(val_sim_matrix, k=1)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate averages
        avg_train_loss = epoch_train_loss / train_batches
        avg_val_loss = epoch_val_loss / val_batches
        avg_train_sim = epoch_train_sim / train_batches
        avg_val_sim = epoch_val_sim / val_batches
        
        # Store metrics
        fold_history['train_losses'].append(avg_train_loss)
        fold_history['val_losses'].append(avg_val_loss)
        fold_history['train_similarities'].append(avg_train_sim)
        fold_history['val_similarities'].append(avg_val_sim)
        fold_history['val_retrieval_acc'].append(val_retrieval_acc)
        fold_history['learning_rates'].append(current_lr)
        
        # Early stopping
        if avg_val_sim > best_val_similarity:
            best_val_similarity = avg_val_sim
            patience_counter = 0
            best_model_state = brain_encoder.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    brain_encoder.load_state_dict(best_model_state)
    
    # Final evaluation
    brain_encoder.eval()
    final_val_sim = 0.0
    final_val_retrieval = 0.0
    
    val_brain_embs = []
    val_image_embs = []
    
    with torch.no_grad():
        for brain, images in val_loader:
            brain = brain.to(device)
            images = images.to(device)
            
            brain_emb = brain_encoder(brain)
            image_emb = clip_model.encode_image(images)
            image_emb = F.normalize(image_emb.float(), dim=-1)
            
            similarities = F.cosine_similarity(brain_emb, image_emb, dim=1)
            final_val_sim += similarities.mean().item()
            
            val_brain_embs.append(brain_emb.cpu())
            val_image_embs.append(image_emb.cpu())
    
    final_val_sim /= len(val_loader)
    
    # Calculate final retrieval accuracy
    val_brain_all = torch.cat(val_brain_embs, dim=0)
    val_image_all = torch.cat(val_image_embs, dim=0)
    val_sim_matrix = torch.matmul(val_brain_all, val_image_all.T)
    final_val_retrieval = calculate_retrieval_accuracy(val_sim_matrix, k=1)
    
    print(f"   ✅ Fold {fold_num} completed: Similarity={final_val_sim:.3f}, Retrieval={final_val_retrieval:.3f}")
    
    return {
        'similarity': final_val_sim,
        'retrieval_accuracy': final_val_retrieval,
        'history': fold_history,
        'model_state': best_model_state
    }

def calculate_retrieval_accuracy(similarity_matrix, k=1):
    """Calculate image retrieval accuracy from brain signals"""
    n_samples = similarity_matrix.shape[0]
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    correct_retrieval = 0
    for i in range(n_samples):
        if i in top_k_indices[i]:
            correct_retrieval += 1
    
    accuracy = correct_retrieval / n_samples
    return accuracy

def run_5fold_cross_validation():
    """Run 5-fold cross validation"""
    print("🔄 5-FOLD CROSS VALIDATION FOR BRAIN RECONSTRUCTION")
    print("=" * 70)
    print("Task: Brain signals → Image reconstruction")
    print("Method: Contrastive learning with 5-fold CV")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    all_brain, all_images = load_brain_image_data()
    
    # Create CLIP preprocessing
    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    all_histories = []
    
    print(f"🔄 Starting 5-fold cross validation...")
    print(f"   Total samples: {len(all_brain)}")
    print(f"   Folds: 5")
    print(f"   Training epochs per fold: 50")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_brain)):
        print(f"\n📊 FOLD {fold + 1}/5")
        print(f"   Training samples: {len(train_idx)}")
        print(f"   Validation samples: {len(val_idx)}")
        
        # Create fold datasets
        train_brain = all_brain[train_idx]
        train_images = all_images[train_idx]
        val_brain = all_brain[val_idx]
        val_images = all_images[val_idx]
        
        train_dataset = BrainImageDataset(train_brain, train_images, transform=clip_preprocess)
        val_dataset = BrainImageDataset(val_brain, val_images, transform=clip_preprocess)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create fresh model for each fold
        brain_encoder = BrainEncoder(input_dim=3092, embedding_dim=512).to(device)
        
        # Train fold
        fold_result = train_single_fold(
            brain_encoder, train_loader, val_loader, device,
            epochs=50, lr=1e-3, fold_num=fold+1
        )
        
        fold_results.append(fold_result)
        all_histories.append(fold_result['history'])
    
    # Analyze results
    similarities = [result['similarity'] for result in fold_results]
    retrievals = [result['retrieval_accuracy'] for result in fold_results]
    
    print(f"\n🏆 5-FOLD CROSS VALIDATION RESULTS:")
    print(f"=" * 50)
    
    # Individual fold results
    for i, (sim, ret) in enumerate(zip(similarities, retrievals)):
        print(f"   Fold {i+1}: Similarity={sim:.3f}, Retrieval={ret:.3f} ({ret*100:.1f}%)")
    
    # Statistical summary
    sim_mean, sim_std = np.mean(similarities), np.std(similarities)
    ret_mean, ret_std = np.mean(retrievals), np.std(retrievals)
    
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"   Similarity: {sim_mean:.3f} ± {sim_std:.3f}")
    print(f"   Retrieval:  {ret_mean:.3f} ± {ret_std:.3f} ({ret_mean*100:.1f}% ± {ret_std*100:.1f}%)")
    
    # Random baseline
    random_baseline = 1.0 / len(val_idx)  # 1/validation_size
    improvement = (ret_mean - random_baseline) / random_baseline * 100
    
    print(f"\n🎯 PERFORMANCE ANALYSIS:")
    print(f"   Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)")
    print(f"   Achieved: {ret_mean:.3f} ({ret_mean*100:.1f}%)")
    print(f"   Improvement: +{improvement:.1f}%")
    
    # Statistical significance (t-test against random)
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(retrievals, random_baseline)
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"   ✅ Statistically significant (p < 0.05)")
    else:
        print(f"   ⚠️ Not statistically significant (p >= 0.05)")
    
    # Create comprehensive plots
    plot_5fold_results(fold_results, all_histories)
    
    # Save results
    results_df = pd.DataFrame({
        'Fold': range(1, 6),
        'Similarity': similarities,
        'Retrieval_Accuracy': retrievals
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'brain_reconstruction_5fold_results_{timestamp}.csv', index=False)
    
    print(f"\n📊 Results saved: brain_reconstruction_5fold_results_{timestamp}.csv")
    
    return fold_results, all_histories

def plot_5fold_results(fold_results, all_histories):
    """Plot 5-fold cross validation results"""
    print("📊 Creating 5-fold CV visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('5-Fold Cross Validation - Brain Reconstruction Results', fontsize=16, fontweight='bold')
    
    # Extract data
    similarities = [result['similarity'] for result in fold_results]
    retrievals = [result['retrieval_accuracy'] for result in fold_results]
    
    # Plot 1: Individual fold performance
    folds = range(1, 6)
    bars1 = axes[0, 0].bar([f-0.2 for f in folds], similarities, 0.4, label='Similarity', alpha=0.8, color='blue')
    bars2 = axes[0, 0].bar([f+0.2 for f in folds], retrievals, 0.4, label='Retrieval Acc', alpha=0.8, color='red')
    
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Performance')
    axes[0, 0].set_title('Individual Fold Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(folds)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Statistical summary
    metrics = ['Similarity', 'Retrieval Accuracy']
    means = [np.mean(similarities), np.mean(retrievals)]
    stds = [np.std(similarities), np.std(retrievals)]
    
    bars = axes[0, 1].bar(metrics, means, yerr=stds, capsize=5, alpha=0.8, color=['blue', 'red'])
    axes[0, 1].set_ylabel('Performance')
    axes[0, 1].set_title('Mean Performance ± Std')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[0, 1].annotate(f'{mean:.3f}±{std:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Learning curves (average across folds)
    max_epochs = max(len(hist['val_similarities']) for hist in all_histories)
    
    avg_train_sim = []
    avg_val_sim = []
    
    for epoch in range(max_epochs):
        epoch_train_sims = []
        epoch_val_sims = []
        
        for hist in all_histories:
            if epoch < len(hist['train_similarities']):
                epoch_train_sims.append(hist['train_similarities'][epoch])
                epoch_val_sims.append(hist['val_similarities'][epoch])
        
        if epoch_train_sims:
            avg_train_sim.append(np.mean(epoch_train_sims))
            avg_val_sim.append(np.mean(epoch_val_sims))
    
    epochs = range(1, len(avg_train_sim) + 1)
    axes[0, 2].plot(epochs, avg_train_sim, label='Training Similarity', alpha=0.8, color='blue')
    axes[0, 2].plot(epochs, avg_val_sim, label='Validation Similarity', alpha=0.8, color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Cosine Similarity')
    axes[0, 2].set_title('Average Learning Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Performance distribution
    axes[1, 0].hist(similarities, bins=10, alpha=0.7, color='blue', label='Similarity')
    axes[1, 0].axvline(np.mean(similarities), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
    axes[1, 0].set_xlabel('Similarity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Similarity Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Retrieval accuracy distribution
    axes[1, 1].hist(retrievals, bins=10, alpha=0.7, color='red', label='Retrieval Acc')
    axes[1, 1].axvline(np.mean(retrievals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(retrievals):.3f}')
    random_baseline = 1.0 / 20  # Approximate baseline
    axes[1, 1].axvline(random_baseline, color='gray', linestyle=':', linewidth=2, label=f'Random: {random_baseline:.3f}')
    axes[1, 1].set_xlabel('Retrieval Accuracy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Retrieval Accuracy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    summary_text = f"""5-Fold Cross Validation Summary:

Task: Brain → Image Reconstruction
Method: Contrastive Learning

Results:
  Similarity: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}
  Retrieval:  {np.mean(retrievals):.3f} ± {np.std(retrievals):.3f}
  
Performance:
  Mean Retrieval: {np.mean(retrievals)*100:.1f}%
  Std Retrieval:  {np.std(retrievals)*100:.1f}%
  
Statistical Test:
  t-test vs random baseline
  p-value: {stats.ttest_1samp(retrievals, random_baseline)[1]:.6f}
  
Robustness:
  CV coefficient: {np.std(retrievals)/np.mean(retrievals)*100:.1f}%
  All folds > random: {'Yes' if all(r > random_baseline for r in retrievals) else 'No'}
  
Architecture:
  Brain Encoder: 3092 → 512 dims
  Target: CLIP ViT-B/32 embeddings
  Loss: Contrastive (negative cosine similarity)"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Statistical Summary')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'brain_reconstruction_5fold_cv_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 5-fold CV plot saved: {plot_filename}")

def main():
    """Main function"""
    return run_5fold_cross_validation()

if __name__ == "__main__":
    main()
