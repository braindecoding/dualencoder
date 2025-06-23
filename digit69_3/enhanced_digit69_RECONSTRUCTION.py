#!/usr/bin/env python3
"""
Enhanced Digit69 RECONSTRUCTION Training
Pure contrastive learning for brain-to-image reconstruction (NO CLASSIFICATION)
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

class BrainImageDataset(Dataset):
    """Dataset for brain-to-image reconstruction (NO LABELS NEEDED)"""
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
        
        return brain, image  # NO LABELS!

def load_brain_image_data():
    """Load brain-image pairs for reconstruction (NO LABELS)"""
    print("📂 Loading brain-image pairs for reconstruction...")
    
    try:
        # Load original .mat file
        mat_data = sio.loadmat('../dataset/digit69_28x28.mat')
        
        # Extract data WITHOUT labels
        brainTrn = mat_data['fmriTrn'].astype(np.float32)
        imagesTrn = mat_data['stimTrn'].astype(np.float32)
        
        brainTest = mat_data['fmriTest'].astype(np.float32)
        imagesTest = mat_data['stimTest'].astype(np.float32)
        
        print(f"✅ Brain-image dataset loaded for reconstruction!")
        print(f"   Training: {len(brainTrn)} brain-image pairs")
        print(f"   Test: {len(brainTest)} brain-image pairs")
        print(f"   Brain signals: {brainTrn.shape}")
        print(f"   Visual stimuli: {imagesTrn.shape}")
        print(f"   Task: Brain → Image reconstruction via contrastive learning")
        
        return brainTrn, imagesTrn, brainTest, imagesTest
        
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

def enhanced_reconstruction_training(brain_encoder, train_loader, test_loader, device, epochs=100, lr=1e-3):
    """Enhanced training for brain-to-image reconstruction"""
    print("🧠 Starting brain-to-image RECONSTRUCTION training...")
    print(f"📊 Task: Brain signals → Visual image reconstruction")
    print(f"🎯 Method: Contrastive learning (brain ↔ CLIP image embeddings)")
    print(f"🚫 NO CLASSIFICATION - Pure reconstruction task")
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(brain_encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track reconstruction metrics
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_similarities': [],
        'val_similarities': [],
        'train_retrieval_acc': [],  # Can brain retrieve correct image?
        'val_retrieval_acc': [],
        'train_top5_retrieval': [],
        'val_top5_retrieval': [],
        'learning_rates': []
    }
    
    best_val_similarity = 0.0
    patience = 25
    patience_counter = 0
    
    print(f"🚀 Reconstruction Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Patience: {patience}")
    print(f"   Loss: Negative cosine similarity")
    print(f"   Evaluation: Image retrieval accuracy")
    
    for epoch in range(epochs):
        # TRAINING PHASE
        brain_encoder.train()
        epoch_train_loss = 0.0
        epoch_train_sim = 0.0
        train_batches = 0
        
        train_brain_embs = []
        train_image_embs = []
        
        for batch_idx, (brain, images) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            brain = brain.to(device)
            images = images.to(device)
            
            # Forward pass
            brain_emb = brain_encoder(brain)
            
            with torch.no_grad():
                image_emb = clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
            
            # Compute contrastive loss (simple variant: negative cosine similarity)
            # This maximizes similarity between brain-image pairs (positive pairs)
            similarities = F.cosine_similarity(brain_emb, image_emb, dim=1)
            loss = -similarities.mean()  # Contrastive loss: maximize similarity
            
            avg_similarity = similarities.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain_encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_sim += avg_similarity.item()
            train_batches += 1
            
            # Store embeddings for retrieval evaluation
            train_brain_embs.append(brain_emb.detach().cpu())
            train_image_embs.append(image_emb.detach().cpu())
        
        # Calculate training retrieval metrics
        train_brain_all = torch.cat(train_brain_embs, dim=0)
        train_image_all = torch.cat(train_image_embs, dim=0)
        train_sim_matrix = torch.matmul(train_brain_all, train_image_all.T)
        
        train_retrieval_acc = calculate_retrieval_accuracy(train_sim_matrix, k=1)
        train_top5_retrieval = calculate_retrieval_accuracy(train_sim_matrix, k=5)
        
        # VALIDATION PHASE
        brain_encoder.eval()
        epoch_val_loss = 0.0
        epoch_val_sim = 0.0
        val_batches = 0
        
        val_brain_embs = []
        val_image_embs = []
        
        with torch.no_grad():
            for brain, images in tqdm(test_loader, desc=f"Val Epoch {epoch+1}"):
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
        
        # Calculate validation retrieval metrics
        val_brain_all = torch.cat(val_brain_embs, dim=0)
        val_image_all = torch.cat(val_image_embs, dim=0)
        val_sim_matrix = torch.matmul(val_brain_all, val_image_all.T)
        
        val_retrieval_acc = calculate_retrieval_accuracy(val_sim_matrix, k=1)
        val_top5_retrieval = calculate_retrieval_accuracy(val_sim_matrix, k=5)
        
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
        history['train_retrieval_acc'].append(train_retrieval_acc)
        history['val_retrieval_acc'].append(val_retrieval_acc)
        history['train_top5_retrieval'].append(train_top5_retrieval)
        history['val_top5_retrieval'].append(val_top5_retrieval)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"           Train Sim={avg_train_sim:.3f}, Val Sim={avg_val_sim:.3f}")
            print(f"           Train Retr={train_retrieval_acc:.3f}, Val Retr={val_retrieval_acc:.3f}")
            print(f"           Train Top5={train_top5_retrieval:.3f}, Val Top5={val_top5_retrieval:.3f}, LR={current_lr:.6f}")
        
        # Early stopping based on validation similarity
        if avg_val_sim > best_val_similarity:
            best_val_similarity = avg_val_sim
            patience_counter = 0
            
            # Save best model
            torch.save({
                'brain_encoder_state_dict': brain_encoder.state_dict(),
                'epoch': epoch,
                'best_val_similarity': best_val_similarity,
                'history': history
            }, 'brain_reconstruction_best.pth')
            print(f"   💾 New best model saved! Val Similarity: {avg_val_sim:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            print(f"   Best validation similarity: {best_val_similarity:.3f}")
            break
    
    return history

def calculate_retrieval_accuracy(similarity_matrix, k=1):
    """Calculate image retrieval accuracy from brain signals"""
    n_samples = similarity_matrix.shape[0]
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    correct_retrieval = 0
    for i in range(n_samples):
        if i in top_k_indices[i]:  # Can brain signal retrieve its corresponding image?
            correct_retrieval += 1
    
    accuracy = correct_retrieval / n_samples
    return accuracy

def visualize_reconstruction_results(brain_encoder, test_loader, clip_model, device, num_examples=5):
    """Visualize brain-to-image reconstruction results"""
    print("🎨 Visualizing reconstruction results...")
    
    brain_encoder.eval()
    
    # Get some test samples
    test_iter = iter(test_loader)
    brain_batch, image_batch = next(test_iter)
    
    brain_batch = brain_batch[:num_examples].to(device)
    image_batch = image_batch[:num_examples].to(device)
    
    with torch.no_grad():
        # Get brain embeddings
        brain_embs = brain_encoder(brain_batch)
        
        # Get all test image embeddings for retrieval
        all_brain_embs = []
        all_image_embs = []
        all_images = []
        
        for brain, images in test_loader:
            brain = brain.to(device)
            images = images.to(device)
            
            brain_emb = brain_encoder(brain)
            image_emb = clip_model.encode_image(images)
            image_emb = F.normalize(image_emb.float(), dim=-1)
            
            all_brain_embs.append(brain_emb.cpu())
            all_image_embs.append(image_emb.cpu())
            all_images.append(images.cpu())
        
        all_brain_embs = torch.cat(all_brain_embs, dim=0)
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(brain_embs.cpu(), all_image_embs.T)
        
        # Create visualization
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
        fig.suptitle('Brain-to-Image Reconstruction Results', fontsize=16, fontweight='bold')
        
        for i in range(num_examples):
            # Original image
            orig_img = image_batch[i].cpu()
            orig_img = orig_img.permute(1, 2, 0)
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f'Original Image {i+1}')
            axes[i, 0].axis('off')
            
            # Retrieved image (top-1)
            similarities = similarity_matrix[i]
            _, top_idx = torch.topk(similarities, 1)
            retrieved_img = all_images[top_idx[0]]
            retrieved_img = retrieved_img.permute(1, 2, 0)
            retrieved_img = (retrieved_img - retrieved_img.min()) / (retrieved_img.max() - retrieved_img.min())
            
            axes[i, 1].imshow(retrieved_img)
            axes[i, 1].set_title(f'Retrieved Image\nSimilarity: {similarities[top_idx[0]]:.3f}')
            axes[i, 1].axis('off')
            
            # Top-5 similarities
            _, top_5_indices = torch.topk(similarities, 5)
            top_5_sims = similarities[top_5_indices]
            
            axes[i, 2].bar(range(5), top_5_sims)
            axes[i, 2].set_title('Top-5 Similarities')
            axes[i, 2].set_xlabel('Rank')
            axes[i, 2].set_ylabel('Similarity')
            axes[i, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'brain_reconstruction_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🎨 Reconstruction visualization saved: {plot_filename}")

def plot_comprehensive_training_results(history, save_plots=True):
    """Create comprehensive training plots for reconstruction"""
    print("📊 Creating comprehensive training plots...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Brain-to-Image Reconstruction Training Results', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_losses']) + 1)

    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', alpha=0.8, color='blue')
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', alpha=0.8, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Contrastive Loss')
    axes[0, 0].set_title('Training vs Validation Contrastive Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Similarity curves
    axes[0, 1].plot(epochs, history['train_similarities'], label='Training Similarity', alpha=0.8, color='blue')
    axes[0, 1].plot(epochs, history['val_similarities'], label='Validation Similarity', alpha=0.8, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Brain-CLIP Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Retrieval Accuracy
    axes[0, 2].plot(epochs, history['train_retrieval_acc'], label='Training Retrieval', alpha=0.8, color='blue')
    axes[0, 2].plot(epochs, history['val_retrieval_acc'], label='Validation Retrieval', alpha=0.8, color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Retrieval Accuracy')
    axes[0, 2].set_title('Image Retrieval Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Top-5 Retrieval
    axes[1, 0].plot(epochs, history['train_top5_retrieval'], label='Training Top-5', alpha=0.8, color='blue')
    axes[1, 0].plot(epochs, history['val_top5_retrieval'], label='Validation Top-5', alpha=0.8, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Retrieval Accuracy')
    axes[1, 0].set_title('Top-5 Image Retrieval')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Learning rate schedule
    axes[1, 1].plot(epochs, history['learning_rates'], alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Loss difference (overfitting detection)
    loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
    axes[1, 2].plot(epochs, loss_diff, alpha=0.8, color='orange')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Val Loss - Train Loss')
    axes[1, 2].set_title('Overfitting Detection')
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 7: Performance comparison
    axes[2, 0].plot(epochs, history['val_similarities'], label='Similarity', alpha=0.8, color='red')
    axes[2, 0].plot(epochs, history['val_retrieval_acc'], label='Top-1 Retrieval', alpha=0.8, color='blue')
    axes[2, 0].plot(epochs, history['val_top5_retrieval'], label='Top-5 Retrieval', alpha=0.8, color='green')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Validation Performance')
    axes[2, 0].set_title('Validation Metrics Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Training summary
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    best_val_similarity = max(history['val_similarities'])
    best_retrieval_acc = max(history['val_retrieval_acc'])

    summary_text = f"""Brain Reconstruction Training Summary:

Task: Brain signals → Image reconstruction
Method: Contrastive learning (NO CLASSIFICATION)

Final Results:
  Train Loss: {final_train_loss:.4f}
  Val Loss: {final_val_loss:.4f}
  Train Sim: {history['train_similarities'][-1]:.3f}
  Val Sim: {history['val_similarities'][-1]:.3f}

Best Performance:
  Best Val Similarity: {best_val_similarity:.3f}
  Best Retrieval Acc: {best_retrieval_acc:.3f}
  Best Top-5 Retrieval: {max(history['val_top5_retrieval']):.3f}

Training Info:
  Total Epochs: {len(epochs)}
  Final LR: {history['learning_rates'][-1]:.2e}

Architecture:
  Brain Encoder: 3092 → 512 dims
  Target: CLIP ViT-B/32 embeddings
  Loss: Contrastive loss (negative cosine similarity)

Reconstruction Performance:
  Task: Brain → Image retrieval
  Random baseline: 10.0% (1/10)
  Achieved: {best_retrieval_acc*100:.1f}%
  Improvement: +{((best_retrieval_acc-0.1)/0.1)*100:.0f}%"""

    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].axis('off')
    axes[2, 1].set_title('Training Summary')

    # Plot 9: Similarity progression
    val_sim_best_so_far = []
    current_best = 0
    for sim in history['val_similarities']:
        if sim > current_best:
            current_best = sim
        val_sim_best_so_far.append(current_best)

    axes[2, 2].plot(epochs, history['val_similarities'], alpha=0.6, label='Val Similarity')
    axes[2, 2].plot(epochs, val_sim_best_so_far, linewidth=2, label='Best So Far')
    axes[2, 2].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Target')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Cosine Similarity')
    axes[2, 2].set_title('Similarity Progression')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'brain_reconstruction_training_plots_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved: {plot_filename}")

    plt.show()

    return fig

def main():
    """Main function for brain-to-image reconstruction"""
    print("🧠 BRAIN-TO-IMAGE RECONSTRUCTION TRAINING")
    print("=" * 70)
    print("Task: Brain signals → Visual image reconstruction")
    print("Method: Contrastive learning (NO CLASSIFICATION)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    try:
        # Load brain-image pairs (NO LABELS)
        brainTrn, imagesTrn, brainTest, imagesTest = load_brain_image_data()
        
        # Create CLIP preprocessing
        clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Create datasets (NO LABELS)
        train_dataset = BrainImageDataset(brainTrn, imagesTrn, transform=clip_preprocess)
        test_dataset = BrainImageDataset(brainTest, imagesTest, transform=clip_preprocess)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"📊 Reconstruction dataset prepared:")
        print(f"   Training pairs: {len(train_dataset)}")
        print(f"   Test pairs: {len(test_dataset)}")
        print(f"   Task: Brain → Image reconstruction")
        
        # Create brain encoder
        brain_encoder = BrainEncoder(input_dim=3092, embedding_dim=512).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in brain_encoder.parameters())
        print(f"🤖 Brain Encoder created: {total_params:,} parameters")
        
        # Load CLIP for visualization
        clip_model, _ = clip.load("ViT-B/32", device=device)
        
        # Run reconstruction training
        history = enhanced_reconstruction_training(
            brain_encoder, train_loader, test_loader, device,
            epochs=100, lr=1e-3
        )

        # Create comprehensive training plots
        plot_comprehensive_training_results(history, save_plots=True)

        # Visualize reconstruction results
        visualize_reconstruction_results(brain_encoder, test_loader, clip_model, device)
        
        # Print final performance
        best_val_similarity = max(history['val_similarities'])
        best_retrieval_acc = max(history['val_retrieval_acc'])
        print(f"\n🏆 RECONSTRUCTION PERFORMANCE:")
        print(f"   Best Validation Similarity: {best_val_similarity:.3f}")
        print(f"   Best Retrieval Accuracy: {best_retrieval_acc:.3f} ({best_retrieval_acc*100:.1f}%)")
        print(f"   Random Retrieval Baseline: {1/len(test_dataset):.3f} ({100/len(test_dataset):.1f}%)")
        
        if best_retrieval_acc > 1/len(test_dataset):
            improvement = (best_retrieval_acc - 1/len(test_dataset)) / (1/len(test_dataset)) * 100
            print(f"   Improvement over random: +{improvement:.1f}% 🎉")
        
        return brain_encoder, history
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    main()
