#!/usr/bin/env python3
"""
Analyze Miyawaki3 CLIP-based Results
Load trained model and create comprehensive visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import clip
from PIL import Image
import torchvision.transforms as transforms
from runembedding import MiyawakiDecoder

def load_trained_model():
    """Load the trained miyawaki3 model"""
    print("🔍 Loading Trained Miyawaki3 Model")
    print("=" * 50)
    
    model_path = "miyawaki_contrastive_clip.pth"
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please run runembedding.py first to train the model")
        return None
    
    # Initialize decoder
    decoder = MiyawakiDecoder()
    
    # Load data to get proper dimensions
    mat_file_path = "../dataset/miyawaki_structured_28x28.mat"
    decoder.load_data(mat_file_path)
    
    # Initialize models
    decoder.initialize_models()
    
    # Load trained weights
    decoder.load_model(model_path)
    
    print(f"✅ Model loaded successfully")
    print(f"📊 fMRI Encoder parameters: {sum(p.numel() for p in decoder.fmri_encoder.parameters()):,}")
    
    return decoder

def analyze_embeddings(decoder):
    """Analyze fMRI and image embeddings"""
    print("\n🧠 Analyzing Embeddings")
    print("=" * 50)
    
    # Create dataloaders
    train_loader, test_loader = decoder.create_dataloaders(batch_size=32)
    
    # Get embeddings
    decoder.fmri_encoder.eval()
    
    all_fmri_embeddings = []
    all_image_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        # Test set embeddings
        for fmri_batch, image_batch in test_loader:
            fmri_batch = fmri_batch.to(decoder.device)
            image_batch = image_batch.to(decoder.device)
            
            # Get fMRI embeddings
            fmri_emb = decoder.fmri_encoder(fmri_batch)
            fmri_emb = F.normalize(fmri_emb, dim=1)
            
            # Get CLIP image embeddings
            image_emb = decoder.clip_model.encode_image(image_batch)
            image_emb = F.normalize(image_emb.float(), dim=1)
            
            all_fmri_embeddings.append(fmri_emb.cpu())
            all_image_embeddings.append(image_emb.cpu())
            all_labels.extend([f"Test_{i}" for i in range(len(fmri_batch))])
    
    # Concatenate all embeddings
    fmri_embeddings = torch.cat(all_fmri_embeddings, dim=0)
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    print(f"📊 fMRI embeddings shape: {fmri_embeddings.shape}")
    print(f"📊 Image embeddings shape: {image_embeddings.shape}")
    
    return fmri_embeddings, image_embeddings, all_labels

def create_similarity_analysis(fmri_embeddings, image_embeddings, labels):
    """Create similarity matrix analysis"""
    print("\n📊 Creating Similarity Analysis")
    print("=" * 50)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(fmri_embeddings, image_embeddings.t())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Miyawaki3 CLIP-based Cross-Modal Analysis', fontsize=16, fontweight='bold')
    
    # 1. Similarity Matrix Heatmap
    im1 = axes[0, 0].imshow(similarity_matrix.numpy(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Cross-Modal Similarity Matrix')
    axes[0, 0].set_xlabel('Image Index')
    axes[0, 0].set_ylabel('fMRI Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Add diagonal line (perfect matches)
    n_samples = min(similarity_matrix.shape)
    axes[0, 0].plot([0, n_samples-1], [0, n_samples-1], 'r--', alpha=0.7, linewidth=2, label='Perfect Match')
    axes[0, 0].legend()
    
    # 2. Similarity Distribution
    similarities = similarity_matrix.numpy()
    diagonal_sims = np.diag(similarities)  # Correct matches
    off_diagonal_sims = similarities[~np.eye(similarities.shape[0], dtype=bool)]  # Incorrect matches
    
    axes[0, 1].hist(off_diagonal_sims, bins=30, alpha=0.7, label='Incorrect Matches', color='red')
    axes[0, 1].hist(diagonal_sims, bins=30, alpha=0.7, label='Correct Matches', color='green')
    axes[0, 1].set_title('Similarity Score Distribution')
    axes[0, 1].set_xlabel('Similarity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Retrieval Accuracy by Rank
    ranks = []
    for i in range(similarity_matrix.shape[0]):
        # Get rank of correct match
        similarities_i = similarity_matrix[i]
        sorted_indices = torch.argsort(similarities_i, descending=True)
        correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(correct_rank)
    
    # Calculate cumulative accuracy
    max_rank = max(ranks)
    cumulative_acc = []
    rank_range = range(1, max_rank + 1)
    
    for k in rank_range:
        acc = sum(1 for r in ranks if r <= k) / len(ranks)
        cumulative_acc.append(acc)
    
    axes[1, 0].plot(rank_range, cumulative_acc, 'b-', linewidth=2, marker='o')
    axes[1, 0].set_title('Cumulative Retrieval Accuracy')
    axes[1, 0].set_xlabel('Rank (Top-K)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    # Add key metrics as text
    top1_acc = cumulative_acc[0] if cumulative_acc else 0
    top5_acc = cumulative_acc[4] if len(cumulative_acc) > 4 else cumulative_acc[-1]
    top10_acc = cumulative_acc[9] if len(cumulative_acc) > 9 else cumulative_acc[-1]
    
    axes[1, 0].text(0.05, 0.95, f'Top-1: {top1_acc:.1%}\nTop-5: {top5_acc:.1%}\nTop-10: {top10_acc:.1%}', 
                   transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Embedding Space Analysis (PCA)
    from sklearn.decomposition import PCA
    
    # Combine embeddings for PCA
    all_embeddings = torch.cat([fmri_embeddings, image_embeddings], dim=0)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings.numpy())
    
    n_fmri = fmri_embeddings.shape[0]
    fmri_2d = embeddings_2d[:n_fmri]
    image_2d = embeddings_2d[n_fmri:]
    
    axes[1, 1].scatter(fmri_2d[:, 0], fmri_2d[:, 1], c='red', alpha=0.7, label='fMRI', s=50)
    axes[1, 1].scatter(image_2d[:, 0], image_2d[:, 1], c='blue', alpha=0.7, label='Images', s=50)
    
    # Draw lines connecting corresponding pairs
    for i in range(min(n_fmri, len(image_2d))):
        axes[1, 1].plot([fmri_2d[i, 0], image_2d[i, 0]], 
                       [fmri_2d[i, 1], image_2d[i, 1]], 
                       'gray', alpha=0.3, linewidth=1)
    
    axes[1, 1].set_title('Embedding Space (PCA)')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('miyawaki3_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Similarity analysis saved as 'miyawaki3_similarity_analysis.png'")
    
    return similarity_matrix, ranks

def create_retrieval_examples(decoder, similarity_matrix, n_examples=6):
    """Create retrieval examples visualization"""
    print("\n🎨 Creating Retrieval Examples")
    print("=" * 50)
    
    # Get test data
    _, test_loader = decoder.create_dataloaders(batch_size=32)
    
    # Get all test images
    test_images = []
    for _, images in test_loader:
        for img in images:
            # Denormalize CLIP preprocessing
            img_denorm = img * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            img_denorm += torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            img_denorm = torch.clamp(img_denorm, 0, 1)
            test_images.append(img_denorm.permute(1, 2, 0).numpy())
    
    # Create visualization
    n_examples = min(n_examples, similarity_matrix.shape[0])
    fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4*n_examples))
    fig.suptitle('Miyawaki3 Cross-Modal Retrieval Examples', fontsize=16, fontweight='bold')
    
    for i in range(n_examples):
        # Get top-3 retrieved images
        similarities_i = similarity_matrix[i]
        _, top_indices = torch.topk(similarities_i, 3)
        
        # Original image (ground truth)
        axes[i, 0].imshow(test_images[i])
        axes[i, 0].set_title(f'Ground Truth {i}')
        axes[i, 0].axis('off')
        
        # Top-3 retrieved images
        for j, idx in enumerate(top_indices):
            retrieved_idx = idx.item()
            similarity_score = similarities_i[retrieved_idx].item()
            
            axes[i, j+1].imshow(test_images[retrieved_idx])
            
            # Color code: green for correct, red for incorrect
            color = 'green' if retrieved_idx == i else 'red'
            rank_text = f'Rank {j+1}' if retrieved_idx == i else f'Rank {j+1} ❌'
            
            axes[i, j+1].set_title(f'{rank_text}\nSim: {similarity_score:.3f}', 
                                  color=color, fontweight='bold')
            axes[i, j+1].axis('off')
            
            # Add border
            for spine in axes[i, j+1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    # Add column labels
    axes[0, 0].set_ylabel('Ground Truth', rotation=90, fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Top-1', rotation=90, fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Top-2', rotation=90, fontsize=12, fontweight='bold')
    axes[0, 3].set_ylabel('Top-3', rotation=90, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('miyawaki3_retrieval_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Retrieval examples saved as 'miyawaki3_retrieval_examples.png'")

def create_performance_summary(ranks, similarity_matrix):
    """Create performance summary visualization"""
    print("\n📈 Creating Performance Summary")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Miyawaki3 Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Rank Distribution
    axes[0, 0].hist(ranks, bins=range(1, max(ranks)+2), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Correct Match Ranks')
    axes[0, 0].set_xlabel('Rank of Correct Match')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    axes[0, 0].axvline(mean_rank, color='red', linestyle='--', label=f'Mean: {mean_rank:.1f}')
    axes[0, 0].axvline(median_rank, color='green', linestyle='--', label=f'Median: {median_rank:.1f}')
    axes[0, 0].legend()
    
    # 2. Top-K Accuracy
    max_k = min(10, similarity_matrix.shape[1])
    k_values = range(1, max_k + 1)
    accuracies = []
    
    for k in k_values:
        acc = sum(1 for r in ranks if r <= k) / len(ranks)
        accuracies.append(acc)
    
    axes[0, 1].bar(k_values, accuracies, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Top-K Retrieval Accuracy')
    axes[0, 1].set_xlabel('K (Top-K)')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, acc in enumerate(accuracies):
        axes[0, 1].text(i+1, acc + 0.02, f'{acc:.1%}', ha='center', fontweight='bold')
    
    # 3. Similarity Score Statistics
    similarities = similarity_matrix.numpy()
    diagonal_sims = np.diag(similarities)
    off_diagonal_sims = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    
    stats_data = [diagonal_sims, off_diagonal_sims]
    labels = ['Correct Matches', 'Incorrect Matches']
    colors = ['green', 'red']
    
    bp = axes[1, 0].boxplot(stats_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_title('Similarity Score Distribution')
    axes[1, 0].set_ylabel('Similarity Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance Metrics Table
    axes[1, 1].axis('off')
    
    # Calculate metrics
    top1_acc = accuracies[0] if accuracies else 0
    top5_acc = accuracies[4] if len(accuracies) > 4 else accuracies[-1]
    top10_acc = accuracies[9] if len(accuracies) > 9 else accuracies[-1]
    
    mean_correct_sim = np.mean(diagonal_sims)
    mean_incorrect_sim = np.mean(off_diagonal_sims)
    
    # Create table
    metrics_data = [
        ['Metric', 'Value'],
        ['Top-1 Accuracy', f'{top1_acc:.1%}'],
        ['Top-5 Accuracy', f'{top5_acc:.1%}'],
        ['Top-10 Accuracy', f'{top10_acc:.1%}'],
        ['Mean Rank', f'{mean_rank:.1f}'],
        ['Median Rank', f'{median_rank:.1f}'],
        ['Mean Correct Similarity', f'{mean_correct_sim:.3f}'],
        ['Mean Incorrect Similarity', f'{mean_incorrect_sim:.3f}'],
        ['Similarity Gap', f'{mean_correct_sim - mean_incorrect_sim:.3f}']
    ]
    
    table = axes[1, 1].table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    axes[1, 1].set_title('Performance Metrics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('miyawaki3_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Performance summary saved as 'miyawaki3_performance_summary.png'")
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'mean_correct_similarity': mean_correct_sim,
        'mean_incorrect_similarity': mean_incorrect_sim
    }

def main():
    """Main analysis function"""
    print("🔍 Miyawaki3 Results Analysis")
    print("=" * 60)
    
    # Load trained model
    decoder = load_trained_model()
    if decoder is None:
        return
    
    # Analyze embeddings
    fmri_embeddings, image_embeddings, labels = analyze_embeddings(decoder)
    
    # Create similarity analysis
    similarity_matrix, ranks = create_similarity_analysis(fmri_embeddings, image_embeddings, labels)
    
    # Create retrieval examples
    create_retrieval_examples(decoder, similarity_matrix)
    
    # Create performance summary
    metrics = create_performance_summary(ranks, similarity_matrix)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📋 MIYAWAKI3 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"🎯 Cross-Modal Retrieval Performance:")
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.1%}")
    print(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.1%}")
    
    print(f"\n📊 Ranking Statistics:")
    print(f"  Mean Rank: {metrics['mean_rank']:.1f}")
    print(f"  Median Rank: {metrics['median_rank']:.1f}")
    
    print(f"\n🔗 Similarity Analysis:")
    print(f"  Correct Match Similarity: {metrics['mean_correct_similarity']:.3f}")
    print(f"  Incorrect Match Similarity: {metrics['mean_incorrect_similarity']:.3f}")
    print(f"  Similarity Gap: {metrics['mean_correct_similarity'] - metrics['mean_incorrect_similarity']:.3f}")
    
    print(f"\n📁 Generated Visualizations:")
    print(f"  - miyawaki3_similarity_analysis.png")
    print(f"  - miyawaki3_retrieval_examples.png")
    print(f"  - miyawaki3_performance_summary.png")
    
    print(f"\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
