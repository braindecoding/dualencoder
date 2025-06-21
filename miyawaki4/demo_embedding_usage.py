#!/usr/bin/env python3
"""
Demo: How to Use Miyawaki4 Embeddings for Downstream Tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import json

def load_embeddings():
    """Load the converted embeddings"""
    print("📥 Loading Miyawaki4 Embeddings")
    print("=" * 40)
    
    # Load embeddings
    with open("miyawaki4_embeddings.pkl", 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load metadata
    with open("miyawaki4_embeddings_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Embeddings loaded successfully")
    print(f"   📊 Training samples: {metadata['train_samples']}")
    print(f"   📊 Test samples: {metadata['test_samples']}")
    print(f"   📊 Embedding dimension: {metadata['clip_dim']}")
    
    return embeddings_data, metadata

def demo_cross_modal_retrieval(embeddings_data):
    """Demo: Cross-modal retrieval using embeddings"""
    print("\n🔍 Demo: Cross-Modal Retrieval")
    print("=" * 40)
    
    # Get test embeddings
    test_fmri = embeddings_data['test']['fmri_embeddings']
    test_image = embeddings_data['test']['image_embeddings']
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(test_fmri, test_image)
    
    # Compute retrieval metrics
    n_samples = len(test_fmri)
    top1_correct = 0
    top3_correct = 0
    
    for i in range(n_samples):
        # Get top-3 retrieved images for fMRI i
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1][:3]
        
        if i in top_indices[:1]:
            top1_correct += 1
        if i in top_indices[:3]:
            top3_correct += 1
    
    top1_acc = top1_correct / n_samples
    top3_acc = top3_correct / n_samples
    
    print(f"🎯 Retrieval Results:")
    print(f"   Top-1 Accuracy: {top1_acc:.1%} ({top1_correct}/{n_samples})")
    print(f"   Top-3 Accuracy: {top3_acc:.1%} ({top3_correct}/{n_samples})")
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cross-Modal Similarity Matrix\n(fMRI vs Images)')
    plt.xlabel('Image Index')
    plt.ylabel('fMRI Index')
    
    # Add diagonal line for perfect matches
    plt.plot([0, n_samples-1], [0, n_samples-1], 'r--', alpha=0.7, linewidth=2, label='Perfect Match')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('demo_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'top1_accuracy': top1_acc, 'top3_accuracy': top3_acc, 'similarity_matrix': similarity_matrix}

def demo_embedding_space_analysis(embeddings_data):
    """Demo: Analyze embedding space structure"""
    print("\n🧠 Demo: Embedding Space Analysis")
    print("=" * 40)
    
    # Combine all embeddings
    all_fmri = np.vstack([embeddings_data['train']['fmri_embeddings'], 
                         embeddings_data['test']['fmri_embeddings']])
    all_image = np.vstack([embeddings_data['train']['image_embeddings'], 
                          embeddings_data['test']['image_embeddings']])
    
    # Create labels
    n_train = len(embeddings_data['train']['fmri_embeddings'])
    n_test = len(embeddings_data['test']['fmri_embeddings'])
    labels = ['Train'] * n_train + ['Test'] * n_test
    
    print(f"📊 Analyzing {len(all_fmri)} fMRI and {len(all_image)} image embeddings")
    
    # PCA Analysis
    print("🔍 Performing PCA...")
    combined_embeddings = np.vstack([all_fmri, all_image])
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(combined_embeddings)
    
    fmri_2d = embeddings_2d[:len(all_fmri)]
    image_2d = embeddings_2d[len(all_fmri):]
    
    # Visualize PCA
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(fmri_2d[:n_train, 0], fmri_2d[:n_train, 1], c='red', alpha=0.7, label='fMRI Train', s=50)
    plt.scatter(fmri_2d[n_train:, 0], fmri_2d[n_train:, 1], c='darkred', alpha=0.7, label='fMRI Test', s=50)
    plt.scatter(image_2d[:n_train, 0], image_2d[:n_train, 1], c='blue', alpha=0.7, label='Image Train', s=50)
    plt.scatter(image_2d[n_train:, 0], image_2d[n_train:, 1], c='darkblue', alpha=0.7, label='Image Test', s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA: fMRI vs Image Embeddings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compute embedding similarities
    plt.subplot(1, 3, 2)
    train_similarities = np.sum(embeddings_data['train']['fmri_embeddings'] * 
                               embeddings_data['train']['image_embeddings'], axis=1)
    test_similarities = np.sum(embeddings_data['test']['fmri_embeddings'] * 
                              embeddings_data['test']['image_embeddings'], axis=1)
    
    plt.hist(train_similarities, bins=15, alpha=0.7, label='Training', color='green')
    plt.hist(test_similarities, bins=15, alpha=0.7, label='Test', color='orange')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('fMRI-Image Similarity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Embedding norms
    plt.subplot(1, 3, 3)
    fmri_norms = np.linalg.norm(all_fmri, axis=1)
    image_norms = np.linalg.norm(all_image, axis=1)
    
    plt.scatter(fmri_norms[:n_train], image_norms[:n_train], alpha=0.7, label='Training', s=50)
    plt.scatter(fmri_norms[n_train:], image_norms[n_train:], alpha=0.7, label='Test', s=50)
    plt.xlabel('fMRI Embedding Norm')
    plt.ylabel('Image Embedding Norm')
    plt.title('Embedding Norms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_embedding_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Analysis completed:")
    print(f"   📊 PCA explained variance: {pca.explained_variance_ratio_[:2].sum():.1%}")
    print(f"   📊 Mean fMRI norm: {fmri_norms.mean():.3f}")
    print(f"   📊 Mean image norm: {image_norms.mean():.3f}")
    
    return {
        'pca': pca,
        'fmri_2d': fmri_2d,
        'image_2d': image_2d,
        'train_similarities': train_similarities,
        'test_similarities': test_similarities
    }

def demo_simple_decoder(embeddings_data):
    """Demo: Simple decoder from fMRI to image embeddings"""
    print("\n🤖 Demo: Simple fMRI→Image Decoder")
    print("=" * 40)
    
    # Prepare data
    X_train = torch.FloatTensor(embeddings_data['train']['fmri_embeddings'])
    y_train = torch.FloatTensor(embeddings_data['train']['image_embeddings'])
    X_test = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'])
    y_test = torch.FloatTensor(embeddings_data['test']['image_embeddings'])
    
    print(f"📊 Training data: {X_train.shape} → {y_train.shape}")
    print(f"📊 Test data: {X_test.shape} → {y_test.shape}")
    
    # Simple linear decoder
    class SimpleDecoder(nn.Module):
        def __init__(self, input_dim=512, output_dim=512):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, output_dim)
            )
        
        def forward(self, x):
            return self.decoder(x)
    
    # Train decoder
    model = SimpleDecoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("🏋️ Training decoder...")
    train_losses = []
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        test_loss = criterion(pred_test, y_test).item()
        
        # Compute cosine similarities
        pred_norm = torch.nn.functional.normalize(pred_test, dim=1)
        target_norm = torch.nn.functional.normalize(y_test, dim=1)
        similarities = torch.sum(pred_norm * target_norm, dim=1)
        mean_similarity = similarities.mean().item()
    
    print(f"✅ Training completed:")
    print(f"   📊 Final train loss: {train_losses[-1]:.4f}")
    print(f"   📊 Test loss: {test_loss:.4f}")
    print(f"   📊 Mean cosine similarity: {mean_similarity:.3f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    similarities_np = similarities.numpy()
    plt.hist(similarities_np, bins=10, alpha=0.7, color='green')
    plt.axvline(mean_similarity, color='red', linestyle='--', label=f'Mean: {mean_similarity:.3f}')
    plt.title('Predicted vs Target Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_decoder_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_loss': test_loss,
        'mean_similarity': mean_similarity,
        'predictions': pred_test.numpy()
    }

def main():
    """Main demo function"""
    print("🎯 Miyawaki4 Embedding Usage Demo")
    print("=" * 50)
    
    # Load embeddings
    embeddings_data, metadata = load_embeddings()
    
    # Demo 1: Cross-modal retrieval
    retrieval_results = demo_cross_modal_retrieval(embeddings_data)
    
    # Demo 2: Embedding space analysis
    analysis_results = demo_embedding_space_analysis(embeddings_data)
    
    # Demo 3: Simple decoder
    decoder_results = demo_simple_decoder(embeddings_data)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DEMO SUMMARY")
    print("=" * 50)
    
    print(f"🔍 Cross-Modal Retrieval:")
    print(f"   Top-1 Accuracy: {retrieval_results['top1_accuracy']:.1%}")
    print(f"   Top-3 Accuracy: {retrieval_results['top3_accuracy']:.1%}")
    
    print(f"\n🧠 Embedding Analysis:")
    print(f"   PCA Variance Explained: {analysis_results['pca'].explained_variance_ratio_[:2].sum():.1%}")
    print(f"   Mean Train Similarity: {analysis_results['train_similarities'].mean():.3f}")
    print(f"   Mean Test Similarity: {analysis_results['test_similarities'].mean():.3f}")
    
    print(f"\n🤖 Simple Decoder:")
    print(f"   Test Loss: {decoder_results['test_loss']:.4f}")
    print(f"   Mean Similarity: {decoder_results['mean_similarity']:.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   - demo_similarity_matrix.png")
    print(f"   - demo_embedding_analysis.png")
    print(f"   - demo_decoder_training.png")
    
    print(f"\n🎯 Next Steps:")
    print(f"   - Use embeddings for image generation")
    print(f"   - Implement more sophisticated decoders")
    print(f"   - Transfer learning to other datasets")
    print(f"   - Real-time brain-computer interface")

if __name__ == "__main__":
    main()
