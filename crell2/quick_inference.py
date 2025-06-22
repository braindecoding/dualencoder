#!/usr/bin/env python3
"""
Quick Inference - Generate embeddings without heavy visualizations
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel

def quick_inference():
    """Quick embedding generation and analysis"""
    print("🚀 QUICK EEG EMBEDDING INFERENCE")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load model
    print("🔄 Loading model...")
    model = EnhancedEEGToEmbeddingModel(
        n_channels=64, seq_len=500, d_model=256, embedding_dim=512,
        nhead=8, num_layers=8, patch_size=25, dropout=0.1
    ).to(device)
    
    checkpoint = torch.load('stable_eeg_model_best.pth', map_location=device)
    model_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()
    
    # Filter out classifier layers
    filtered_state = {k: v for k, v in model_state.items() 
                     if k in current_model_state and current_model_state[k].shape == v.shape}
    current_model_state.update(filtered_state)
    model.load_state_dict(current_model_state)
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   Accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
    
    # Load data
    print("📂 Loading data...")
    with open('crell_processed_data_correct.pkl', 'rb') as f:
        data = pickle.load(f)
    
    eeg_data = data['validation']['eeg']
    labels = data['validation']['labels']
    
    print(f"📊 Data: {eeg_data.shape}, Labels: {len(labels)}")
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eeg_data), 16)):
            batch = torch.FloatTensor(eeg_data[i:i+16]).to(device)
            batch_emb = model(batch)
            embeddings.append(batch_emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"   Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")
    
    # Quick analysis
    print("\n📊 QUICK ANALYSIS")
    print("=" * 30)
    
    # PCA
    pca = PCA()
    pca.fit(embeddings)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"🔍 PCA Analysis:")
    print(f"   90% variance: {np.argmax(cumsum_var >= 0.9) + 1} components")
    print(f"   95% variance: {np.argmax(cumsum_var >= 0.95) + 1} components")
    print(f"   99% variance: {np.argmax(cumsum_var >= 0.99) + 1} components")
    
    # Similarity
    similarity_matrix = cosine_similarity(embeddings)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    
    print(f"🔗 Similarity Analysis:")
    print(f"   Mean cosine similarity: {similarities.mean():.4f}")
    print(f"   Std cosine similarity: {similarities.std():.4f}")
    print(f"   Range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    # Letter analysis
    unique_labels = np.unique(labels)
    print(f"📝 Letter Analysis:")
    print(f"   Unique letters: {len(unique_labels)}")
    print(f"   Letters: {unique_labels}")
    
    # Inter-letter similarity
    letter_similarities = []
    for label in unique_labels:
        mask_label = labels == label
        if np.sum(mask_label) > 1:
            letter_emb = embeddings[mask_label]
            letter_sim = cosine_similarity(letter_emb)
            letter_mask = ~np.eye(letter_sim.shape[0], dtype=bool)
            letter_similarities.extend(letter_sim[letter_mask])
    
    if letter_similarities:
        letter_similarities = np.array(letter_similarities)
        print(f"   Intra-letter similarity: {letter_similarities.mean():.4f} ± {letter_similarities.std():.4f}")
    
    # Quick visualization
    print("🎨 Creating quick visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # PCA explained variance
    axes[0, 0].plot(cumsum_var[:50])
    axes[0, 0].set_title('PCA Explained Variance')
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Cumulative Variance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCA 2D
    embeddings_2d = pca.transform(embeddings)[:, :2]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        axes[0, 1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[colors[i]], label=f'{label}', alpha=0.7, s=30)
    
    axes[0, 1].set_title('PCA 2D Projection')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Embedding distribution
    axes[0, 2].hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 2].set_title('Embedding Distribution')
    axes[0, 2].set_xlabel('Value')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Similarity distribution
    axes[1, 0].hist(similarities, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_title('Cosine Similarity Distribution')
    axes[1, 0].set_xlabel('Similarity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    axes[1, 1].hist(norms, bins=50, alpha=0.7, density=True)
    axes[1, 1].set_title('Embedding L2 Norms')
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Component variance
    component_vars = np.var(embeddings, axis=0)
    axes[1, 2].plot(component_vars)
    axes[1, 2].set_title('Variance per Dimension')
    axes[1, 2].set_xlabel('Dimension')
    axes[1, 2].set_ylabel('Variance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_inference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save embeddings
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_data = {
        'embeddings': embeddings,
        'labels': labels,
        'model_info': {
            'path': 'stable_eeg_model_best.pth',
            'epoch': checkpoint.get('epoch', 'unknown'),
            'accuracy': checkpoint.get('val_accuracy', 'unknown')
        },
        'analysis': {
            'pca_90': np.argmax(cumsum_var >= 0.9) + 1,
            'pca_95': np.argmax(cumsum_var >= 0.95) + 1,
            'mean_similarity': similarities.mean(),
            'std_similarity': similarities.std()
        },
        'timestamp': timestamp
    }
    
    output_file = f"crell_embeddings_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✅ QUICK INFERENCE COMPLETED!")
    print(f"📊 Results:")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Mean similarity: {similarities.mean():.4f}")
    print(f"   PCA 90%: {np.argmax(cumsum_var >= 0.9) + 1} components")
    print(f"   PCA 95%: {np.argmax(cumsum_var >= 0.95) + 1} components")
    
    print(f"\n📁 Generated files:")
    print(f"   - {output_file}")
    print(f"   - quick_inference_analysis.png")
    
    print(f"\n🧠 Embedding Quality Assessment:")
    if similarities.mean() > 0.8:
        print(f"   🟢 HIGH similarity ({similarities.mean():.3f}) - Good clustering")
    elif similarities.mean() > 0.6:
        print(f"   🟡 MEDIUM similarity ({similarities.mean():.3f}) - Moderate clustering")
    else:
        print(f"   🔴 LOW similarity ({similarities.mean():.3f}) - Poor clustering")
    
    pca_efficiency = np.argmax(cumsum_var >= 0.9) + 1
    if pca_efficiency < 50:
        print(f"   🟢 EFFICIENT representation ({pca_efficiency} components for 90%)")
    elif pca_efficiency < 100:
        print(f"   🟡 MODERATE efficiency ({pca_efficiency} components for 90%)")
    else:
        print(f"   🔴 LOW efficiency ({pca_efficiency} components for 90%)")
    
    print(f"\n🚀 Ready for downstream applications!")
    
    return embeddings, labels, output_data

if __name__ == "__main__":
    embeddings, labels, data = quick_inference()
