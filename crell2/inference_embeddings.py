#!/usr/bin/env python3
"""
Inference Embeddings Generator
Generate embeddings from EEG data using trained stable_eeg_model_best.pth
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel

class EEGEmbeddingInference:
    """Class for generating embeddings from trained EEG model"""
    
    def __init__(self, model_path='stable_eeg_model_best.pth', device='auto'):
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
        self.model_path = model_path
        
        print(f"🧠 EEG EMBEDDING INFERENCE")
        print(f"=" * 50)
        print(f"📱 Device: {self.device}")
        print(f"📁 Model: {model_path}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading trained model...")
        
        # Initialize model with same architecture as training
        model = EnhancedEEGToEmbeddingModel(
            n_channels=64,
            seq_len=500,
            d_model=256,
            embedding_dim=512,
            nhead=8,
            num_layers=8,
            patch_size=25,
            dropout=0.1
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            # Filter out classifier layers if they exist
            model_state = checkpoint['model_state_dict']
            current_model_state = model.state_dict()

            # Load only matching keys (exclude classifier)
            filtered_state = {k: v for k, v in model_state.items()
                            if k in current_model_state and current_model_state[k].shape == v.shape}
            current_model_state.update(filtered_state)
            model.load_state_dict(current_model_state)

            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Training accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
            print(f"   Loaded {len(filtered_state)} layers (excluded classifier)")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
        
        model.eval()
        print(f"🎯 Model ready for inference!")
        
        return model
    
    def generate_embeddings(self, eeg_data, batch_size=16, return_labels=False):
        """
        Generate embeddings from EEG data
        
        Args:
            eeg_data: numpy array of shape (N, channels, timepoints) or dataset
            batch_size: batch size for inference
            return_labels: whether to return labels if available
            
        Returns:
            embeddings: numpy array of shape (N, embedding_dim)
            labels: numpy array of labels (if return_labels=True and available)
        """
        print(f"🔄 Generating embeddings...")
        
        # Handle different input types
        if isinstance(eeg_data, str):
            # Load from file
            eeg_data, labels = self._load_data_from_file(eeg_data)
        elif hasattr(eeg_data, '__getitem__') and hasattr(eeg_data, '__len__'):
            # Dataset object
            eeg_data, labels = self._extract_from_dataset(eeg_data)
        else:
            # Numpy array
            labels = None
            
        print(f"📊 Input EEG data shape: {eeg_data.shape}")
        
        # Generate embeddings in batches
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(eeg_data), batch_size), desc="Generating embeddings"):
                batch_eeg = eeg_data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_eeg).to(self.device)
                
                # Generate embeddings
                batch_embeddings = self.model(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        print(f"✅ Generated embeddings:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"   Mean: {embeddings.mean():.3f}")
        print(f"   Std: {embeddings.std():.3f}")
        
        if return_labels and labels is not None:
            return embeddings, labels
        else:
            return embeddings
    
    def _load_data_from_file(self, file_path):
        """Load EEG data from pickle file"""
        print(f"📂 Loading data from {file_path}...")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'validation' in data:
            # Crell dataset format
            eeg_data = data['validation']['eeg']
            labels = data['validation']['labels']
        elif 'eeg' in data and 'labels' in data:
            # Direct format
            eeg_data = data['eeg']
            labels = data['labels']
        else:
            raise ValueError("Unknown data format")
            
        return eeg_data, labels
    
    def _extract_from_dataset(self, dataset):
        """Extract data from dataset object"""
        print(f"📊 Extracting data from dataset...")
        
        eeg_data = []
        labels = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            if len(item) >= 2:
                eeg_data.append(item[0].numpy() if torch.is_tensor(item[0]) else item[0])
                labels.append(item[1])
        
        eeg_data = np.array(eeg_data)
        labels = np.array(labels)
        
        return eeg_data, labels
    
    def analyze_embeddings(self, embeddings, labels=None, save_prefix="inference"):
        """Analyze and visualize generated embeddings"""
        print(f"\n📊 ANALYZING EMBEDDINGS")
        print(f"=" * 40)
        
        # Basic statistics
        print(f"📈 Embedding Statistics:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Mean: {embeddings.mean():.4f}")
        print(f"   Std: {embeddings.std():.4f}")
        print(f"   Min: {embeddings.min():.4f}")
        print(f"   Max: {embeddings.max():.4f}")
        
        # Dimensionality analysis
        print(f"\n🔍 Dimensionality Analysis:")
        
        # PCA analysis
        pca = PCA()
        pca.fit(embeddings)
        
        # Explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        
        print(f"   90% variance: {n_components_90} components")
        print(f"   95% variance: {n_components_95} components")
        print(f"   First 10 components: {cumsum_var[:10]}")
        
        # Similarity analysis
        print(f"\n🔗 Similarity Analysis:")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarity)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        print(f"   Mean cosine similarity: {similarities.mean():.4f}")
        print(f"   Std cosine similarity: {similarities.std():.4f}")
        print(f"   Min similarity: {similarities.min():.4f}")
        print(f"   Max similarity: {similarities.max():.4f}")
        
        # Visualizations
        self._create_visualizations(embeddings, labels, save_prefix, pca, similarity_matrix)
        
        return {
            'pca_components_90': n_components_90,
            'pca_components_95': n_components_95,
            'mean_similarity': similarities.mean(),
            'std_similarity': similarities.std()
        }
    
    def _create_visualizations(self, embeddings, labels, save_prefix, pca, similarity_matrix):
        """Create visualization plots"""
        print(f"🎨 Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PCA Explained Variance
        plt.subplot(3, 4, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_)[:50])
        plt.title('PCA Explained Variance (First 50 Components)')
        plt.xlabel('Component')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True, alpha=0.3)
        
        # 2. PCA 2D Visualization
        plt.subplot(3, 4, 2)
        embeddings_2d = pca.transform(embeddings)[:, :2]
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=f'Label {label}', alpha=0.7, s=30)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=30)
        
        plt.title('PCA 2D Projection')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        # 3. t-SNE Visualization (if not too many samples)
        if len(embeddings) <= 1000:
            plt.subplot(3, 4, 3)
            print("   Computing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_tsne = tsne.fit_transform(embeddings)
            
            if labels is not None:
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                               c=[colors[i]], label=f'Label {label}', alpha=0.7, s=30)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7, s=30)
            
            plt.title('t-SNE 2D Projection')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.grid(True, alpha=0.3)
        
        # 4. Embedding Distribution
        plt.subplot(3, 4, 4)
        plt.hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
        plt.title('Embedding Value Distribution')
        plt.xlabel('Embedding Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 5. Similarity Matrix Heatmap (sample)
        plt.subplot(3, 4, 5)
        sample_size = min(100, len(similarity_matrix))
        sample_indices = np.random.choice(len(similarity_matrix), sample_size, replace=False)
        sample_sim = similarity_matrix[np.ix_(sample_indices, sample_indices)]
        
        im = plt.imshow(sample_sim, cmap='viridis', aspect='auto')
        plt.title(f'Similarity Matrix (Sample {sample_size}x{sample_size})')
        plt.colorbar(im)
        
        # 6. Similarity Distribution
        plt.subplot(3, 4, 6)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        plt.hist(similarities, bins=50, alpha=0.7, density=True)
        plt.title('Cosine Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 7. Embedding Norms
        plt.subplot(3, 4, 7)
        norms = np.linalg.norm(embeddings, axis=1)
        plt.hist(norms, bins=50, alpha=0.7, density=True)
        plt.title('Embedding L2 Norms')
        plt.xlabel('L2 Norm')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 8. Component Variance
        plt.subplot(3, 4, 8)
        component_vars = np.var(embeddings, axis=0)
        plt.plot(component_vars)
        plt.title('Variance per Embedding Dimension')
        plt.xlabel('Dimension')
        plt.ylabel('Variance')
        plt.grid(True, alpha=0.3)
        
        # 9-12. First few principal components
        for i in range(4):
            plt.subplot(3, 4, 9+i)
            pc_data = embeddings_2d[:, 0] if i < 2 else embeddings_2d[:, 1]
            
            if labels is not None:
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    plt.hist(pc_data[mask], bins=20, alpha=0.7, 
                            label=f'Label {label}', density=True)
                plt.legend()
            else:
                plt.hist(pc_data, bins=30, alpha=0.7, density=True)
            
            plt.title(f'PC{i+1} Distribution')
            plt.xlabel(f'PC{i+1} Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved as '{save_prefix}_embedding_analysis.png'")

def main():
    """Main inference function"""
    print("🚀 EEG EMBEDDING INFERENCE")
    print("=" * 60)
    print("Generate embeddings from trained stable EEG model")
    print("=" * 60)
    
    # Initialize inference
    inference = EEGEmbeddingInference(
        model_path='stable_eeg_model_best.pth',
        device='auto'
    )
    
    # Generate embeddings from Crell validation data
    embeddings, labels = inference.generate_embeddings(
        'crell_processed_data_correct.pkl',
        batch_size=16,
        return_labels=True
    )
    
    # Analyze embeddings
    analysis_results = inference.analyze_embeddings(
        embeddings, 
        labels, 
        save_prefix="crell_inference"
    )
    
    # Save embeddings
    output_data = {
        'embeddings': embeddings,
        'labels': labels,
        'analysis': analysis_results,
        'model_path': 'stable_eeg_model_best.pth',
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    output_file = f"crell_embeddings_inference_{output_data['timestamp']}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✅ INFERENCE COMPLETED!")
    print(f"   Generated embeddings: {embeddings.shape}")
    print(f"   Mean similarity: {analysis_results['mean_similarity']:.4f}")
    print(f"   PCA 90% variance: {analysis_results['pca_components_90']} components")
    print(f"   Saved to: {output_file}")
    
    print(f"\n📁 Generated files:")
    print(f"   - {output_file}")
    print(f"   - crell_inference_embedding_analysis.png")
    
    print(f"\n🧠 Ready for downstream applications!")
    print(f"   Use embeddings for: classification, clustering, similarity search")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Quality: {analysis_results['mean_similarity']:.4f} mean cosine similarity")

if __name__ == "__main__":
    main()
