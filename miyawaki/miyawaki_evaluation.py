#!/usr/bin/env python3
"""
Evaluasi dan visualisasi hasil training Miyawaki dual encoder
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miyawaki_dual_encoder import MiyawakiDualEncoder
from miyawaki_dataset_loader import load_miyawaki_dataset, create_dataloaders
from pathlib import Path

def load_trained_model(model_path, device='cuda'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MiyawakiDualEncoder(fmri_dim=967, latent_dim=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

def extract_latent_representations(model, dataloader, device='cuda'):
    """Extract latent representations dari trained model"""
    
    fmri_latents = []
    stimulus_latents = []
    correlations = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            label = batch['label']
            
            outputs = model(fmri, stimulus)
            
            fmri_latents.append(outputs['fmri_latent'].cpu())
            stimulus_latents.append(outputs['stimulus_latent'].cpu())
            correlations.append(outputs['correlation'].cpu())
            labels.append(label)
    
    return {
        'fmri_latent': torch.cat(fmri_latents, dim=0).numpy(),
        'stimulus_latent': torch.cat(stimulus_latents, dim=0).numpy(),
        'correlation': torch.cat(correlations, dim=0).numpy(),
        'labels': torch.cat(labels, dim=0).numpy()
    }

def visualize_latent_space(train_repr, test_repr, save_prefix='miyawaki'):
    """Visualisasi latent space menggunakan t-SNE"""
    
    print("Generating t-SNE visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Combine train and test data
    all_fmri = np.vstack([train_repr['fmri_latent'], test_repr['fmri_latent']])
    all_stimulus = np.vstack([train_repr['stimulus_latent'], test_repr['stimulus_latent']])
    all_correlation = np.vstack([train_repr['correlation'], test_repr['correlation']])
    all_labels = np.hstack([train_repr['labels'], test_repr['labels']])
    
    # Create split indicators
    n_train = len(train_repr['labels'])
    split_indicator = ['Train'] * n_train + ['Test'] * len(test_repr['labels'])
    
    # t-SNE for each representation
    representations = {
        'fMRI Latent': all_fmri,
        'Stimulus Latent': all_stimulus,
        'Correlation': all_correlation
    }
    
    for i, (name, data) in enumerate(representations.items()):
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
        embedded = tsne.fit_transform(data)
        
        # Plot by class
        ax = axes[0, i]
        unique_labels = np.unique(all_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = all_labels == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=[color], label=f'Class {label}', alpha=0.7, s=50)
        
        ax.set_title(f'{name} - Colored by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot by train/test split
        ax = axes[1, i]
        train_mask = np.array(split_indicator) == 'Train'
        test_mask = np.array(split_indicator) == 'Test'
        
        ax.scatter(embedded[train_mask, 0], embedded[train_mask, 1], 
                  c='blue', label='Train', alpha=0.7, s=50)
        ax.scatter(embedded[test_mask, 0], embedded[test_mask, 1], 
                  c='red', label='Test', alpha=0.7, s=50)
        
        ax.set_title(f'{name} - Colored by Split')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_latent_space_tsne.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_retrieval_performance(train_repr, test_repr):
    """Evaluate cross-modal retrieval performance"""
    
    print("Evaluating cross-modal retrieval...")
    
    # Use correlation embeddings for retrieval
    train_corr = train_repr['correlation']
    test_corr = test_repr['correlation']
    train_labels = train_repr['labels']
    test_labels = test_repr['labels']
    
    # Compute similarity matrix (test vs train)
    similarity_matrix = np.dot(test_corr, train_corr.T)
    
    # For each test sample, find most similar train sample
    retrieval_results = []
    
    for i, test_label in enumerate(test_labels):
        # Get similarities for this test sample
        similarities = similarity_matrix[i]
        
        # Find top-k most similar train samples
        top_k_indices = np.argsort(similarities)[::-1][:5]  # Top 5
        top_k_labels = train_labels[top_k_indices]
        
        # Check if correct label is in top-k
        correct_in_top1 = (top_k_labels[0] == test_label)
        correct_in_top3 = (test_label in top_k_labels[:3])
        correct_in_top5 = (test_label in top_k_labels[:5])
        
        retrieval_results.append({
            'test_label': test_label,
            'top_k_labels': top_k_labels,
            'similarities': similarities[top_k_indices],
            'correct_top1': correct_in_top1,
            'correct_top3': correct_in_top3,
            'correct_top5': correct_in_top5
        })
    
    # Compute metrics
    top1_accuracy = np.mean([r['correct_top1'] for r in retrieval_results])
    top3_accuracy = np.mean([r['correct_top3'] for r in retrieval_results])
    top5_accuracy = np.mean([r['correct_top5'] for r in retrieval_results])
    
    print(f"Cross-modal Retrieval Results:")
    print(f"  Top-1 Accuracy: {top1_accuracy:.3f}")
    print(f"  Top-3 Accuracy: {top3_accuracy:.3f}")
    print(f"  Top-5 Accuracy: {top5_accuracy:.3f}")
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title('Cross-modal Similarity Matrix\n(Test samples vs Train samples)')
    plt.xlabel('Train Samples')
    plt.ylabel('Test Samples')
    
    # Add labels
    plt.yticks(range(len(test_labels)), [f'Test {i} (Class {l})' for i, l in enumerate(test_labels)])
    
    plt.tight_layout()
    plt.savefig('miyawaki_similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return retrieval_results

def evaluate_classification_performance(train_repr, test_repr):
    """Evaluate classification performance using latent representations"""
    
    print("Evaluating classification performance...")
    
    # Test different representations
    representations = {
        'fMRI Latent': (train_repr['fmri_latent'], test_repr['fmri_latent']),
        'Stimulus Latent': (train_repr['stimulus_latent'], test_repr['stimulus_latent']),
        'Correlation': (train_repr['correlation'], test_repr['correlation'])
    }
    
    results = {}
    
    for name, (train_data, test_data) in representations.items():
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_data, train_repr['labels'])
        
        # Predict on test set
        predictions = knn.predict(test_data)
        accuracy = accuracy_score(test_repr['labels'], predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': test_repr['labels']
        }
        
        print(f"{name} Classification Accuracy: {accuracy:.3f}")
    
    return results

def analyze_correlation_patterns(train_repr, test_repr):
    """Analyze learned correlation patterns"""
    
    print("Analyzing correlation patterns...")
    
    # Compute class-wise correlation patterns
    unique_labels = np.unique(train_repr['labels'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Mean correlation per class
    ax = axes[0, 0]
    for label in unique_labels:
        mask = train_repr['labels'] == label
        mean_corr = train_repr['correlation'][mask].mean(axis=0)
        ax.plot(mean_corr, label=f'Class {label}', alpha=0.8)
    
    ax.set_title('Mean Correlation Patterns per Class')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Correlation Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Correlation between fMRI and stimulus latents
    ax = axes[0, 1]
    fmri_lat = train_repr['fmri_latent']
    stim_lat = train_repr['stimulus_latent']
    
    # Compute correlation for each dimension
    correlations = []
    for i in range(fmri_lat.shape[1]):
        corr = np.corrcoef(fmri_lat[:, i], stim_lat[:, i])[0, 1]
        correlations.append(corr)
    
    ax.plot(correlations)
    ax.set_title('fMRI-Stimulus Latent Correlation per Dimension')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)
    
    # 3. Class separability in correlation space
    ax = axes[1, 0]
    corr_data = train_repr['correlation']
    
    # Compute pairwise distances between class centroids
    centroids = []
    for label in unique_labels:
        mask = train_repr['labels'] == label
        centroid = corr_data[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    distances = np.zeros((len(unique_labels), len(unique_labels)))
    
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    im = ax.imshow(distances, cmap='viridis')
    ax.set_title('Inter-class Distances in Correlation Space')
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels)
    plt.colorbar(im, ax=ax)
    
    # 4. Distribution of correlation magnitudes
    ax = axes[1, 1]
    corr_magnitudes = np.linalg.norm(train_repr['correlation'], axis=1)
    
    for label in unique_labels:
        mask = train_repr['labels'] == label
        ax.hist(corr_magnitudes[mask], alpha=0.6, label=f'Class {label}', bins=10)
    
    ax.set_title('Distribution of Correlation Magnitudes')
    ax.set_xlabel('Correlation Magnitude')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('miyawaki_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation function"""
    
    # Load trained model
    model_path = 'miyawaki_dual_encoder.pth'
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Please run miyawaki_dual_encoder.py first to train the model.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_trained_model(model_path, device)
    
    # Load dataset
    filepath = Path("../dataset/miyawaki_structured_28x28.mat")
    dataset_dict = load_miyawaki_dataset(filepath)
    train_loader, test_loader = create_dataloaders(dataset_dict, batch_size=16)
    
    # Extract latent representations
    print("Extracting latent representations...")
    train_repr = extract_latent_representations(model, train_loader, device)
    test_repr = extract_latent_representations(model, test_loader, device)
    
    print(f"Train representations: {train_repr['fmri_latent'].shape}")
    print(f"Test representations: {test_repr['fmri_latent'].shape}")
    
    # Visualize latent space
    visualize_latent_space(train_repr, test_repr)
    
    # Evaluate retrieval performance
    retrieval_results = evaluate_retrieval_performance(train_repr, test_repr)
    
    # Evaluate classification performance
    classification_results = evaluate_classification_performance(train_repr, test_repr)
    
    # Analyze correlation patterns
    analyze_correlation_patterns(train_repr, test_repr)
    
    print("\nEvaluation completed!")
    print("Generated visualizations:")
    print("  - miyawaki_latent_space_tsne.png")
    print("  - miyawaki_similarity_matrix.png") 
    print("  - miyawaki_correlation_analysis.png")

if __name__ == "__main__":
    main()
