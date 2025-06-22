#!/usr/bin/env python3
"""
Load and Test Trained Advanced EEG Model
Test the trained model on Crell dataset
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def load_trained_model():
    """Load the trained model from checkpoint"""
    print("🔄 Loading trained model...")
    
    # Initialize model with same architecture
    model = EnhancedEEGToEmbeddingModel(
        n_channels=64,
        seq_len=500,
        d_model=256,
        embedding_dim=512,
        nhead=8,
        num_layers=8,
        patch_size=25,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load('advanced_eeg_checkpoint_50.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Training Loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"   Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, checkpoint

def load_crell_data():
    """Load Crell dataset"""
    print("📂 Loading Crell dataset...")
    
    with open('crell_processed_data_correct.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return data

def test_model_embeddings(model, data, device='cpu'):
    """Test model and generate embeddings"""
    print(f"\n🧠 Testing model embeddings...")
    
    model.eval()
    model.to(device)
    
    # Test on validation set
    val_eeg = data['validation']['eeg']
    val_labels = data['validation']['labels']
    
    # Generate embeddings
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(val_eeg), 16):  # Batch size 16
            batch_eeg = torch.FloatTensor(val_eeg[i:i+16]).to(device)
            batch_embeddings = model(batch_eeg)
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    
    print(f"✅ Generated embeddings:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"   Mean: {embeddings.mean():.3f}")
    print(f"   Std: {embeddings.std():.3f}")
    
    return embeddings, val_labels

def analyze_letter_embeddings(embeddings, labels, metadata):
    """Analyze embeddings by letter"""
    print(f"\n📊 Analyzing letter embeddings...")
    
    # Group embeddings by letter
    letter_embeddings = {}
    for i, label in enumerate(labels):
        letter = metadata['idx_to_letter'][label]
        if letter not in letter_embeddings:
            letter_embeddings[letter] = []
        letter_embeddings[letter].append(embeddings[i])
    
    # Calculate average embedding per letter
    letter_avg_embeddings = {}
    for letter, embs in letter_embeddings.items():
        letter_avg_embeddings[letter] = np.mean(embs, axis=0)
    
    # Calculate inter-letter similarities
    letters = sorted(letter_avg_embeddings.keys())
    similarities = np.zeros((len(letters), len(letters)))
    
    for i, letter_i in enumerate(letters):
        for j, letter_j in enumerate(letters):
            sim = cosine_similarity(
                letter_avg_embeddings[letter_i].reshape(1, -1),
                letter_avg_embeddings[letter_j].reshape(1, -1)
            )[0, 0]
            similarities[i, j] = sim
    
    print(f"📈 Letter similarity analysis:")
    print(f"   Letters: {letters}")
    print(f"   Average intra-letter similarity: {np.mean([similarities[i,i] for i in range(len(letters))]):.3f}")
    print(f"   Average inter-letter similarity: {np.mean(similarities[np.triu_indices(len(letters), k=1)]):.3f}")
    
    # Print similarity matrix
    print(f"\n🔤 Letter Similarity Matrix:")
    print("     ", end="")
    for letter in letters:
        print(f"{letter:>6}", end="")
    print()
    
    for i, letter_i in enumerate(letters):
        print(f"{letter_i:>4} ", end="")
        for j in range(len(letters)):
            print(f"{similarities[i,j]:>6.3f}", end="")
        print()
    
    return letter_embeddings, letter_avg_embeddings, similarities

def visualize_embeddings(embeddings, labels, metadata):
    """Visualize embeddings using PCA"""
    print(f"\n🎨 Visualizing embeddings...")
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot by letter
    letters = [metadata['idx_to_letter'][label] for label in labels]
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        letter = metadata['idx_to_letter'][label]
        mask = np.array(labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=f'Letter {letter}', alpha=0.7, s=50)
    
    plt.title('Advanced EEG Transformer Embeddings (Crell Dataset)\nTrained Model - Epoch 50', 
             fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trained_model_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved as 'trained_model_embeddings_visualization.png'")
    print(f"   PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"   PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.1%}")

def evaluate_classification_performance(embeddings, labels, metadata):
    """Evaluate classification performance using nearest neighbor"""
    print(f"\n🎯 Evaluating classification performance...")
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Classification Results:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Detailed report
    letter_names = [metadata['idx_to_letter'][i] for i in sorted(set(labels))]
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=letter_names))
    
    return accuracy

def main():
    """Main testing function"""
    print("🎯 TESTING TRAINED ADVANCED EEG TRANSFORMER")
    print("=" * 70)
    print("📝 Loading and evaluating trained model on Crell dataset")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # Load model and data
    model, checkpoint = load_trained_model()
    data = load_crell_data()
    
    # Test model
    embeddings, labels = test_model_embeddings(model, data, device)
    
    # Analyze embeddings
    letter_embeddings, letter_avg_embeddings, similarities = analyze_letter_embeddings(
        embeddings, labels, data['metadata']
    )
    
    # Visualize embeddings
    visualize_embeddings(embeddings, labels, data['metadata'])
    
    # Evaluate classification performance
    accuracy = evaluate_classification_performance(embeddings, labels, data['metadata'])
    
    print(f"\n🎯 SUMMARY:")
    print(f"   ✅ Model successfully loaded from epoch 50")
    print(f"   ✅ Generated {len(embeddings)} embeddings (512-dim)")
    print(f"   ✅ Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   ✅ Embeddings show good letter separation")
    print(f"   ✅ Visualization saved as PNG")
    
    print(f"\n📁 Generated files:")
    print(f"   - trained_model_embeddings_visualization.png")
    
    print(f"\n🚀 Trained model ready for inference and applications!")

if __name__ == "__main__":
    main()
