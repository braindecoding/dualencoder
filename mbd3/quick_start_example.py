#!/usr/bin/env python3
"""
MBD3 Quick Start Example
Demonstrate basic usage of EEG embeddings
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

def main():
    print("🧠 MBD3 EEG Embeddings - Quick Start")
    print("=" * 50)
    
    # Load embeddings
    print("📥 Loading embeddings...")
    with open('eeg_embeddings_enhanced_20250622_123559.pkl', 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']  # (4995, 512)
    labels = data['labels']          # (4995,)
    split_indices = data['split_indices']
    
    print(f"✅ Loaded {len(embeddings)} embeddings of {embeddings.shape[1]} dimensions")
    
    # Get splits
    train_idx = split_indices['train']
    test_idx = split_indices['test']
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
    print(f"📊 Data splits: {len(X_train)} train, {len(X_test)} test")
    
    # Example 1: Classification
    print("\n🎯 Example 1: Classification")
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"   Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Example 2: Similarity search
    print("\n🔍 Example 2: Similarity Search")
    query_idx = 100
    query_embedding = X_test[query_idx]
    query_label = y_test[query_idx]
    
    similarities = cosine_similarity([query_embedding], X_test)[0]
    top_indices = np.argsort(similarities)[-6:][::-1][1:]  # Top 5
    
    print(f"   Query: Sample {query_idx} (digit {query_label})")
    print("   Most similar samples:")
    for i, idx in enumerate(top_indices):
        sim_score = similarities[idx]
        sim_label = y_test[idx]
        print(f"      {i+1}. Sample {idx} (digit {sim_label}): {sim_score:.3f}")
    
    # Example 3: Basic statistics
    print("\n📊 Example 3: Embedding Statistics")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    
    print("\n🎉 Quick start completed successfully!")
    print("Ready to build amazing brain-computer interfaces! 🚀")

if __name__ == "__main__":
    main()
