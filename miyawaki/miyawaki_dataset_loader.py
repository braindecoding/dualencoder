#!/usr/bin/env python3
"""
Dataset loader dan analisis khusus untuk Miyawaki dataset
Fokus pada pemahaman struktur data dan implementasi data loader
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path

class MiyawakiDataset(Dataset):
    """PyTorch Dataset untuk Miyawaki data"""
    
    def __init__(self, fmri_data, stim_data, labels, transform=None):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stim_data = torch.FloatTensor(stim_data)
        self.labels = torch.LongTensor(labels.flatten())
        self.transform = transform
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        stim = self.stim_data[idx].reshape(28, 28)  # Reshape ke 28x28
        label = self.labels[idx]
        
        if self.transform:
            stim = self.transform(stim)
            
        return {
            'fmri': fmri,
            'stimulus': stim,
            'label': label,
            'index': idx
        }

def load_miyawaki_dataset(filepath):
    """Load dan preprocess Miyawaki dataset"""
    print("Loading Miyawaki dataset...")
    
    try:
        data = loadmat(filepath)
        
        # Extract data
        fmri_train = data['fmriTrn']  # (107, 967)
        fmri_test = data['fmriTest']   # (12, 967)
        stim_train = data['stimTrn']   # (107, 784)
        stim_test = data['stimTest']   # (12, 784)
        label_train = data['labelTrn'] # (107, 1)
        label_test = data['labelTest'] # (12, 1)
        
        # Optional: train/test indices
        train_indices = data.get('train_indices', None)
        test_indices = data.get('test_indices', None)
        
        print(f"Training data: fMRI {fmri_train.shape}, Stimuli {stim_train.shape}, Labels {label_train.shape}")
        print(f"Test data: fMRI {fmri_test.shape}, Stimuli {stim_test.shape}, Labels {label_test.shape}")
        
        # Analyze labels
        unique_labels = np.unique(label_train)
        print(f"Unique labels: {unique_labels}")
        
        for label in unique_labels:
            count_train = np.sum(label_train == label)
            count_test = np.sum(label_test == label)
            print(f"  Label {label}: {count_train} train, {count_test} test")
        
        # Normalize stimuli (sudah dalam range 0-1)
        print(f"Stimuli range: [{stim_train.min():.3f}, {stim_train.max():.3f}]")
        
        # Normalize fMRI (z-score normalization)
        fmri_mean = fmri_train.mean(axis=0, keepdims=True)
        fmri_std = fmri_train.std(axis=0, keepdims=True)
        
        fmri_train_norm = (fmri_train - fmri_mean) / (fmri_std + 1e-8)
        fmri_test_norm = (fmri_test - fmri_mean) / (fmri_std + 1e-8)
        
        print(f"fMRI normalized - Train: [{fmri_train_norm.min():.3f}, {fmri_train_norm.max():.3f}]")
        print(f"fMRI normalized - Test: [{fmri_test_norm.min():.3f}, {fmri_test_norm.max():.3f}]")
        
        return {
            'train': {
                'fmri': fmri_train_norm,
                'stimuli': stim_train,
                'labels': label_train
            },
            'test': {
                'fmri': fmri_test_norm,
                'stimuli': stim_test,
                'labels': label_test
            },
            'metadata': {
                'fmri_mean': fmri_mean,
                'fmri_std': fmri_std,
                'unique_labels': unique_labels,
                'train_indices': train_indices,
                'test_indices': test_indices
            }
        }
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def visualize_miyawaki_samples(dataset_dict, n_samples=3):
    """Visualisasi sample data dari Miyawaki dataset"""
    
    train_data = dataset_dict['train']
    unique_labels = dataset_dict['metadata']['unique_labels']
    
    fig, axes = plt.subplots(len(unique_labels), n_samples + 1, 
                            figsize=(15, len(unique_labels) * 3))
    
    for i, label in enumerate(unique_labels):
        # Find samples for this label
        label_mask = train_data['labels'].flatten() == label
        label_indices = np.where(label_mask)[0]
        
        # Plot fMRI pattern (mean for this class)
        fmri_mean = train_data['fmri'][label_mask].mean(axis=0)
        axes[i, 0].plot(fmri_mean)
        axes[i, 0].set_title(f'Label {label}\nMean fMRI Pattern')
        axes[i, 0].set_xlabel('fMRI Feature')
        axes[i, 0].set_ylabel('Signal')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot sample stimuli
        sample_indices = label_indices[:n_samples]
        for j, idx in enumerate(sample_indices):
            if j < n_samples:
                stim = train_data['stimuli'][idx].reshape(28, 28)
                axes[i, j+1].imshow(stim, cmap='gray')
                axes[i, j+1].set_title(f'Sample {j+1}')
                axes[i, j+1].axis('off')
    
    plt.suptitle('Miyawaki Dataset - fMRI Patterns and Visual Stimuli')
    plt.tight_layout()
    plt.savefig('miyawaki_samples_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_fmri_patterns(dataset_dict):
    """Analisis pola fMRI untuk setiap class"""
    
    train_data = dataset_dict['train']
    unique_labels = dataset_dict['metadata']['unique_labels']
    
    print("\nAnalyzing fMRI patterns...")
    
    # Compute class-wise statistics
    class_stats = {}
    for label in unique_labels:
        label_mask = train_data['labels'].flatten() == label
        fmri_class = train_data['fmri'][label_mask]
        
        class_stats[label] = {
            'mean': fmri_class.mean(axis=0),
            'std': fmri_class.std(axis=0),
            'n_samples': fmri_class.shape[0]
        }
        
        print(f"Label {label}: {fmri_class.shape[0]} samples")
        print(f"  fMRI mean: {fmri_class.mean():.4f} ± {fmri_class.std():.4f}")
    
    # Correlation analysis between classes
    print("\nClass correlation analysis:")
    class_means = np.array([class_stats[label]['mean'] for label in unique_labels])
    corr_matrix = np.corrcoef(class_means)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title('fMRI Pattern Correlation Between Classes')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.xticks(range(len(unique_labels)), unique_labels)
    plt.yticks(range(len(unique_labels)), unique_labels)
    
    # Add correlation values
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            plt.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('miyawaki_fmri_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return class_stats

def create_dataloaders(dataset_dict, batch_size=8):
    """Create PyTorch DataLoaders"""
    
    train_dataset = MiyawakiDataset(
        dataset_dict['train']['fmri'],
        dataset_dict['train']['stimuli'],
        dataset_dict['train']['labels']
    )
    
    test_dataset = MiyawakiDataset(
        dataset_dict['test']['fmri'],
        dataset_dict['test']['stimuli'],
        dataset_dict['test']['labels']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, test_loader

def test_dataloader(train_loader):
    """Test DataLoader functionality"""
    print("\nTesting DataLoader...")
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  fMRI shape: {batch['fmri'].shape}")
        print(f"  Stimulus shape: {batch['stimulus'].shape}")
        print(f"  Labels: {batch['label']}")
        print(f"  Indices: {batch['index']}")
        
        if i >= 2:  # Only show first 3 batches
            break

def main():
    """Main function untuk test Miyawaki dataset"""
    
    # Load dataset
    filepath = Path("../dataset/miyawaki_structured_28x28.mat")
    
    if not filepath.exists():
        print(f"Dataset file not found: {filepath}")
        return
    
    # Load and analyze
    dataset_dict = load_miyawaki_dataset(filepath)
    
    if dataset_dict is None:
        print("Failed to load dataset")
        return
    
    # Visualize samples
    visualize_miyawaki_samples(dataset_dict)
    
    # Analyze fMRI patterns
    class_stats = analyze_fmri_patterns(dataset_dict)
    
    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(dataset_dict, batch_size=4)
    
    # Test DataLoader
    test_dataloader(train_loader)
    
    print("\nMiyawaki dataset analysis completed!")
    print("Dataset ready untuk dual encoder training.")

if __name__ == "__main__":
    main()
