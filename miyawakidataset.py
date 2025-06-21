#!/usr/bin/env python3
"""
CORRECTED Miyawaki Dataset Loader
Fixes data leaking and uses proper .mat file format
"""

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from pathlib import Path

class MiyawakiDataset(Dataset):
    """
    Corrected Miyawaki Dataset Loader
    - Fixes data leaking in normalization
    - Uses proper .mat file format
    - Consistent with proven implementation
    """

    def __init__(self, fmri_data, stim_data, labels, fmri_stats=None, transform=None):
        """
        Args:
            fmri_data: fMRI signals array [N, 967]
            stim_data: Stimulus images array [N, 784] or [N, 28, 28]
            labels: Labels array [N, 1]
            fmri_stats: Dict with 'mean' and 'std' from training data (to prevent leaking)
            transform: Optional transforms
        """
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stim_data = torch.FloatTensor(stim_data)
        self.labels = torch.LongTensor(labels.flatten())
        self.transform = transform

        # Apply normalization using provided statistics (no data leaking)
        if fmri_stats is not None:
            fmri_mean = torch.FloatTensor(fmri_stats['mean'])
            fmri_std = torch.FloatTensor(fmri_stats['std'])
            self.fmri_data = (self.fmri_data - fmri_mean) / (fmri_std + 1e-8)

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]  # [967]
        stim = self.stim_data[idx]  # [784] or [28, 28]
        label = self.labels[idx]

        # Reshape stimulus to 28x28 if needed
        if len(stim.shape) == 1 and stim.shape[0] == 784:
            stim = stim.reshape(28, 28)

        if self.transform:
            stim = self.transform(stim)

        return {
            'fmri': fmri,
            'stimulus': stim,
            'label': label,
            'index': idx
        }

    def __len__(self):
        return len(self.fmri_data)

def load_miyawaki_dataset_corrected(filepath):
    """
    Load Miyawaki dataset from .mat file with proper normalization
    Prevents data leaking by using only training statistics
    """
    print(f"Loading Miyawaki dataset from {filepath}...")

    try:
        data = loadmat(filepath)

        # Extract data
        fmri_train = data['fmriTrn']    # (107, 967)
        fmri_test = data['fmriTest']    # (12, 967)
        stim_train = data['stimTrn']    # (107, 784)
        stim_test = data['stimTest']    # (12, 784)
        label_train = data['labelTrn']  # (107, 1)
        label_test = data['labelTest']  # (12, 1)

        print(f"Loaded: Train {fmri_train.shape}, Test {fmri_test.shape}")

        # ✅ CORRECT: Compute normalization statistics ONLY from training data
        fmri_mean = fmri_train.mean(axis=0, keepdims=True)
        fmri_std = fmri_train.std(axis=0, keepdims=True)

        fmri_stats = {
            'mean': fmri_mean,
            'std': fmri_std
        }

        print(f"fMRI normalization - Mean: {fmri_mean.mean():.4f}, Std: {fmri_std.mean():.4f}")

        # Create datasets with proper normalization
        train_dataset = MiyawakiDataset(
            fmri_train, stim_train, label_train,
            fmri_stats=fmri_stats
        )

        test_dataset = MiyawakiDataset(
            fmri_test, stim_test, label_test,
            fmri_stats=fmri_stats  # ✅ Use same stats for test (no leaking)
        )

        return train_dataset, test_dataset, fmri_stats

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

def create_dataloaders_corrected(train_dataset, test_dataset, batch_size=8):
    """Create DataLoaders with proper settings"""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # ✅ Shuffle training data
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # ✅ Don't shuffle test data
        drop_last=False
    )

    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, test_loader

# ✅ CORRECT USAGE:
if __name__ == "__main__":
    # Load dataset from correct .mat file
    filepath = Path("dataset/miyawaki_structured_28x28.mat")

    if filepath.exists():
        train_dataset, test_dataset, fmri_stats = load_miyawaki_dataset_corrected(filepath)

        if train_dataset is not None:
            train_loader, test_loader = create_dataloaders_corrected(
                train_dataset, test_dataset, batch_size=8  # Smaller batch for small dataset
            )

            # Test loading
            print("\nTesting DataLoader:")
            for i, batch in enumerate(train_loader):
                print(f"Batch {i+1}:")
                print(f"  fMRI: {batch['fmri'].shape}")
                print(f"  Stimulus: {batch['stimulus'].shape}")
                print(f"  Labels: {batch['label']}")
                if i >= 1:  # Only show first 2 batches
                    break
        else:
            print("Failed to load dataset")
    else:
        print(f"Dataset file not found: {filepath}")