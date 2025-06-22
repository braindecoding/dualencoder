#!/usr/bin/env python3
"""
Data Augmentation Pipeline for EEG Signals
- Noise injection
- Temporal shifting
- Electrode dropout
- Signal scaling
- Frequency domain augmentation
"""

import torch
import torch.nn as nn
import numpy as np
import random
from scipy import signal
from typing import Tuple, List, Optional

class EEGAugmentation:
    """
    Comprehensive EEG data augmentation pipeline
    """
    
    def __init__(self, 
                 noise_std: float = 0.05,
                 temporal_shift_range: int = 10,
                 electrode_dropout_prob: float = 0.1,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 freq_shift_range: float = 2.0,
                 augment_prob: float = 0.7):
        """
        Initialize augmentation parameters
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            temporal_shift_range: Maximum temporal shift in samples
            electrode_dropout_prob: Probability of dropping each electrode
            scale_range: Range for amplitude scaling
            freq_shift_range: Range for frequency shifting (Hz)
            augment_prob: Probability of applying augmentation
        """
        self.noise_std = noise_std
        self.temporal_shift_range = temporal_shift_range
        self.electrode_dropout_prob = electrode_dropout_prob
        self.scale_range = scale_range
        self.freq_shift_range = freq_shift_range
        self.augment_prob = augment_prob
        
    def add_gaussian_noise(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to EEG signal
        
        Args:
            eeg_signal: [channels, timepoints]
            
        Returns:
            Augmented signal with noise
        """
        noise = np.random.normal(0, self.noise_std, eeg_signal.shape)
        return eeg_signal + noise
    
    def temporal_shift(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Apply temporal shifting to EEG signal
        
        Args:
            eeg_signal: [channels, timepoints]
            
        Returns:
            Temporally shifted signal
        """
        shift = np.random.randint(-self.temporal_shift_range, self.temporal_shift_range + 1)
        
        if shift == 0:
            return eeg_signal
        
        shifted_signal = np.zeros_like(eeg_signal)
        
        if shift > 0:
            # Shift right (delay)
            shifted_signal[:, shift:] = eeg_signal[:, :-shift]
            # Pad with edge values
            shifted_signal[:, :shift] = eeg_signal[:, [0]]
        else:
            # Shift left (advance)
            shifted_signal[:, :shift] = eeg_signal[:, -shift:]
            # Pad with edge values
            shifted_signal[:, shift:] = eeg_signal[:, [-1]]
            
        return shifted_signal
    
    def electrode_dropout(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Randomly dropout electrodes
        
        Args:
            eeg_signal: [channels, timepoints]
            
        Returns:
            Signal with some electrodes zeroed out
        """
        n_channels = eeg_signal.shape[0]
        dropout_mask = np.random.random(n_channels) > self.electrode_dropout_prob
        
        # Ensure at least half of electrodes remain
        if np.sum(dropout_mask) < n_channels // 2:
            # Keep random half of electrodes
            keep_indices = np.random.choice(n_channels, n_channels // 2, replace=False)
            dropout_mask = np.zeros(n_channels, dtype=bool)
            dropout_mask[keep_indices] = True
        
        augmented_signal = eeg_signal.copy()
        augmented_signal[~dropout_mask] = 0
        
        return augmented_signal
    
    def amplitude_scaling(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Apply random amplitude scaling
        
        Args:
            eeg_signal: [channels, timepoints]
            
        Returns:
            Amplitude-scaled signal
        """
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return eeg_signal * scale_factor
    
    def frequency_shift(self, eeg_signal: np.ndarray, fs: float = 250.0) -> np.ndarray:
        """
        Apply frequency domain shifting
        
        Args:
            eeg_signal: [channels, timepoints]
            fs: Sampling frequency
            
        Returns:
            Frequency-shifted signal
        """
        freq_shift = np.random.uniform(-self.freq_shift_range, self.freq_shift_range)
        
        if abs(freq_shift) < 0.1:
            return eeg_signal
        
        augmented_signal = np.zeros_like(eeg_signal)
        
        for ch in range(eeg_signal.shape[0]):
            # Apply frequency shift using Hilbert transform
            analytic_signal = signal.hilbert(eeg_signal[ch])
            
            # Create frequency shift
            t = np.arange(len(analytic_signal)) / fs
            shift_factor = np.exp(1j * 2 * np.pi * freq_shift * t)
            
            # Apply shift and take real part
            shifted_analytic = analytic_signal * shift_factor
            augmented_signal[ch] = np.real(shifted_analytic)
        
        return augmented_signal
    
    def channel_shuffle(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Randomly shuffle some channels (simulate electrode placement variation)
        
        Args:
            eeg_signal: [channels, timepoints]
            
        Returns:
            Channel-shuffled signal
        """
        n_channels = eeg_signal.shape[0]
        
        # Shuffle only a subset of channels
        n_shuffle = np.random.randint(1, max(2, n_channels // 3))
        shuffle_indices = np.random.choice(n_channels, n_shuffle, replace=False)
        
        augmented_signal = eeg_signal.copy()
        shuffled_channels = augmented_signal[shuffle_indices].copy()
        np.random.shuffle(shuffled_channels)
        augmented_signal[shuffle_indices] = shuffled_channels
        
        return augmented_signal
    
    def apply_augmentation(self, eeg_signal: np.ndarray, 
                          augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply random combination of augmentations
        
        Args:
            eeg_signal: [channels, timepoints]
            augmentation_types: List of augmentation types to apply
            
        Returns:
            Augmented EEG signal
        """
        if np.random.random() > self.augment_prob:
            return eeg_signal
        
        if augmentation_types is None:
            augmentation_types = ['noise', 'temporal_shift', 'electrode_dropout', 
                                'amplitude_scaling', 'frequency_shift']
        
        # Randomly select augmentations to apply
        n_augmentations = np.random.randint(1, min(4, len(augmentation_types) + 1))
        selected_augmentations = np.random.choice(augmentation_types, n_augmentations, replace=False)
        
        augmented_signal = eeg_signal.copy()
        
        for aug_type in selected_augmentations:
            if aug_type == 'noise':
                augmented_signal = self.add_gaussian_noise(augmented_signal)
            elif aug_type == 'temporal_shift':
                augmented_signal = self.temporal_shift(augmented_signal)
            elif aug_type == 'electrode_dropout':
                augmented_signal = self.electrode_dropout(augmented_signal)
            elif aug_type == 'amplitude_scaling':
                augmented_signal = self.amplitude_scaling(augmented_signal)
            elif aug_type == 'frequency_shift':
                augmented_signal = self.frequency_shift(augmented_signal)
            elif aug_type == 'channel_shuffle':
                augmented_signal = self.channel_shuffle(augmented_signal)
        
        return augmented_signal

class AugmentedDataLoader:
    """
    Data loader with real-time augmentation
    """
    
    def __init__(self, eeg_data, stim_data, labels, augmentation: EEGAugmentation, 
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize augmented data loader
        
        Args:
            eeg_data: EEG signals [N, channels, timepoints]
            stim_data: Stimulus images
            labels: Labels
            augmentation: EEGAugmentation instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.eeg_data = eeg_data
        self.stim_data = stim_data
        self.labels = labels
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n_samples = len(eeg_data)
        self.indices = np.arange(self.n_samples)
        
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Get batch data
            batch_eeg = self.eeg_data[batch_indices]
            batch_stim = [self.stim_data[i] for i in batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # Apply augmentation to EEG signals
            augmented_eeg = []
            for eeg_signal in batch_eeg:
                aug_signal = self.augmentation.apply_augmentation(eeg_signal)
                augmented_eeg.append(aug_signal)
            
            augmented_eeg = np.array(augmented_eeg)
            
            yield augmented_eeg, batch_stim, batch_labels

def test_augmentation():
    """
    Test augmentation pipeline
    """
    print("🧪 Testing EEG Augmentation Pipeline...")
    
    # Create test data
    batch_size = 4
    n_channels = 14
    seq_len = 256
    
    test_eeg = np.random.randn(batch_size, n_channels, seq_len)
    test_stim = [f"image_{i}" for i in range(batch_size)]
    test_labels = np.array([0, 1, 2, 3])
    
    print(f"📊 Original EEG shape: {test_eeg.shape}")
    print(f"📊 Original EEG range: [{test_eeg.min():.3f}, {test_eeg.max():.3f}]")
    
    # Test individual augmentations
    augmenter = EEGAugmentation(
        noise_std=0.05,
        temporal_shift_range=10,
        electrode_dropout_prob=0.1,
        scale_range=(0.8, 1.2),
        freq_shift_range=2.0,
        augment_prob=1.0  # Always augment for testing
    )
    
    # Test each augmentation
    print("\n🔧 Testing individual augmentations:")
    
    original_signal = test_eeg[0]
    
    # Noise
    noisy_signal = augmenter.add_gaussian_noise(original_signal)
    print(f"   Noise: {np.std(noisy_signal - original_signal):.4f} std difference")
    
    # Temporal shift
    shifted_signal = augmenter.temporal_shift(original_signal)
    print(f"   Temporal shift: Signal modified")
    
    # Electrode dropout
    dropout_signal = augmenter.electrode_dropout(original_signal)
    zero_channels = np.sum(np.all(dropout_signal == 0, axis=1))
    print(f"   Electrode dropout: {zero_channels}/{n_channels} channels zeroed")
    
    # Amplitude scaling
    scaled_signal = augmenter.amplitude_scaling(original_signal)
    scale_factor = np.mean(scaled_signal) / np.mean(original_signal)
    print(f"   Amplitude scaling: {scale_factor:.3f}x scale factor")
    
    # Combined augmentation
    augmented_signal = augmenter.apply_augmentation(original_signal)
    print(f"   Combined augmentation: Shape {augmented_signal.shape}")
    
    # Test data loader
    print("\n🔧 Testing augmented data loader:")
    
    augmented_loader = AugmentedDataLoader(
        test_eeg, test_stim, test_labels, augmenter, batch_size=2, shuffle=True
    )
    
    for i, (aug_eeg, stim, labels) in enumerate(augmented_loader):
        print(f"   Batch {i+1}: EEG {aug_eeg.shape}, Stim {len(stim)}, Labels {labels.shape}")
        if i >= 1:  # Test first 2 batches
            break
    
    print("✅ Augmentation pipeline test completed successfully!")
    
    return augmenter

if __name__ == "__main__":
    test_augmentation()
