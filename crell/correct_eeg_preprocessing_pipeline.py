#!/usr/bin/env python3
"""
Correct EEG Preprocessing Pipeline
Apply preprocessing steps in the CORRECT ORDER from raw data:
1. Bandpass filtering (FIRST - on raw data)
2. Artifact removal 
3. Epoching
4. Baseline correction
5. Normalization (LAST)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append('..')
from load_mindbigdata_eeg import load_mindbigdata_eeg

def step1_bandpass_filter(raw_signal, fs=256, lowcut=0.5, highcut=50, order=4):
    """
    STEP 1: Apply bandpass filter to RAW EEG signal
    This MUST be done FIRST on raw data before any other processing
    
    Args:
        raw_signal: 1D numpy array of RAW EEG data
        fs: Sampling frequency (Hz)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Bandpass filtered signal
    """
    print(f"   Step 1: Bandpass filtering ({lowcut}-{highcut} Hz)...")
    
    # Design Butterworth bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create second-order sections for stability
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    # Apply zero-phase filtering (forward and backward)
    filtered_signal = signal.sosfiltfilt(sos, raw_signal)
    
    return filtered_signal

def step2_artifact_detection(filtered_signal, fs=256):
    """
    STEP 2: Detect artifacts in filtered signal
    Apply AFTER filtering but BEFORE epoching
    
    Args:
        filtered_signal: 1D numpy array of filtered EEG data
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary with artifact detection results
    """
    print(f"   Step 2: Artifact detection...")
    
    artifacts = {
        'amplitude': False,
        'gradient': False,
        'flatline': False,
        'is_artifact': False,
        'artifact_samples': []
    }
    
    # Use adaptive thresholds based on signal statistics
    signal_std = np.std(filtered_signal)
    signal_mean = np.mean(np.abs(filtered_signal))
    
    # 1. Amplitude artifacts (signal too large)
    # Use 6 standard deviations as threshold (very conservative)
    amplitude_threshold = signal_mean + 6 * signal_std
    amplitude_mask = np.abs(filtered_signal) > amplitude_threshold
    
    if np.any(amplitude_mask):
        artifacts['amplitude'] = True
        artifacts['artifact_samples'].extend(np.where(amplitude_mask)[0].tolist())
    
    # 2. Gradient artifacts (sudden jumps)
    gradient = np.diff(filtered_signal)
    gradient_std = np.std(gradient)
    gradient_threshold = 6 * gradient_std  # 6 standard deviations
    gradient_mask = np.abs(gradient) > gradient_threshold
    
    if np.any(gradient_mask):
        artifacts['gradient'] = True
        artifacts['artifact_samples'].extend((np.where(gradient_mask)[0] + 1).tolist())
    
    # 3. Flatline detection (signal too constant)
    if signal_std < 0.001:  # Completely flat signal
        artifacts['flatline'] = True
    
    # Overall artifact decision - be conservative
    # Only reject if multiple severe artifacts OR completely flat
    severe_amplitude = np.sum(amplitude_mask) > len(filtered_signal) * 0.1  # >10% of samples
    severe_gradient = np.sum(gradient_mask) > len(gradient) * 0.1  # >10% of samples
    
    artifacts['is_artifact'] = (severe_amplitude and severe_gradient) or artifacts['flatline']
    artifacts['artifact_samples'] = list(set(artifacts['artifact_samples']))
    
    return artifacts

def step3_epoching(filtered_signal, fs=256, epoch_length=1.0, stimulus_onset=None):
    """
    STEP 3: Extract epochs from filtered signal
    Apply AFTER filtering and artifact detection
    
    Args:
        filtered_signal: 1D numpy array of filtered EEG data
        fs: Sampling frequency (Hz)
        epoch_length: Length of epoch in seconds
        stimulus_onset: Sample index of stimulus onset (if known)
    
    Returns:
        Epoched signal or original if no stimulus timing
    """
    print(f"   Step 3: Epoching...")
    
    target_samples = int(epoch_length * fs)  # e.g., 1.0 sec * 256 Hz = 256 samples
    
    if stimulus_onset is not None:
        # If we have stimulus timing, extract around stimulus
        pre_stim_samples = int(0.2 * fs)  # 200ms before
        post_stim_samples = target_samples - pre_stim_samples
        
        start_idx = stimulus_onset - pre_stim_samples
        end_idx = stimulus_onset + post_stim_samples
        
        if start_idx >= 0 and end_idx < len(filtered_signal):
            epoch = filtered_signal[start_idx:end_idx]
            return epoch
    
    # If no stimulus timing, extract from center or standardize length
    if len(filtered_signal) == target_samples:
        return filtered_signal
    elif len(filtered_signal) > target_samples:
        # Extract from center to preserve temporal structure
        start_idx = (len(filtered_signal) - target_samples) // 2
        epoch = filtered_signal[start_idx:start_idx + target_samples]
        return epoch
    else:
        # Pad if too short
        pad_width = target_samples - len(filtered_signal)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        epoch = np.pad(filtered_signal, (pad_left, pad_right), mode='constant')
        return epoch

def step4_baseline_correction(epoch, fs=256, baseline_duration=0.2):
    """
    STEP 4: Apply baseline correction to epoch
    Apply AFTER epoching but BEFORE normalization
    
    Args:
        epoch: 1D numpy array of epoched EEG data
        fs: Sampling frequency (Hz)
        baseline_duration: Duration of baseline period (seconds)
    
    Returns:
        Baseline-corrected epoch
    """
    print(f"   Step 4: Baseline correction...")
    
    baseline_samples = int(baseline_duration * fs)
    
    if len(epoch) < baseline_samples:
        # If epoch too short, use first 20% as baseline
        baseline_samples = max(1, len(epoch) // 5)
    
    # Calculate baseline (mean of pre-stimulus period)
    baseline = np.mean(epoch[:baseline_samples])
    
    # Subtract baseline from entire epoch
    corrected_epoch = epoch - baseline
    
    return corrected_epoch

def step5_normalization(baseline_corrected_signal, method='zscore'):
    """
    STEP 5: Apply normalization (FINAL STEP)
    Apply AFTER all other preprocessing steps
    
    Args:
        baseline_corrected_signal: 1D numpy array of baseline-corrected EEG data
        method: Normalization method ('zscore', 'robust', 'minmax')
    
    Returns:
        Normalized signal
    """
    print(f"   Step 5: Normalization ({method})...")
    
    if method == 'zscore':
        # Standard z-score normalization
        normalized = zscore(baseline_corrected_signal)
    
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(baseline_corrected_signal)
        mad = np.median(np.abs(baseline_corrected_signal - median))
        if mad > 0:
            normalized = (baseline_corrected_signal - median) / mad
        else:
            normalized = baseline_corrected_signal - median
    
    elif method == 'minmax':
        # Min-max normalization to [-1, 1]
        sig_min, sig_max = np.min(baseline_corrected_signal), np.max(baseline_corrected_signal)
        if sig_max > sig_min:
            normalized = 2 * (baseline_corrected_signal - sig_min) / (sig_max - sig_min) - 1
        else:
            normalized = np.zeros_like(baseline_corrected_signal)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def correct_preprocessing_pipeline(raw_signal, fs=256, target_length=256):
    """
    Complete CORRECT preprocessing pipeline for single EEG signal
    Apply steps in the CORRECT ORDER:
    1. Bandpass filtering (on raw data)
    2. Artifact detection
    3. Epoching
    4. Baseline correction
    5. Normalization
    
    Args:
        raw_signal: 1D numpy array of RAW EEG data
        fs: Sampling frequency (Hz)
        target_length: Target signal length after preprocessing
    
    Returns:
        Processed signal or None if rejected due to artifacts
    """
    try:
        print(f"🔧 Processing signal (length: {len(raw_signal)})...")
        
        # STEP 1: Bandpass filtering (FIRST - on raw data)
        filtered_signal = step1_bandpass_filter(raw_signal, fs=fs)
        
        # STEP 2: Artifact detection (on filtered data)
        artifacts = step2_artifact_detection(filtered_signal, fs=fs)
        if artifacts['is_artifact']:
            print(f"   ❌ Signal rejected due to artifacts")
            return None
        
        # STEP 3: Epoching (standardize length)
        epoched_signal = step3_epoching(filtered_signal, fs=fs, 
                                       epoch_length=target_length/fs)
        
        # STEP 4: Baseline correction
        baseline_corrected = step4_baseline_correction(epoched_signal, fs=fs)
        
        # STEP 5: Normalization (FINAL STEP)
        normalized_signal = step5_normalization(baseline_corrected, method='zscore')
        
        print(f"   ✅ Signal processed successfully")
        return normalized_signal
        
    except Exception as e:
        print(f"   ❌ Error processing signal: {e}")
        return None

def process_organized_data_correctly(organized_data, max_samples_per_digit=500):
    """
    Apply CORRECT preprocessing pipeline to organized EEG data
    
    Args:
        organized_data: Dictionary of organized RAW EEG data
        max_samples_per_digit: Maximum samples to process per digit
    
    Returns:
        Correctly processed data and statistics
    """
    print("🧠 APPLYING CORRECT EEG PREPROCESSING PIPELINE")
    print("=" * 70)
    print("📋 Processing Order:")
    print("   1. Bandpass filtering (0.5-50 Hz) - on RAW data")
    print("   2. Artifact detection and removal")
    print("   3. Epoching (length standardization)")
    print("   4. Baseline correction")
    print("   5. Normalization (z-score) - FINAL step")
    print("=" * 70)
    
    correctly_processed_data = {}
    processing_stats = {
        'total_signals': 0,
        'processed_signals': 0,
        'rejected_artifacts': 0,
        'rejection_rate': 0.0,
        'electrodes': [],
        'digits': []
    }
    
    electrodes = list(organized_data.keys())
    processing_stats['electrodes'] = electrodes
    
    print(f"📊 Processing {len(electrodes)} electrodes...")
    
    for electrode in tqdm(electrodes, desc="Processing electrodes"):
        correctly_processed_data[electrode] = {}
        
        for digit in organized_data[electrode]:
            if digit == -1:  # Skip resting state for now
                continue
            
            if digit not in processing_stats['digits']:
                processing_stats['digits'].append(digit)
            
            print(f"\n🔧 Processing {electrode} - Digit {digit}...")
            
            raw_signals = organized_data[electrode][digit]
            processed_signals = []
            
            # Limit number of signals to process
            signals_to_process = raw_signals[:max_samples_per_digit]
            
            for i, raw_signal in enumerate(tqdm(signals_to_process, 
                                              desc=f"  {electrode}-{digit}", 
                                              leave=False)):
                processing_stats['total_signals'] += 1
                
                # Apply CORRECT preprocessing pipeline
                processed_signal = correct_preprocessing_pipeline(raw_signal)
                
                if processed_signal is not None:
                    processed_signals.append(processed_signal)
                    processing_stats['processed_signals'] += 1
                else:
                    processing_stats['rejected_artifacts'] += 1
            
            correctly_processed_data[electrode][digit] = processed_signals
            
            print(f"     ✅ Processed: {len(processed_signals)}/{len(signals_to_process)} signals")
    
    # Calculate rejection rate
    if processing_stats['total_signals'] > 0:
        processing_stats['rejection_rate'] = processing_stats['rejected_artifacts'] / processing_stats['total_signals']
    
    print(f"\n📊 CORRECT PREPROCESSING STATISTICS:")
    print(f"   Total signals processed: {processing_stats['total_signals']:,}")
    print(f"   Successfully processed: {processing_stats['processed_signals']:,}")
    print(f"   Rejected due to artifacts: {processing_stats['rejected_artifacts']:,}")
    print(f"   Rejection rate: {processing_stats['rejection_rate']:.1%}")
    
    return correctly_processed_data, processing_stats

def main():
    """Main function to apply CORRECT preprocessing pipeline"""
    print("🧠 CORRECT EEG PREPROCESSING PIPELINE")
    print("=" * 70)
    print("🎯 CORRECT ORDER:")
    print("   1. Bandpass filtering (on RAW data)")
    print("   2. Artifact removal")
    print("   3. Epoching")
    print("   4. Baseline correction")
    print("   5. Normalization (FINAL)")
    print("=" * 70)
    
    print("📂 Loading RAW EEG data...")
    result = load_mindbigdata_eeg()
    if result is None:
        print("❌ Failed to load EEG data")
        return
    
    organized_data, eeg_records = result
    
    print(f"✅ Loaded {len(eeg_records):,} RAW EEG records")
    
    # Apply CORRECT preprocessing pipeline
    correctly_processed_data, stats = process_organized_data_correctly(
        organized_data, max_samples_per_digit=500)
    
    # Save correctly preprocessed data
    print(f"\n💾 Saving correctly preprocessed data...")
    
    correct_dataset = {
        'correctly_processed_eeg_data': correctly_processed_data,
        'processing_statistics': stats,
        'preprocessing_steps_order': [
            '1. Bandpass filtering (0.5-50 Hz) - applied to RAW data',
            '2. Artifact detection and removal',
            '3. Epoching (length standardization to 256 samples)',
            '4. Baseline correction (subtract first 20% mean)',
            '5. Z-score normalization (final step)'
        ],
        'metadata': {
            'sampling_rate': 256,
            'filter_range': '0.5-50 Hz',
            'filter_order': 4,
            'filter_type': 'Butterworth bandpass with zero-phase',
            'artifact_detection': 'Adaptive thresholds (6 std)',
            'epoching': 'Center extraction or padding to 256 samples',
            'baseline_correction': 'First 20% of epoch',
            'normalization': 'Z-score per signal (mean=0, std=1)',
            'processing_order': 'CORRECT: Filter→Artifact→Epoch→Baseline→Normalize'
        }
    }
    
    with open('correctly_preprocessed_eeg_data.pkl', 'wb') as f:
        pickle.dump(correct_dataset, f)
    
    print(f"✅ Correctly preprocessed data saved as 'correctly_preprocessed_eeg_data.pkl'")
    
    print(f"\n🎉 CORRECT PREPROCESSING COMPLETED!")
    print("=" * 60)
    print("✅ Step 1: Bandpass filtering (on RAW data)")
    print("✅ Step 2: Artifact detection and removal")
    print("✅ Step 3: Epoching (length standardization)")
    print("✅ Step 4: Baseline correction")
    print("✅ Step 5: Normalization (FINAL step)")
    print(f"✅ Rejection rate: {stats['rejection_rate']:.1%}")
    print("✅ Processing order: CORRECT!")
    
    return correctly_processed_data, stats

if __name__ == "__main__":
    correctly_processed_data, processing_stats = main()
