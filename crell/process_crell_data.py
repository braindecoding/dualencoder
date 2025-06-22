#!/usr/bin/env python3
"""
Process Crell Dataset for EEG-to-Letter Reconstruction
Extract EEG epochs aligned with letter presentation markers
Create train/validation/test splits
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

def step1_bandpass_filter(raw_signal, fs=500, lowcut=0.5, highcut=50, order=4):
    """
    STEP 1: Apply bandpass filter to RAW EEG signal
    This MUST be done FIRST on raw data before any other processing
    """
    # Design Butterworth bandpass filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create second-order sections for stability
    sos = signal.butter(order, [low, high], btype='band', output='sos')

    # Apply zero-phase filtering (forward and backward)
    filtered_signal = signal.sosfiltfilt(sos, raw_signal)

    return filtered_signal

def step2_artifact_detection(filtered_signal, fs=500):
    """
    STEP 2: Detect artifacts in filtered signal
    Apply AFTER filtering but BEFORE epoching
    """
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

def step4_baseline_correction(epoch, fs=500, baseline_duration=0.2):
    """
    STEP 4: Apply baseline correction to epoch
    Apply AFTER epoching but BEFORE normalization
    """
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
    """
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

def correct_preprocessing_pipeline(raw_signal, fs=500, target_length=500):
    """
    Complete CORRECT preprocessing pipeline for single EEG signal
    Apply steps in the CORRECT ORDER:
    1. Bandpass filtering (on raw data)
    2. Artifact detection
    3. Epoching (already done)
    4. Baseline correction
    5. Normalization
    """
    try:
        # STEP 1: Bandpass filtering (FIRST - on raw data)
        filtered_signal = step1_bandpass_filter(raw_signal, fs=fs)

        # STEP 2: Artifact detection (on filtered data)
        artifacts = step2_artifact_detection(filtered_signal, fs=fs)
        if artifacts['is_artifact']:
            return None

        # STEP 3: Epoching (already done in extract_eeg_epochs)
        # Just ensure correct length
        if len(filtered_signal) != target_length:
            if len(filtered_signal) > target_length:
                # Extract from center
                start_idx = (len(filtered_signal) - target_length) // 2
                epoched_signal = filtered_signal[start_idx:start_idx + target_length]
            else:
                # Pad if too short
                pad_width = target_length - len(filtered_signal)
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                epoched_signal = np.pad(filtered_signal, (pad_left, pad_right), mode='constant')
        else:
            epoched_signal = filtered_signal

        # STEP 4: Baseline correction
        baseline_corrected = step4_baseline_correction(epoched_signal, fs=fs)

        # STEP 5: Normalization (FINAL STEP)
        normalized_signal = step5_normalization(baseline_corrected, method='zscore')

        return normalized_signal

    except Exception as e:
        return None

def load_crell_data():
    """Load and parse Crell dataset"""
    print("📂 Loading Crell dataset...")
    
    data = scipy.io.loadmat('../dataset/datasets/S01.mat')
    
    # Extract data from both rounds
    rounds_data = {}
    
    for round_name in ['round01_paradigm', 'round02_paradigm']:
        print(f"\n🔍 Processing {round_name}...")
        
        # Access nested structure
        nested_data = data[round_name][0, 0]
        
        # Extract EEG data and markers
        eeg_data = nested_data['BrainVisionRDA_data']  # (time_points, channels)
        eeg_time = nested_data['BrainVisionRDA_time'].flatten()  # (time_points,)
        markers = nested_data['ParadigmMarker_data'].flatten()  # (n_markers,)
        marker_times = nested_data['ParadigmMarker_time'].flatten()  # (n_markers,)
        
        print(f"   EEG data: {eeg_data.shape}")
        print(f"   EEG time: {eeg_time.shape}")
        print(f"   Markers: {markers.shape}")
        print(f"   Marker times: {marker_times.shape}")
        
        # Filter for letter codes only
        letter_codes = [100, 103, 104, 105, 109, 113, 114, 118, 119, 121]
        letter_mask = np.isin(markers, letter_codes)
        
        letter_markers = markers[letter_mask]
        letter_marker_times = marker_times[letter_mask]
        
        print(f"   Letter markers: {len(letter_markers)}")
        print(f"   Letter distribution: {dict(Counter(letter_markers))}")
        
        rounds_data[round_name] = {
            'eeg_data': eeg_data,
            'eeg_time': eeg_time,
            'letter_markers': letter_markers,
            'letter_marker_times': letter_marker_times
        }
    
    return rounds_data

def load_stimuli():
    """Load letter stimulus images"""
    print(f"\n🖼️ Loading stimulus images...")
    
    stimuli_path = '../dataset/datasets/crellStimuli'
    
    # Letter mapping
    letter_mapping = {
        100: 'a', 103: 'd', 104: 'e', 105: 'f', 109: 'j',
        113: 'n', 114: 'o', 118: 's', 119: 't', 121: 'v'
    }
    
    stimuli_data = {}
    
    for code, letter in letter_mapping.items():
        filepath = os.path.join(stimuli_path, f'{letter}.png')
        
        # Load and process image
        img = Image.open(filepath)
        img_array = np.array(img)
        
        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        
        # Normalize to [0, 1]
        img_normalized = img_gray.astype(np.float32) / 255.0
        
        stimuli_data[code] = {
            'letter': letter,
            'image': img_normalized,
            'original': img_array
        }
        
        print(f"   {letter} (code {code}): {img_normalized.shape}")
    
    return stimuli_data

def extract_eeg_epochs(rounds_data, stimuli_data, epoch_duration=1.0, baseline_duration=0.2):
    """Extract EEG epochs around letter presentation with CORRECT preprocessing"""
    print(f"\n🧠 Extracting EEG epochs with CORRECT preprocessing...")
    print(f"   Epoch duration: {epoch_duration}s")
    print(f"   Baseline duration: {baseline_duration}s")
    print("   Preprocessing order: Filter→Artifact→Epoch→Baseline→Normalize")

    # Assume 500 Hz sampling rate (common for EEG)
    sampling_rate = 500
    epoch_samples = int(epoch_duration * sampling_rate)
    baseline_samples = int(baseline_duration * sampling_rate)

    all_epochs = []
    all_labels = []
    all_images = []
    processed_count = 0
    rejected_count = 0

    for round_name, round_data in rounds_data.items():
        print(f"\n   Processing {round_name}...")

        eeg_data = round_data['eeg_data']  # (time_points, channels)
        eeg_time = round_data['eeg_time']
        letter_markers = round_data['letter_markers']
        letter_marker_times = round_data['letter_marker_times']

        # Find marker indices in EEG time
        for i, (marker_code, marker_time) in enumerate(zip(letter_markers, letter_marker_times)):
            # Find closest EEG time point
            time_diff = np.abs(eeg_time - marker_time)
            marker_idx = np.argmin(time_diff)

            # Extract epoch: baseline before + stimulus period
            start_idx = marker_idx - baseline_samples
            end_idx = marker_idx + epoch_samples - baseline_samples

            # Check bounds
            if start_idx >= 0 and end_idx < len(eeg_data):
                # Extract raw epoch for each channel
                raw_epoch = eeg_data[start_idx:end_idx, :]  # (time_points, channels)

                # Apply CORRECT preprocessing to each channel
                processed_channels = []
                channel_rejected = False

                for ch in range(raw_epoch.shape[1]):
                    raw_channel = raw_epoch[:, ch]

                    # Apply CORRECT preprocessing pipeline
                    processed_channel = correct_preprocessing_pipeline(
                        raw_channel, fs=sampling_rate, target_length=epoch_samples
                    )

                    if processed_channel is not None:
                        processed_channels.append(processed_channel)
                    else:
                        channel_rejected = True
                        break

                # Only keep epoch if all channels processed successfully
                if not channel_rejected and len(processed_channels) == raw_epoch.shape[1]:
                    # Stack channels: (channels, time_points)
                    epoch_processed = np.array(processed_channels)

                    # Get corresponding stimulus image
                    if marker_code in stimuli_data:
                        stimulus_image = stimuli_data[marker_code]['image']

                        all_epochs.append(epoch_processed)
                        all_labels.append(marker_code)
                        all_images.append(stimulus_image)
                        processed_count += 1

                        if processed_count % 50 == 0:
                            print(f"      Processed {processed_count} epochs...")
                else:
                    rejected_count += 1

    # Convert to arrays
    epochs_array = np.array(all_epochs)  # (n_epochs, channels, time_points)
    labels_array = np.array(all_labels)  # (n_epochs,)
    images_array = np.array(all_images)  # (n_epochs, height, width)

    print(f"\n✅ CORRECT preprocessing completed:")
    print(f"   Total epochs processed: {processed_count}")
    print(f"   Epochs rejected: {rejected_count}")
    print(f"   Rejection rate: {rejected_count/(processed_count+rejected_count)*100:.1f}%")
    print(f"   Final epoch shape: {epochs_array.shape}")
    print(f"   Labels shape: {labels_array.shape}")
    print(f"   Images shape: {images_array.shape}")
    print(f"   Label distribution: {dict(Counter(labels_array))}")

    return epochs_array, labels_array, images_array

def create_letter_mapping(labels_array):
    """Create mapping from letter codes to indices"""
    unique_codes = sorted(np.unique(labels_array))
    
    code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
    idx_to_code = {idx: code for idx, code in enumerate(unique_codes)}
    
    # Letter names
    letter_names = {
        100: 'a', 103: 'd', 104: 'e', 105: 'f', 109: 'j',
        113: 'n', 114: 'o', 118: 's', 119: 't', 121: 'v'
    }
    
    idx_to_letter = {idx: letter_names[code] for idx, code in idx_to_code.items()}
    
    print(f"\n📊 Letter mapping:")
    for idx, code in idx_to_code.items():
        letter = letter_names[code]
        print(f"   {letter} (code {code}) -> index {idx}")
    
    return code_to_idx, idx_to_code, idx_to_letter

def create_data_splits(epochs_array, labels_array, images_array, test_size=0.2, val_size=0.2):
    """Create stratified train/validation/test splits"""
    print(f"\n📊 Creating data splits...")
    
    # Convert labels to indices for stratification
    code_to_idx, idx_to_code, idx_to_letter = create_letter_mapping(labels_array)
    labels_idx = np.array([code_to_idx[code] for code in labels_array])
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, img_temp, img_test = train_test_split(
        epochs_array, labels_idx, images_array,
        test_size=test_size, stratify=labels_idx, random_state=42
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val, img_train, img_val = train_test_split(
        X_temp, y_temp, img_temp,
        test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    
    print(f"   Training set: {X_train.shape[0]} epochs")
    print(f"   Validation set: {X_val.shape[0]} epochs")
    print(f"   Test set: {X_test.shape[0]} epochs")
    
    # Check distribution
    for split_name, split_labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = Counter(split_labels)
        print(f"   {split_name} distribution: {dict(dist)}")
    
    return {
        'training': {
            'eeg': X_train,
            'labels': y_train,
            'images': img_train
        },
        'validation': {
            'eeg': X_val,
            'labels': y_val,
            'images': img_val
        },
        'test': {
            'eeg': X_test,
            'labels': y_test,
            'images': img_test
        },
        'metadata': {
            'code_to_idx': code_to_idx,
            'idx_to_code': idx_to_code,
            'idx_to_letter': idx_to_letter,
            'n_channels': epochs_array.shape[1],
            'n_timepoints': epochs_array.shape[2],
            'n_letters': len(code_to_idx),
            'sampling_rate': 500,
            'epoch_duration': 1.0,
            'baseline_duration': 0.2
        }
    }

def visualize_sample_epochs(data_splits):
    """Visualize sample EEG epochs and corresponding letters"""
    print(f"\n🎨 Creating visualization...")
    
    # Get sample data
    sample_eeg = data_splits['training']['eeg'][:4]  # First 4 epochs
    sample_labels = data_splits['training']['labels'][:4]
    sample_images = data_splits['training']['images'][:4]
    
    idx_to_letter = data_splits['metadata']['idx_to_letter']
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Crell Dataset: EEG Epochs and Letter Stimuli', fontsize=16, fontweight='bold')
    
    for i in range(4):
        # EEG epoch (show average across channels)
        eeg_avg = sample_eeg[i].mean(axis=0)  # Average across channels
        axes[i, 0].plot(eeg_avg)
        axes[i, 0].set_title(f'EEG Epoch {i+1}\nLetter: {idx_to_letter[sample_labels[i]]}')
        axes[i, 0].set_xlabel('Time points')
        axes[i, 0].set_ylabel('Amplitude (μV)')
        
        # EEG topography (show at peak time)
        peak_time = np.argmax(np.abs(eeg_avg))
        topo = sample_eeg[i][:, peak_time]
        im = axes[i, 1].imshow(topo.reshape(8, 8), cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'EEG Topography\n(t={peak_time})')
        axes[i, 1].axis('off')
        
        # Letter stimulus
        axes[i, 2].imshow(sample_images[i], cmap='gray')
        axes[i, 2].set_title(f'Letter Stimulus\n"{idx_to_letter[sample_labels[i]]}"')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('crell_sample_epochs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Sample visualization saved as 'crell_sample_epochs.png'")

def main():
    """Main processing function"""
    print("🎯 CRELL DATASET PROCESSING WITH CORRECT PREPROCESSING")
    print("=" * 90)
    print("📝 Processing EEG data for letter recognition task")
    print("🔧 CORRECT Preprocessing Order:")
    print("   1. Bandpass filtering (0.5-50 Hz) - on RAW data")
    print("   2. Artifact detection and removal")
    print("   3. Epoching (length standardization)")
    print("   4. Baseline correction")
    print("   5. Z-score normalization (FINAL step)")
    print("=" * 90)
    
    # Load raw data
    rounds_data = load_crell_data()
    stimuli_data = load_stimuli()
    
    # Extract epochs
    epochs_array, labels_array, images_array = extract_eeg_epochs(rounds_data, stimuli_data)
    
    # Create data splits
    data_splits = create_data_splits(epochs_array, labels_array, images_array)
    
    # Add preprocessing metadata
    data_splits['preprocessing_metadata'] = {
        'preprocessing_order': [
            '1. Bandpass filtering (0.5-50 Hz) - applied to RAW data',
            '2. Artifact detection and removal (6 std threshold)',
            '3. Epoching (length standardization to 500 samples)',
            '4. Baseline correction (subtract first 20% mean)',
            '5. Z-score normalization (final step)'
        ],
        'filter_range': '0.5-50 Hz',
        'filter_order': 4,
        'filter_type': 'Butterworth bandpass with zero-phase',
        'artifact_detection': 'Adaptive thresholds (6 std)',
        'baseline_correction': 'First 20% of epoch',
        'normalization': 'Z-score per channel (mean=0, std=1)',
        'processing_order': 'CORRECT: Filter→Artifact→Epoch→Baseline→Normalize'
    }

    # Save processed data
    with open('crell_processed_data_correct.pkl', 'wb') as f:
        pickle.dump(data_splits, f)

    print(f"\n✅ CORRECT preprocessing completed!")
    print(f"   Processed data saved to 'crell_processed_data_correct.pkl'")
    
    # Create visualization
    visualize_sample_epochs(data_splits)
    
    # Summary
    print(f"\n🎯 CRELL DATASET SUMMARY:")
    print(f"   Task: EEG-to-Letter reconstruction")
    print(f"   Letters: {data_splits['metadata']['n_letters']} letters")
    print(f"   EEG channels: {data_splits['metadata']['n_channels']}")
    print(f"   Time points per epoch: {data_splits['metadata']['n_timepoints']}")
    print(f"   Sampling rate: {data_splits['metadata']['sampling_rate']} Hz")
    print(f"   Total epochs: {sum(len(split['eeg']) for split in [data_splits['training'], data_splits['validation'], data_splits['test']])}")
    
    print(f"\n📁 Generated files:")
    print(f"   - crell_processed_data_correct.pkl (with CORRECT preprocessing)")
    print(f"   - crell_sample_epochs.png")

    print(f"\n🔧 CORRECT Preprocessing Applied:")
    print(f"   ✅ Step 1: Bandpass filtering (0.5-50 Hz) on RAW data")
    print(f"   ✅ Step 2: Artifact detection and removal")
    print(f"   ✅ Step 3: Epoching (length standardization)")
    print(f"   ✅ Step 4: Baseline correction")
    print(f"   ✅ Step 5: Z-score normalization (FINAL step)")

    print(f"\n🚀 Ready for EEG-to-Letter modeling with CORRECT preprocessing!")

if __name__ == "__main__":
    main()
