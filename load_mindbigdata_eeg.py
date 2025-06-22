#!/usr/bin/env python3
"""
Load and Analyze MindBigData EEG Dataset (EP1.01.txt)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def parse_eeg_line(line):
    """Parse a single EEG line from the dataset"""
    parts = line.strip().split('\t')
    if len(parts) < 7:
        return None

    try:
        # Extract metadata
        record_id = int(parts[0])
        event_id = int(parts[1])
        event_type = parts[2]  # 'EP'
        electrode = parts[3]   # e.g., 'AF3', 'F7', etc.
        digit = int(parts[4])  # 0-9
        length = int(parts[5]) # Should be 260

        # Extract EEG signal data (comma-separated values)
        signal_data = parts[6].split(',')
        signal_values = [float(x) for x in signal_data if x.strip()]

        return {
            'record_id': record_id,
            'event_id': event_id,
            'event_type': event_type,
            'electrode': electrode,
            'digit': digit,
            'length': length,
            'signal': np.array(signal_values)
        }
    except Exception as e:
        return None

def load_mindbigdata_eeg():
    """Load MindBigData EEG dataset from EP1.01.txt"""
    print("📂 Loading MindBigData EEG Dataset...")

    # Path to the dataset
    dataset_path = Path("../dataset/datasets/EP1.01.txt")

    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return None

    print(f"✅ Found dataset at: {dataset_path}")
    print(f"📊 File size: {dataset_path.stat().st_size / (1024*1024):.2f} MB")

    # Parse the EEG data
    print("\n🔍 Parsing EEG data structure...")

    # Sample first few lines to understand structure
    with open(dataset_path, 'r') as f:
        sample_lines = [f.readline().strip() for _ in range(10)]

    print("Sample line structure:")
    for i, line in enumerate(sample_lines[:3], 1):
        parsed = parse_eeg_line(line)
        if parsed:
            print(f"  Line {i}: Electrode={parsed['electrode']}, Digit={parsed['digit']}, Signal_length={len(parsed['signal'])}")

    # Load all data
    print("\n📊 Loading all EEG data...")
    eeg_records = []
    electrodes = set()
    digits = set()

    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"   Processed {line_num:,} lines...")

            parsed = parse_eeg_line(line)
            if parsed:
                eeg_records.append(parsed)
                electrodes.add(parsed['electrode'])
                digits.add(parsed['digit'])

    print(f"✅ Loaded {len(eeg_records):,} EEG records")
    print(f"📊 Unique electrodes: {sorted(electrodes)}")
    print(f"📊 Unique digits: {sorted(digits)}")

    # Organize data by electrode and digit
    print("\n🔄 Organizing data structure...")
    organized_data = {}

    for record in eeg_records:
        electrode = record['electrode']
        digit = record['digit']

        if electrode not in organized_data:
            organized_data[electrode] = {}
        if digit not in organized_data[electrode]:
            organized_data[electrode][digit] = []

        organized_data[electrode][digit].append(record['signal'])

    # Print organization summary
    print(f"📊 Data organization:")
    for electrode in sorted(organized_data.keys()):
        digit_counts = {digit: len(signals) for digit, signals in organized_data[electrode].items()}
        print(f"   {electrode}: {dict(sorted(digit_counts.items()))}")

    return organized_data, eeg_records

def analyze_eeg_data(organized_data, eeg_records):
    """Analyze the loaded EEG data"""
    print("\n📊 ANALYZING EEG DATA")
    print("=" * 50)

    if organized_data is None:
        print("❌ No data to analyze")
        return

    # Analyze data structure
    electrodes = list(organized_data.keys())
    digits = set()
    for electrode_data in organized_data.values():
        digits.update(electrode_data.keys())
    digits = sorted(digits)

    print(f"📊 Data Structure:")
    print(f"   Electrodes: {len(electrodes)} ({electrodes})")
    print(f"   Digits: {len(digits)} ({digits})")
    print(f"   Total records: {len(eeg_records):,}")

    # Analyze signal properties
    all_signals = []
    signal_lengths = []

    for electrode in electrodes:
        for digit in digits:
            if digit in organized_data[electrode]:
                for signal in organized_data[electrode][digit]:
                    all_signals.append(signal)
                    signal_lengths.append(len(signal))

    if all_signals:
        all_signals_flat = np.concatenate(all_signals)
        print(f"\n📈 Signal Statistics:")
        print(f"   Total signals: {len(all_signals):,}")
        print(f"   Signal lengths: min={min(signal_lengths)}, max={max(signal_lengths)}, mean={np.mean(signal_lengths):.1f}")
        print(f"   Amplitude range: [{all_signals_flat.min():.3f}, {all_signals_flat.max():.3f}]")
        print(f"   Amplitude mean: {all_signals_flat.mean():.3f}")
        print(f"   Amplitude std: {all_signals_flat.std():.3f}")

    # Analyze samples per digit per electrode
    print(f"\n📊 Samples per Digit per Electrode:")
    sample_matrix = np.zeros((len(electrodes), len(digits)))

    for i, electrode in enumerate(electrodes):
        for j, digit in enumerate(digits):
            if digit in organized_data[electrode]:
                sample_matrix[i, j] = len(organized_data[electrode][digit])

    # Create summary table
    print(f"{'Electrode':<8}", end='')
    for digit in digits:
        print(f"{digit:>6}", end='')
    print(f"{'Total':>8}")

    for i, electrode in enumerate(electrodes):
        print(f"{electrode:<8}", end='')
        total = 0
        for j, digit in enumerate(digits):
            count = int(sample_matrix[i, j])
            print(f"{count:>6}", end='')
            total += count
        print(f"{total:>8}")

    # Expected structure check
    expected_samples_per_digit = 80
    expected_total_per_electrode = expected_samples_per_digit * 10  # 10 digits

    print(f"\n🎯 Expected Structure Check:")
    print(f"   Expected samples per digit: {expected_samples_per_digit}")
    print(f"   Expected total per electrode: {expected_total_per_electrode}")

    # Check if any electrode has the expected structure
    for i, electrode in enumerate(electrodes):
        total_samples = int(sample_matrix[i].sum())
        if total_samples == expected_total_per_electrode:
            print(f"   ✅ {electrode}: Perfect match ({total_samples} samples)")
        else:
            print(f"   ⚠️  {electrode}: {total_samples} samples (expected {expected_total_per_electrode})")

    # Create visualization
    create_eeg_visualization_new(organized_data, electrodes, digits)

    return organized_data

def create_eeg_visualization_new(organized_data, electrodes, digits):
    """Create visualizations for the new EEG data structure"""
    print("\n🎨 Creating EEG visualizations...")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MindBigData EEG Dataset Analysis', fontsize=16, fontweight='bold')

        # 1. Sample count heatmap
        ax1 = axes[0, 0]
        sample_matrix = np.zeros((len(electrodes), len(digits)))

        for i, electrode in enumerate(electrodes):
            for j, digit in enumerate(digits):
                if digit in organized_data[electrode]:
                    sample_matrix[i, j] = len(organized_data[electrode][digit])

        im1 = ax1.imshow(sample_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Samples per Electrode per Digit')
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('Electrode')
        ax1.set_xticks(range(len(digits)))
        ax1.set_xticklabels(digits)
        ax1.set_yticks(range(len(electrodes)))
        ax1.set_yticklabels(electrodes)
        plt.colorbar(im1, ax=ax1)

        # 2. Sample EEG signals from one electrode
        ax2 = axes[0, 1]
        electrode = electrodes[0]  # Use first electrode
        colors = plt.cm.tab10(np.linspace(0, 1, len(digits)))

        for i, digit in enumerate(digits[:5]):  # Show first 5 digits
            if digit in organized_data[electrode] and organized_data[electrode][digit]:
                signal = organized_data[electrode][digit][0]  # First signal for this digit
                ax2.plot(signal, alpha=0.7, label=f'Digit {digit}', color=colors[i])

        ax2.set_title(f'Sample EEG Signals ({electrode})')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Signal length distribution
        ax3 = axes[0, 2]
        signal_lengths = []

        for electrode in electrodes:
            for digit in digits:
                if digit in organized_data[electrode]:
                    for signal in organized_data[electrode][digit]:
                        signal_lengths.append(len(signal))

        ax3.hist(signal_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Signal Length Distribution')
        ax3.set_xlabel('Signal Length')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Amplitude distribution
        ax4 = axes[1, 0]
        all_amplitudes = []

        for electrode in electrodes:
            for digit in digits:
                if digit in organized_data[electrode]:
                    for signal in organized_data[electrode][digit]:
                        all_amplitudes.extend(signal)

        ax4.hist(all_amplitudes, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title('Amplitude Distribution (All Signals)')
        ax4.set_xlabel('Amplitude')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        # 5. Average signal per digit (one electrode)
        ax5 = axes[1, 1]
        electrode = electrodes[0]  # Use first electrode

        for digit in digits:
            if digit in organized_data[electrode] and organized_data[electrode][digit]:
                signals = organized_data[electrode][digit]
                if signals:
                    # Average all signals for this digit
                    min_length = min(len(s) for s in signals)
                    truncated_signals = [s[:min_length] for s in signals]
                    avg_signal = np.mean(truncated_signals, axis=0)
                    ax5.plot(avg_signal, label=f'Digit {digit}', alpha=0.8)

        ax5.set_title(f'Average Signal per Digit ({electrode})')
        ax5.set_xlabel('Time Points')
        ax5.set_ylabel('Amplitude')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Electrode comparison for one digit
        ax6 = axes[1, 2]
        digit = digits[0]  # Use first digit

        for electrode in electrodes[:8]:  # Show first 8 electrodes
            if digit in organized_data[electrode] and organized_data[electrode][digit]:
                signal = organized_data[electrode][digit][0]  # First signal
                ax6.plot(signal, alpha=0.7, label=electrode)

        ax6.set_title(f'Electrode Comparison (Digit {digit})')
        ax6.set_xlabel('Time Points')
        ax6.set_ylabel('Amplitude')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('mindbigdata_eeg_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Visualization saved as 'mindbigdata_eeg_analysis.png'")

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def create_eeg_visualization(data):
    """Create visualizations for EEG data"""
    print("\n🎨 Creating EEG visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MindBigData EEG Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sample EEG signals
        ax1 = axes[0, 0]
        for i in range(min(5, data.shape[0])):
            ax1.plot(data.iloc[i] if isinstance(data, pd.DataFrame) else data[i], 
                    alpha=0.7, label=f'Sample {i+1}')
        ax1.set_title('Sample EEG Signals')
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal distribution
        ax2 = axes[0, 1]
        if isinstance(data, pd.DataFrame):
            flat_data = data.values.flatten()
        else:
            flat_data = data.flatten()
        
        ax2.hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Signal Amplitude Distribution')
        ax2.set_xlabel('Amplitude')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Average signal across all samples
        ax3 = axes[1, 0]
        if isinstance(data, pd.DataFrame):
            mean_signal = data.mean(axis=0)
            std_signal = data.std(axis=0)
        else:
            mean_signal = np.mean(data, axis=0)
            std_signal = np.std(data, axis=0)
        
        time_points = range(len(mean_signal))
        ax3.plot(time_points, mean_signal, 'b-', linewidth=2, label='Mean')
        ax3.fill_between(time_points, 
                        mean_signal - std_signal, 
                        mean_signal + std_signal, 
                        alpha=0.3, label='±1 STD')
        ax3.set_title('Average EEG Signal (All Samples)')
        ax3.set_xlabel('Time Points')
        ax3.set_ylabel('Amplitude')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap of first 50 samples
        ax4 = axes[1, 1]
        n_samples_to_show = min(50, data.shape[0])
        if isinstance(data, pd.DataFrame):
            heatmap_data = data.iloc[:n_samples_to_show]
        else:
            heatmap_data = data[:n_samples_to_show]
        
        im = ax4.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax4.set_title(f'EEG Signals Heatmap (First {n_samples_to_show} samples)')
        ax4.set_xlabel('Time Points')
        ax4.set_ylabel('Sample Index')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('mindbigdata_eeg_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved as 'mindbigdata_eeg_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def analyze_digit_distribution(data):
    """Analyze digit distribution if labels are available"""
    print("\n🔢 DIGIT DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # According to README: 800 samples (80 per digit)
    expected_per_digit = 80
    total_samples = data.shape[0] if hasattr(data, 'shape') else len(data)
    
    print(f"📊 Total samples: {total_samples}")
    print(f"📊 Expected samples per digit: {expected_per_digit}")
    print(f"📊 Expected digits: 0-9 (10 classes)")
    
    if total_samples == 800:
        print("✅ Sample count matches expected (800 samples)")
        
        # Assume sequential organization: first 80 are digit 0, next 80 are digit 1, etc.
        digit_ranges = {}
        for digit in range(10):
            start_idx = digit * 80
            end_idx = start_idx + 80
            digit_ranges[digit] = (start_idx, end_idx)
            print(f"   Digit {digit}: samples {start_idx}-{end_idx-1}")
        
        return digit_ranges
    else:
        print(f"⚠️  Sample count ({total_samples}) doesn't match expected (800)")
        return None

def main():
    """Main function to load and analyze MindBigData EEG dataset"""
    print("🧠 MINDBIGDATA EEG DATASET LOADER")
    print("=" * 60)

    # Load the dataset
    result = load_mindbigdata_eeg()

    if result is not None:
        organized_data, eeg_records = result

        # Analyze the data
        analyzed_data = analyze_eeg_data(organized_data, eeg_records)

        if analyzed_data is not None:
            print(f"\n🎯 DATASET SUMMARY:")
            print(f"   📊 Total EEG records: {len(eeg_records):,}")
            print(f"   📊 Electrodes: {len(organized_data)} channels")
            print(f"   📊 Digits: 0-9 (10 classes)")
            print(f"   📊 Data structure: Multi-electrode EEG recordings")
            print(f"   📊 Signal processing: Ready for EEG-to-image reconstruction")

            return organized_data, eeg_records

    return None, None

if __name__ == "__main__":
    data, digit_info = main()
    
    if data is not None:
        print("\n✅ Dataset loaded successfully!")
        print("🚀 Ready for EEG-to-image reconstruction experiments")
    else:
        print("\n❌ Failed to load dataset")
        print("🔧 Please check file path and format")
