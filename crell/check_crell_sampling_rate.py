#!/usr/bin/env python3
"""
Check actual sampling rate of Crell dataset
"""

import scipy.io
import numpy as np

def check_crell_sampling_rate():
    """Check the actual sampling rate from Crell dataset"""
    print("🔍 CHECKING CRELL DATASET SAMPLING RATE")
    print("=" * 60)
    
    # Load dataset
    data = scipy.io.loadmat('../dataset/datasets/S01.mat')
    
    # Check round01_paradigm
    round01 = data['round01_paradigm'][0, 0]
    eeg_time = round01['BrainVisionRDA_time'].flatten()
    
    print(f"📊 Round 01 Analysis:")
    print(f"   Total time points: {len(eeg_time):,}")
    print(f"   Time range: {eeg_time[0]:.6f} to {eeg_time[-1]:.6f}")
    print(f"   Duration: {eeg_time[-1] - eeg_time[0]:.3f} seconds")
    
    # Calculate sampling rate from time differences
    time_diffs = np.diff(eeg_time)
    mean_time_diff = np.mean(time_diffs)
    sampling_rate = 1.0 / mean_time_diff
    
    print(f"   Mean time difference: {mean_time_diff:.6f} seconds")
    print(f"   Calculated sampling rate: {sampling_rate:.2f} Hz")
    
    # Check consistency
    std_time_diff = np.std(time_diffs)
    print(f"   Time difference std: {std_time_diff:.8f} seconds")
    print(f"   Sampling consistency: {'Good' if std_time_diff < 1e-6 else 'Variable'}")
    
    # Check round02_paradigm
    round02 = data['round02_paradigm'][0, 0]
    eeg_time2 = round02['BrainVisionRDA_time'].flatten()
    
    print(f"\n📊 Round 02 Analysis:")
    print(f"   Total time points: {len(eeg_time2):,}")
    print(f"   Time range: {eeg_time2[0]:.6f} to {eeg_time2[-1]:.6f}")
    print(f"   Duration: {eeg_time2[-1] - eeg_time2[0]:.3f} seconds")
    
    time_diffs2 = np.diff(eeg_time2)
    mean_time_diff2 = np.mean(time_diffs2)
    sampling_rate2 = 1.0 / mean_time_diff2
    
    print(f"   Mean time difference: {mean_time_diff2:.6f} seconds")
    print(f"   Calculated sampling rate: {sampling_rate2:.2f} Hz")
    
    # Overall conclusion
    avg_sampling_rate = (sampling_rate + sampling_rate2) / 2
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   Average sampling rate: {avg_sampling_rate:.2f} Hz")
    
    # Determine closest standard sampling rate
    standard_rates = [250, 256, 500, 512, 1000, 1024]
    closest_rate = min(standard_rates, key=lambda x: abs(x - avg_sampling_rate))
    
    print(f"   Closest standard rate: {closest_rate} Hz")
    print(f"   Difference: {abs(closest_rate - avg_sampling_rate):.2f} Hz")
    
    # Recommendation
    if abs(closest_rate - avg_sampling_rate) < 5:
        print(f"   ✅ Recommendation: Use {closest_rate} Hz")
    else:
        print(f"   ⚠️ Recommendation: Use actual rate {avg_sampling_rate:.0f} Hz")
    
    return avg_sampling_rate, closest_rate

if __name__ == "__main__":
    actual_rate, recommended_rate = check_crell_sampling_rate()
