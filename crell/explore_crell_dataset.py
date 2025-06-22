#!/usr/bin/env python3
"""
Explore Crell Dataset (S01.mat) with Letter Stimuli
Dataset contains EEG data with letter recognition task
Letters: a,d,e,f,j,n,o,s,t,v (10 letters)
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from collections import Counter

def explore_crell_dataset():
    """Explore the Crell dataset structure and content"""
    print("🔍 EXPLORING CRELL DATASET")
    print("=" * 60)
    
    # Load S01.mat dataset
    print("📂 Loading S01.mat dataset...")
    try:
        data = scipy.io.loadmat('../dataset/datasets/S01.mat')
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Explore dataset structure
    print(f"\n📊 Dataset Keys: {list(data.keys())}")
    
    # Remove MATLAB metadata keys
    data_keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"📊 Data Keys (excluding metadata): {data_keys}")
    
    # Explore each key
    for key in data_keys:
        print(f"\n🔍 Key: '{key}'")
        print(f"   Type: {type(data[key])}")
        print(f"   Shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")
        print(f"   Dtype: {data[key].dtype if hasattr(data[key], 'dtype') else 'N/A'}")
        
        # Show sample values for small arrays
        if hasattr(data[key], 'shape') and np.prod(data[key].shape) < 100:
            print(f"   Sample values: {data[key].flatten()[:10]}")
        elif hasattr(data[key], 'shape'):
            print(f"   Value range: [{data[key].min():.3f}, {data[key].max():.3f}]")
    
    return data

def explore_crell_stimuli():
    """Explore the Crell stimuli images"""
    print(f"\n🖼️ EXPLORING CRELL STIMULI")
    print("=" * 60)
    
    stimuli_path = '../dataset/datasets/crellStimuli'
    
    # List all stimulus files
    stimulus_files = [f for f in os.listdir(stimuli_path) if f.endswith('.png')]
    stimulus_files.sort()
    
    print(f"📊 Found {len(stimulus_files)} stimulus images:")
    
    # Letter mapping based on README
    letter_mapping = {
        'a': 100, 'd': 103, 'e': 104, 'f': 105, 'j': 109,
        'n': 113, 'o': 114, 's': 118, 't': 119, 'v': 121
    }
    
    stimuli_data = {}
    
    for filename in stimulus_files:
        letter = filename.split('.')[0]  # Get letter from filename
        filepath = os.path.join(stimuli_path, filename)
        
        # Load image
        img = Image.open(filepath)
        img_array = np.array(img)
        
        print(f"   📄 {filename}: {img_array.shape}, dtype={img_array.dtype}")
        print(f"      Letter: '{letter}' -> Code: {letter_mapping.get(letter, 'Unknown')}")
        print(f"      Value range: [{img_array.min()}, {img_array.max()}]")
        
        stimuli_data[letter] = {
            'image': img_array,
            'code': letter_mapping.get(letter, -1),
            'filename': filename
        }
    
    return stimuli_data

def analyze_eeg_data(data):
    """Analyze EEG data structure and content"""
    print(f"\n🧠 ANALYZING EEG DATA")
    print("=" * 60)

    # This dataset has a complex nested structure
    # Let's explore the nested structure
    for key in data.keys():
        if not key.startswith('__'):
            print(f"\n🔍 Exploring nested structure of '{key}':")

            # Access the nested structure
            nested_data = data[key][0, 0]  # Access the structured array

            # Get field names
            field_names = nested_data.dtype.names
            print(f"   Fields: {field_names}")

            for field in field_names:
                field_data = nested_data[field]
                print(f"\n   📊 Field '{field}':")
                print(f"      Type: {type(field_data)}")
                print(f"      Shape: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")

                if hasattr(field_data, 'shape') and len(field_data.shape) > 0:
                    print(f"      Dtype: {field_data.dtype}")
                    print(f"      Value range: [{field_data.min():.3f}, {field_data.max():.3f}]")

                    # Check if this looks like EEG data
                    if 'BrainVisionRDA_data' in field:
                        print(f"      🧠 This looks like EEG data!")
                        print(f"         Shape interpretation: {field_data.shape[0]} time points, {field_data.shape[1]} channels")

                    # Check if this looks like markers/labels
                    elif 'Marker_data' in field:
                        print(f"      🏷️ This looks like marker/label data!")
                        unique_vals = np.unique(field_data)
                        print(f"         Unique marker values: {unique_vals[:20]}...")  # Show first 20

                        # Check for letter codes
                        letter_codes = [100, 103, 104, 105, 109, 113, 114, 118, 119, 121]
                        if any(val in letter_codes for val in unique_vals):
                            print(f"         ✅ Contains letter codes!")

    return [], []  # Return empty for now, we'll process this differently

def visualize_stimuli(stimuli_data):
    """Visualize all stimulus letters"""
    print(f"\n🎨 VISUALIZING STIMULI")
    print("=" * 60)
    
    letters = sorted(stimuli_data.keys())
    n_letters = len(letters)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Crell Dataset: Letter Stimuli', fontsize=16, fontweight='bold')
    
    for i, letter in enumerate(letters):
        row = i // 5
        col = i % 5
        
        img = stimuli_data[letter]['image']
        code = stimuli_data[letter]['code']
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"Letter '{letter}'\nCode: {code}", fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('crell_stimuli_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Stimuli visualization saved as 'crell_stimuli_overview.png'")

def main():
    """Main exploration function"""
    print("🎯 CRELL DATASET EXPLORATION")
    print("=" * 80)
    print("📝 Dataset: S01.mat with letter stimuli (a,d,e,f,j,n,o,s,t,v)")
    print("=" * 80)
    
    # Explore dataset
    data = explore_crell_dataset()
    if data is None:
        return
    
    # Explore stimuli
    stimuli_data = explore_crell_stimuli()
    
    # Analyze EEG data
    possible_eeg_keys, possible_label_keys = analyze_eeg_data(data)
    
    # Visualize stimuli
    visualize_stimuli(stimuli_data)
    
    print(f"\n🎯 CRELL DATASET SUMMARY:")
    print(f"   Dataset: Letter recognition EEG task")
    print(f"   Letters: {len(stimuli_data)} letters")
    print(f"   Stimulus files: {list(stimuli_data.keys())}")
    print(f"   EEG data candidates: {[key for key, _ in possible_eeg_keys]}")
    print(f"   Label candidates: {[key for key, _ in possible_label_keys]}")
    
    print(f"\n📁 Generated files:")
    print(f"   - crell_stimuli_overview.png")
    
    print(f"\n🚀 Ready for detailed data processing!")

if __name__ == "__main__":
    main()
