# Crell Dataset - EEG-to-Letter Reconstruction

## 📝 Overview

This folder contains the processing and analysis of the **Crell dataset** for EEG-to-letter reconstruction tasks. The dataset includes EEG recordings from letter recognition experiments with 10 letters: **a, d, e, f, j, n, o, s, t, v**.

## 📊 Dataset Information

### **Source Data**
- **File**: `../dataset/datasets/S01.mat`
- **Stimuli**: `../dataset/datasets/crellStimuli/` (10 letter images)
- **Task**: Letter recognition from visual stimuli
- **Subject**: Single subject (S01)

### **Dataset Structure**
- **2 Rounds**: `round01_paradigm` and `round02_paradigm`
- **EEG Channels**: 64 channels
- **Sampling Rate**: 500 Hz (verified)
- **Total Epochs**: 640 (64 per letter × 10 letters)
- **Letter Codes**: a=100, d=103, e=104, f=105, j=109, n=113, o=114, s=118, t=119, v=121

### **Stimulus Images**
- **Size**: 28×28 pixels (RGB converted to grayscale)
- **Format**: PNG files
- **Letters**: a, d, e, f, j, n, o, s, t, v
- **Normalization**: [0, 1] range

## 🔧 Preprocessing Pipeline

### **CORRECT Preprocessing Order Applied**
Following the established `correct_eeg_preprocessing_pipeline.py`:

1. **Bandpass Filtering** (0.5-50 Hz)
   - Applied to RAW data FIRST
   - Butterworth filter, order 4
   - Zero-phase filtering (forward-backward)

2. **Artifact Detection & Removal**
   - Adaptive thresholds (6 standard deviations)
   - Conservative rejection criteria
   - Amplitude, gradient, and flatline detection

3. **Epoching**
   - 1.0 second epochs (500 samples at 500 Hz)
   - Baseline: 0.2 seconds before stimulus
   - Stimulus period: 0.8 seconds after stimulus

4. **Baseline Correction**
   - Subtract mean of first 20% of epoch
   - Applied per channel independently

5. **Z-score Normalization** (FINAL STEP)
   - Mean = 0, Std = 1
   - Applied per channel per epoch

### **Processing Results**
- **Total Epochs Processed**: 640
- **Rejection Rate**: 0.0% (excellent data quality)
- **Final Shape**: (640, 64, 500) - epochs × channels × time points

## 📁 Files Description

### **Core Processing Files**
- `process_crell_data.py` - Main data processing script with CORRECT preprocessing
- `crell_processed_data_correct.pkl` - **Final processed dataset** (use this!)

### **Exploration Files**
- `explore_crell_dataset.py` - Initial dataset exploration and structure analysis
- `check_crell_sampling_rate.py` - Sampling rate verification (confirmed 500 Hz)

### **Visualization Files**
- `crell_stimuli_overview.png` - Overview of all 10 letter stimuli
- `crell_sample_epochs.png` - Sample EEG epochs and corresponding letters

### **Legacy Files**
- `crell_processed_data.pkl` - Old processing (without correct preprocessing)
- `correct_eeg_preprocessing_pipeline.py` - Reference preprocessing pipeline

## 📈 Data Splits

**Stratified splits ensuring balanced letter distribution:**

- **Training**: 384 epochs (60%)
- **Validation**: 128 epochs (20%)
- **Test**: 128 epochs (20%)

Each split contains approximately equal numbers of each letter (38-39 per letter in training, 12-13 per letter in validation/test).

## 🎯 Usage

### **Load Processed Data**
```python
import pickle

# Load the correctly preprocessed data
with open('crell_processed_data_correct.pkl', 'rb') as f:
    data = pickle.load(f)

# Access data splits
train_eeg = data['training']['eeg']        # (384, 64, 500)
train_labels = data['training']['labels']  # (384,) - indices 0-9
train_images = data['training']['images']  # (384, 28, 28)

val_eeg = data['validation']['eeg']        # (128, 64, 500)
test_eeg = data['test']['eeg']            # (128, 64, 500)

# Access metadata
metadata = data['metadata']
letter_mapping = metadata['idx_to_letter']  # {0: 'a', 1: 'd', ...}
```

### **Letter Mapping**
```python
# Index to letter mapping
idx_to_letter = {
    0: 'a', 1: 'd', 2: 'e', 3: 'f', 4: 'j',
    5: 'n', 6: 'o', 7: 's', 8: 't', 9: 'v'
}

# Letter code to index mapping
code_to_idx = {
    100: 0, 103: 1, 104: 2, 105: 3, 109: 4,
    113: 5, 114: 6, 118: 7, 119: 8, 121: 9
}
```

## 🚀 Next Steps

This dataset is ready for:

1. **EEG Encoder Development**
   - 64-channel EEG feature extraction
   - Temporal pattern recognition
   - Letter-specific neural signatures

2. **Letter Reconstruction Models**
   - EEG-to-image generation
   - Diffusion models for letter synthesis
   - Classification-guided generation

3. **Performance Evaluation**
   - Letter recognition accuracy
   - Visual similarity metrics (SSIM, LPIPS)
   - Perceptual quality assessment

## 📋 Technical Specifications

- **EEG Shape**: (n_epochs, 64, 500)
- **Image Shape**: (n_epochs, 28, 28)
- **Sampling Rate**: 500 Hz
- **Epoch Duration**: 1.0 second
- **Baseline Period**: 0.2 seconds
- **Preprocessing**: CORRECT pipeline applied
- **Data Quality**: Excellent (0% rejection rate)

## 🔗 Related Files

- **MindBigData Processing**: `../mbd3/` (digit recognition)
- **Original Preprocessing**: `../mbd/correct_eeg_preprocessing_pipeline.py`
- **Dataset Source**: `../dataset/datasets/S01.mat`

---

**Status**: ✅ Ready for EEG-to-Letter modeling with CORRECT preprocessing applied
