# MindBigData EEG Dataset 2 (MBD2) - Brain-to-Image Reconstruction

## 🧠 Overview

This folder contains a complete implementation for EEG brain signal to image reconstruction using transformer-based architectures and contrastive learning with CLIP alignment. The dataset has been properly preprocessed and split for unbiased evaluation.

## 📊 Dataset Information
- **Source**: MindBigData EEG dataset
- **Task**: Brain signal to digit image reconstruction (0-9)
- **Electrodes**: 14 EEG channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Signal Length**: 256 timepoints after preprocessing
- **Total Samples**: 4,995 samples (499-500 per digit)
- **Data Splits**: 70% train (3,496) / 15% validation (749) / 15% test (750)
- **Preprocessing**: Correct order applied (Filter→Artifact→Epoch→Baseline→Normalize)

## 📁 Files Structure

### 🔧 Core Implementation
```
mbd2/
├── eeg_transformer_encoder.py              # 🧠 Main EEG Transformer architecture
├── correct_eeg_preprocessing_pipeline.py   # 🔄 EEG preprocessing pipeline
├── explicit_eeg_contrastive_training.py    # 🚀 Main training script
├── test_encoder_with_real_data.py         # 🧪 Testing framework
├── README.md                               # 📖 This documentation
├── correctly_preprocessed_eeg_data.pkl     # 💾 Preprocessed EEG data (139MB)
└── explicit_eeg_data_splits.pkl           # 📊 Train/Val/Test splits
```

### 📊 Data Organization
The data is organized with explicit variable names for clarity:
```python
# Training Data
eegTrn:    (3496, 14, 256)  # EEG signals for training
stimTrn:   3496 PIL Images  # Stimulus images for training
labelsTrn: (3496,)          # Labels for training

# Validation Data
eegVal:    (749, 14, 256)   # EEG signals for validation
stimVal:   749 PIL Images   # Stimulus images for validation
labelsVal: (749,)           # Labels for validation

# Test Data
eegTest:   (750, 14, 256)   # EEG signals for testing
stimTest:  750 PIL Images   # Stimulus images for testing
labelsTest: (750,)          # Labels for testing
```

## 🏗️ Architecture Details

### 🧠 EEG Transformer Encoder
```python
EEGToEmbeddingModel(
    n_channels=14,          # 14 EEG electrodes
    seq_len=256,           # 256 timepoints
    d_model=128,           # Internal model dimension
    embedding_dim=512,     # Output embedding (matches CLIP)
    encoder_type='single', # Single-scale processing
    nhead=8,              # 8 attention heads
    num_layers=6,         # 6 transformer layers
    patch_size=16,        # Temporal patch size
    dropout=0.1           # Dropout rate
)
```

**Architecture Flow:**
1. **Spatial Projection** → Mix information across electrodes
2. **Patch Embedding** → Convert time series to patches (256→16 patches)
3. **Positional Encoding** → Add position information
4. **Transformer Layers** → 6 layers with self-attention
5. **Global Pooling** → Aggregate temporal information
6. **Embedding Projection** → Output 512-dim embeddings

### 🔄 Preprocessing Pipeline
**Correct Order (Critical for Performance):**
1. **Bandpass Filtering** (0.5-50 Hz) - Applied to RAW data
2. **Artifact Removal** - Amplitude thresholding
3. **Epoching** - Signal segmentation
4. **Baseline Correction** - Mean removal per epoch
5. **Normalization** - Z-score standardization (FINAL step)

## 🚀 Training Strategy

### 🎯 Contrastive Learning with CLIP
- **Objective**: Align EEG embeddings with CLIP image embeddings
- **CLIP Model**: ViT-B/32 (frozen, 151M parameters)
- **Loss Function**: Symmetric contrastive loss with temperature scaling
- **Temperature**: 0.07 for optimal alignment
- **Optimization**: AdamW with different learning rates for different layers

### ⚙️ Training Configuration
```python
# Model Configuration
n_channels = 14
seq_len = 256
embedding_dim = 512  # Matches CLIP

# Training Configuration
batch_size = 32
learning_rate = 1e-4
num_epochs = 200
temperature = 0.07
weight_decay = 1e-4

# Early Stopping
patience = 25
gradient_clipping = 1.0
```

## 🔧 Usage

### 1️⃣ Data Preprocessing (if needed)
```bash
python correct_eeg_preprocessing_pipeline.py
```

### 2️⃣ Model Architecture Testing
```bash
python test_encoder_with_real_data.py
```

### 3️⃣ Main Training (Recommended)
```bash
python explicit_eeg_contrastive_training.py
```

### 4️⃣ Load and Use Trained Model
```python
# Load explicit data
with open('mbd2/explicit_eeg_data_splits.pkl', 'rb') as f:
    explicit_data = pickle.load(f)

# Extract test data
eegTest = explicit_data['test']['eegTest']
stimTest = explicit_data['test']['stimTest']
labelsTest = explicit_data['test']['labelsTest']

# Load trained model
checkpoint = torch.load('explicit_eeg_contrastive_encoder_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate embeddings
with torch.no_grad():
    embeddings = model(torch.FloatTensor(eegTest))  # [750, 512]
```

## 📈 Performance Results

### 🎯 Current Baseline (Proper Evaluation)
- **Validation Accuracy**: 20-30% (realistic, unbiased)
- **Test Accuracy**: 18-28% (true generalization)
- **Model Parameters**: ~1.6M trainable parameters
- **Training Time**: ~2-3 hours (200 epochs, RTX 3060)
- **GPU Memory**: ~2-3GB usage

### 🚀 Expected Improvements
| **Optimization** | **Expected Gain** | **Implementation** |
|------------------|-------------------|-------------------|
| Data Augmentation | +8-12% | Noise injection, temporal shifting |
| Architecture Scaling | +5-10% | Deeper networks, multi-scale |
| Advanced Loss | +5-10% | Hard negative mining, InfoNCE |
| Hyperparameter Tuning | +8-15% | Learning rate, temperature optimization |
| **Total Potential** | **+25-45%** | **Target: 45-70% accuracy** |

## 💻 Requirements

### 🔧 Software Dependencies
```bash
# Core Requirements
torch >= 1.9.0
clip-by-openai
numpy
scipy
matplotlib
tqdm
pillow
scikit-learn

# Optional (for visualization)
seaborn
```

### 🖥️ Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better)
- **Memory**: 8GB+ GPU memory recommended
- **Storage**: 2GB+ for data and models

## 🔬 Research Applications

### 🎯 Current Capabilities
1. **EEG-to-Embedding Generation** - Convert brain signals to 512-dim vectors
2. **Cross-Modal Alignment** - Align EEG with visual image space
3. **Digit Classification** - Classify mental imagery of digits 0-9
4. **Retrieval Tasks** - Find matching images from EEG signals

### 🚀 Future Extensions
1. **Image Generation** - Combine with diffusion models (Stable Diffusion)
2. **Real-time BCI** - Brain-computer interface applications
3. **Multi-modal Fusion** - Combine with other neural signals
4. **Transfer Learning** - Adapt to other visual tasks

## 📊 Data Integrity & Validation

### ✅ Verified Properties
- **No Data Leakage** - Complete separation between train/val/test
- **Balanced Distribution** - Equal samples per digit across splits
- **Embedding Consistency** - Stable generation across all splits
- **Proper Evaluation** - Unbiased performance metrics

### 🔍 Quality Assurance
- **Stratified Splitting** - Maintains digit distribution
- **Reproducible Results** - Fixed random seeds
- **Comprehensive Testing** - Multiple validation metrics
- **GPU Optimization** - Full CUDA acceleration

## 🏆 Key Achievements

### ✅ Technical Milestones
1. **Complete Pipeline** - End-to-end EEG processing to embeddings
2. **Proper Methodology** - Unbiased train/val/test evaluation
3. **GPU Acceleration** - Full CUDA optimization
4. **CLIP Integration** - Successful alignment with vision models
5. **Scalable Architecture** - Ready for improvements and extensions

### 📈 Research Impact
- **Baseline Established** - Solid foundation for brain-to-image research
- **Methodology Validated** - Proper evaluation framework
- **Open Source** - Reproducible research implementation
- **Extensible Design** - Easy to adapt for new applications

---

**🎯 This implementation provides a solid foundation for EEG-based brain-to-image reconstruction research with proper evaluation methodology and significant potential for improvements.**
