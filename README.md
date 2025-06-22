# Dual Encoder for fMRI-Visual Cross-Modal Learning

A complete implementation of dual encoder architecture with CLIP-style correlation learning for cross-modal alignment between fMRI brain signals and visual stimuli.

## 🧠 Architecture Overview

```
fMRI (X) → X_encoder → X_lat ╲
                              ╲ CLIP_corr → Diffusion/GAN → Y_result
Shapes (Y) → Y_encoder → Y_lat ╱           ╱
                               ╱          ╱
                         X_test ────────╱
```

### Pipeline Flow
```matlab
% Encoding Phase
X_lat = fMRI_Encoder(X)           % fMRI → latent space
Y_lat = Shape_Encoder(Y)          % Shapes → latent space

% Correlation Learning
CLIP_corr = CLIP_Correlation(X_lat, Y_lat)

% Decoding Phase
Y_result_diff = Diffusion(CLIP_corr, X_test)
Y_result_gan = GAN(CLIP_corr, X_test)

% Evaluation
accuracy_diff = Evaluate(Y_result_diff, Y_test)
accuracy_gan = Evaluate(Y_result_gan, Y_test)
```

## 📁 Project Structure

### Core Implementations
- **`miyawaki/`** - Complete working implementation ✅
  - Dual encoder architecture (5.4M parameters)
  - 83.3% cross-modal retrieval accuracy
  - Diffusion & GAN generative backends
  - Comprehensive evaluation framework

- **`miyawaki2/`** - Advanced modular architecture 🚧
  - Sophisticated component-based design
  - OpenAI CLIP integration
  - U-Net diffusion with proper scheduler
  - Advanced GAN with discriminator
  - Three-phase training pipeline
  - **Status**: Fixing dependencies and integration

### Datasets & Processing
- **`dataset/`** - fMRI-visual paired datasets
  - `miyawaki_structured_28x28.mat` - 107 train, 12 test samples ✅
  - `crell.mat` - 576 train, 64 test samples
  - `digit69_28x28.mat` - 90 train, 10 test samples
  - `mindbigdata.mat` - 1080 train, 120 test samples

- **`crell/`** - EEG-to-Letter reconstruction ✅
  - 640 epochs, 64 channels, 500 Hz sampling
  - 10 letters: a,d,e,f,j,n,o,s,t,v
  - CORRECT preprocessing pipeline applied
  - Ready for EEG-to-image modeling

- **`mbd3/`** - MindBigData EEG digit processing 🚧
  - Multiple model variants and experiments
  - Enhanced text templates and perceptual guidance
  - Advanced diffusion models for digit reconstruction

### Documentation
- **`dataset_summary_report.md`** - Comprehensive dataset analysis
- **`miyawaki/README.md`** - Miyawaki implementation guide
- **`miyawaki/miyawaki_experiment_summary.md`** - Detailed results

## 🚀 Quick Start

### 1. Working Implementation (Recommended)
```bash
cd miyawaki
python miyawaki_dual_encoder.py      # Train dual encoder
python miyawaki_evaluation.py        # Evaluate performance
python miyawaki_generative_backend.py # Train generative backends
```

### 2. Advanced Modular Architecture (Development)
```bash
cd miyawaki2
# Note: Currently fixing dependencies and integration
# Will be the primary implementation once completed
```

### 3. Dataset Analysis
```bash
python analyze_datasets.py           # Analyze all datasets
python detailed_image_analysis.py    # Detailed image analysis
```

## 📊 Key Results (Miyawaki Dataset)

### 🎯 Performance Metrics
- **Cross-modal Retrieval**: 83.3% top-1 accuracy
- **Classification**: 100% (stimulus), 83.3% (correlation)
- **Generation Quality**: MSE 0.0428 (diffusion), 0.2162 (GAN)
- **Training Stability**: Converged without overfitting

### 🏗️ Architecture Details
- **Total Parameters**: 5.4M
- **fMRI Encoder**: 967 → 512 features
- **Stimulus Encoder**: 28×28 → 512 features
- **CLIP Correlation**: 1024 → 512 features
- **Training Time**: ~100 epochs for dual encoder

### 🔬 Technical Insights
✅ **CLIP-style learning** highly effective for brain-visual alignment
✅ **Unit sphere normalization** critical for stable training
✅ **Simple MLP decoders** sufficient for structured data
✅ **Diffusion approach** superior to GAN for reconstruction

## 🎯 Next Steps

### Immediate
1. **Implement EEG-to-Letter models** for Crell dataset (640 epochs, 10 letters) ✅
2. **Scale to CRELL fMRI dataset** (576 samples, 10 classes)
3. **Test MINDBIGDATA** (1080 samples, largest dataset)
4. **Cross-dataset generalization** experiments

### Research Directions
1. **Attention mechanisms** for better feature alignment
2. **Multi-scale processing** for complex visual patterns
3. **Temporal modeling** for dynamic fMRI sequences
4. **Real-time applications** for brain-computer interfaces

## 📋 Requirements

```bash
torch >= 1.9.0
torchvision
numpy
scipy
matplotlib
scikit-learn
```

## 🏆 Status

### Implementation Status
- ✅ **Miyawaki (Working)**: Complete implementation with 83.3% accuracy
- 🚧 **Miyawaki2 (Advanced)**: Modular architecture under development
- ✅ **Crell EEG Dataset**: Processed with CORRECT preprocessing (640 epochs)
- 🔄 **CRELL fMRI Dataset**: Ready for implementation
- 🔄 **DIGIT69 Dataset**: Ready for implementation
- 🔄 **MINDBIGDATA Dataset**: Ready for implementation

### Architecture Status
- ✅ **Monolithic Design**: Proven working implementation
- 🚧 **Modular Design**: Advanced architecture with dependency fixes needed
- ✅ **Evaluation Framework**: Comprehensive metrics and visualization

---

**Current Achievement**: 83.3% cross-modal retrieval accuracy on Miyawaki dataset
**Next Target**: Scale to larger datasets and improve generalization


============================================================
SUMMARY PERBANDINGAN DATASET
============================================================
detailed_image_analysis.py


CRELL:
  fmriTrn: (576, 3092) (float64)
  fmriTest: (64, 3092) (float64)
  stimTrn: (576, 784) (uint8)
  stimTest: (64, 784) (uint8)
  labelTrn: (576, 1) (uint8)
  labelTest: (64, 1) (uint8)

DIGIT69_28X28:
  fmriTest: (10, 3092) (float64)
  fmriTrn: (90, 3092) (float64)
  labelTest: (10, 1) (uint8)
  labelTrn: (90, 1) (uint8)
  stimTest: (10, 784) (uint8)
  stimTrn: (90, 784) (uint8)

MINDBIGDATA:
  fmriTrn: (1080, 3092) (float64)
  fmriTest: (120, 3092) (float64)
  stimTrn: (1080, 784) (uint8)
  stimTest: (120, 784) (uint8)
  labelTrn: (1080, 1) (uint8)
  labelTest: (120, 1) (uint8)