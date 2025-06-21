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

### Core Implementation
- **`miyawaki/`** - Complete Miyawaki dataset implementation ✅
  - Dual encoder architecture (5.4M parameters)
  - 83.3% cross-modal retrieval accuracy
  - Diffusion & GAN generative backends
  - Comprehensive evaluation framework

### Legacy Components
- **`clip.py`** - CLIP correlation learning module
- **`fmriencoder.py`** - fMRI encoder architecture
- **`stimencoder.py`** - Visual stimulus encoder
- **`diffusion.py`** - Diffusion decoder implementation
- **`gan.py`** - GAN decoder implementation
- **`corrlearning.py`** - Correlation learning utilities
- **`latentlearn.py`** - Latent space training
- **`genbetrain.py`** - Generative backend training
- **`metriks.py`** - Evaluation metrics
- **`analisys.py`** - Analysis utilities

### Datasets
- **`dataset/`** - fMRI-visual paired datasets
  - `miyawaki_structured_28x28.mat` - 107 train, 12 test samples ✅
  - `crell.mat` - 576 train, 64 test samples
  - `digit69_28x28.mat` - 90 train, 10 test samples
  - `mindbigdata.mat` - 1080 train, 120 test samples

### Documentation
- **`dataset_summary_report.md`** - Comprehensive dataset analysis
- **`miyawaki/README.md`** - Miyawaki implementation guide
- **`miyawaki/miyawaki_experiment_summary.md`** - Detailed results

## 🚀 Quick Start

### 1. Miyawaki Dataset (Recommended)
```bash
cd miyawaki
python miyawaki_dual_encoder.py      # Train dual encoder
python miyawaki_evaluation.py        # Evaluate performance
python miyawaki_generative_backend.py # Train generative backends
```

### 2. Dataset Analysis
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
1. **Scale to CRELL dataset** (576 samples, 10 classes)
2. **Test MINDBIGDATA** (1080 samples, largest dataset)
3. **Cross-dataset generalization** experiments

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

- ✅ **Miyawaki Dataset**: Complete implementation with strong results
- 🔄 **CRELL Dataset**: Ready for implementation
- 🔄 **DIGIT69 Dataset**: Ready for implementation
- 🔄 **MINDBIGDATA Dataset**: Ready for implementation
- ✅ **Framework**: Scalable and modular architecture

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