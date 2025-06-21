# Miyawaki2 - Modular Architecture Implementation

This folder contains the **modular architecture** implementation from the root folder. This represents a more sophisticated, component-based approach to dual encoder design.

## 📁 File Structure

### 🏗️ Core Architecture Components
- **`fmriencoder.py`** - fMRI encoder (967 → 2048 → 1024 → 512)
- **`stimencoder.py`** - Visual stimulus encoder (CNN + FC → 512)
- **`clip.py`** - CLIP-style correlation learning with OpenAI CLIP integration
- **`diffusion.py`** - Advanced diffusion decoder with U-Net architecture
- **`gan.py`** - Sophisticated GAN decoder with discriminator

### 🎓 Training Components
- **`latentlearn.py`** - Phase 1: Train latent encoders
- **`corrlearning.py`** - Phase 2: Train correlation learning
- **`genbetrain.py`** - Phase 3: Train generative backends
- **`miyawakitrain.py`** - Complete training pipeline orchestrator

### 📊 Evaluation & Analysis
- **`metriks.py`** - Comprehensive evaluation metrics (MSE, SSIM, LPIPS, etc.)
- **`analisys.py`** - Qualitative analysis and visualization
- **`evaluatepipeline.py`** - End-to-end evaluation pipeline

### ⚙️ Configuration & Utilities
- **`hyperparam.py`** - Optimal hyperparameters configuration
- **`store.py`** - Correlation storage and retrieval system
- **`miyawakidataset.py`** - Corrected dataset loader (fixed data leaking)

## 🎯 Architecture Philosophy

### Modular Design Principles
1. **Separation of Concerns** - Each component has single responsibility
2. **Reusability** - Components can be used across different datasets
3. **Extensibility** - Easy to add new features or modify existing ones
4. **Testability** - Individual components can be tested independently

### Three-Phase Training Pipeline
```
Phase 1: Latent Encoders
├── fMRI_Encoder: 967 → 512
├── Shape_Encoder: 28×28 → 512
└── CLIP Contrastive Loss

Phase 2: Correlation Learning
├── CLIP_Correlation: 1024 → 512
├── Advanced contrastive learning
└── OpenAI CLIP integration

Phase 3: Generative Backends
├── Diffusion_Decoder: U-Net + DDPM
├── GAN_Decoder: ConvTranspose + Discriminator
└── Multi-objective training
```

## 🔧 Current Status

### ✅ Strengths
- **Advanced Architecture** - Sophisticated design with modern techniques
- **Modular Structure** - Clean separation of components
- **Comprehensive Features** - Multiple decoders, metrics, analysis tools
- **Scalable Design** - Ready for larger datasets and complex scenarios

### ❌ Current Issues
- **Missing Imports** - Many files lack proper import statements
- **Undefined Dependencies** - References to undefined functions/classes
- **No Integration** - Components not connected in working pipeline
- **Incomplete Implementation** - Many functions are stubs or incomplete

### 🚧 Required Fixes
```python
# Missing imports needed:
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2D, DDPMScheduler
import clip
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips

# Undefined classes/functions to implement:
- UNet2D configuration
- LPIPS distance calculation
- Shape classification accuracy
- Contour matching score
- Edge preservation metric
```

## 🔄 Comparison with Miyawaki Folder

| Aspect | Miyawaki (Working) | Miyawaki2 (Modular) | Best Approach |
|--------|-------------------|---------------------|---------------|
| **Architecture** | Monolithic | Modular | Combine both |
| **Completeness** | ✅ Complete | ❌ Incomplete | Fix miyawaki2 |
| **Sophistication** | Basic | Advanced | Use miyawaki2 |
| **Maintainability** | Limited | Excellent | Use miyawaki2 |
| **Testing** | ✅ Working | ❌ Broken | Fix imports |

## 🚀 Development Roadmap

### Phase 1: Fix Dependencies (Immediate)
1. **Add missing imports** to all files
2. **Implement undefined functions** (LPIPS, SSIM, etc.)
3. **Create working main pipeline** connecting all components
4. **Test basic functionality** on Miyawaki dataset

### Phase 2: Integration (Short-term)
1. **Merge best practices** from miyawaki folder
2. **Create unified training script** with proper data loading
3. **Implement comprehensive evaluation** with all metrics
4. **Add proper error handling** and logging

### Phase 3: Enhancement (Medium-term)
1. **Scale to larger datasets** (CRELL, MINDBIGDATA)
2. **Add advanced features** (attention, multi-scale processing)
3. **Optimize performance** and memory usage
4. **Create production-ready pipeline**

## 💡 Key Innovations

### Advanced Features Not in Miyawaki
1. **OpenAI CLIP Integration** - Real CLIP model for reference
2. **U-Net Diffusion** - Proper diffusion with scheduler
3. **Advanced GAN** - ConvTranspose with discriminator
4. **Comprehensive Metrics** - LPIPS, SSIM, perceptual metrics
5. **Modular Training** - Three-phase training pipeline

### Design Patterns
- **Factory Pattern** - For creating different decoders
- **Strategy Pattern** - For different training phases
- **Observer Pattern** - For metrics collection
- **Builder Pattern** - For complex model construction

## 🎯 Usage (Once Fixed)

### Quick Start (Future)
```python
# Initialize trainer
trainer = MiyawakaTrainer()

# Three-phase training
trainer.train_phase1_encoders(train_loader, epochs=100)
trainer.train_phase2_correlation(train_loader, epochs=50)
trainer.train_phase3_decoders(train_loader, test_loader, epochs=100)

# Evaluation
accuracy_diff, accuracy_gan = evaluate_model(trainer, test_loader)
```

### Component Usage
```python
# Use individual components
fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512)
shape_encoder = Shape_Encoder(latent_dim=512)
clip_correlation = CLIP_Correlation(latent_dim=512)

# Advanced decoders
diffusion_decoder = Diffusion_Decoder(correlation_dim=512)
gan_decoder = GAN_Decoder(correlation_dim=512)
```

## 📋 Dependencies (To Install)

```bash
# Core dependencies
pip install torch torchvision
pip install diffusers transformers
pip install clip-by-openai

# Evaluation metrics
pip install torchmetrics
pip install lpips
pip install scikit-image

# Utilities
pip install scipy matplotlib
pip install wandb  # For experiment tracking
```

## 🏆 Future Vision

This modular architecture represents the **future direction** of the dual encoder project:

1. **Scalable** - Can handle multiple datasets and modalities
2. **Extensible** - Easy to add new features and improvements
3. **Maintainable** - Clean code structure for long-term development
4. **Production-Ready** - Designed for real-world applications

Once the dependency issues are resolved, this will become the **primary implementation** for the dual encoder project, replacing the monolithic approach in the miyawaki folder.

---

**Status**: 🚧 Under Development - Fixing Dependencies  
**Priority**: High - Core architecture for future development  
**Goal**: Replace miyawaki folder as primary implementation
