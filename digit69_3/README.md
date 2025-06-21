# Digit69_3 - fMRI to Image Reconstruction

## 🎯 Purpose

**Digit69_3** adalah folder development untuk **fMRI embedding to image reconstruction** menggunakan Latent Diffusion Models (LDM). Tujuan utama adalah merekonstruksi digit images dari fMRI embeddings.

**"From brain signals to visual digits - LDM-powered reconstruction"**

## 🧠 Concept

### **🔄 Reconstruction Pipeline:**
```
fMRI Embeddings (512D) → LDM Conditioning → Generated Digit Images (28x28)
         ↓                      ↓                        ↓
   Brain signals         Diffusion Process        Visual reconstruction
```

### **🎯 Goal:**
Menggunakan fMRI embeddings yang sudah di-generate dari `digit69_embeddings.pkl` sebagai conditioning input untuk Latent Diffusion Model, sehingga bisa merekonstruksi digit images dari brain signals.

## 📁 Project Structure

```
digit69_3/                           # fMRI-to-Image Reconstruction
├── digit69_embedding_converter.py   # Embedding converter (base)
├── digit69_embeddings.pkl          # ⭐ fMRI embeddings (59MB)
├── digit69_contrastive_clip.pth     # Trained encoder model (94MB)
├── digit69_ldm.py                   # 🚀 LDM reconstruction (planned)
├── train_ldm.py                     # 🚀 LDM training script (planned)
├── evaluate_reconstruction.py       # 🚀 Evaluation tools (planned)
├── runembedding.py                  # Original training script
├── digit69_embeddings_metadata.json # Metadata
└── README.md                        # This documentation
```

### **🎯 Development Phases:**
- ✅ **Phase 1**: Embedding conversion (completed)
- 🚀 **Phase 2**: LDM reconstruction (current)
- 📊 **Phase 3**: Evaluation & optimization (planned)

## 🔧 Core Components

### **📊 Phase 1: Embedding Generation (Completed)**
**Digit69EmbeddingConverter**
- Loads trained model: `digit69_contrastive_clip.pth`
- Processes dataset: `../dataset/digit69_28x28.mat`
- Converts fMRI (3092 voxels) + digit images → CLIP embeddings (512D)
- Generated: `digit69_embeddings.pkl` (59 MB)

### **🎨 Phase 2: LDM Reconstruction (Current Development)**
**Digit69LDM (Planned)**
- Input: fMRI embeddings (512D) from `digit69_embeddings.pkl`
- Architecture: Latent Diffusion Model conditioned on fMRI embeddings
- Output: Reconstructed digit images (28x28)
- Training: Supervised learning with paired fMRI-image data

### **📈 Phase 3: Evaluation (Planned)**
**ReconstructionEvaluator (Planned)**
- Metrics: SSIM, MSE, LPIPS, Perceptual similarity
- Comparison: Generated vs original digits
- Analysis: Reconstruction quality per digit class

## 🚀 Development Roadmap

### **✅ Phase 1: Embedding Generation (Completed)**
```python
from digit69_embedding_converter import Digit69EmbeddingConverter

# Load pre-generated embeddings
converter = Digit69EmbeddingConverter()
embeddings_data = converter.load_embeddings("digit69_embeddings.pkl")

# Available data:
# - fmri_embeddings: (90, 512) train, (10, 512) test
# - image_embeddings: (90, 512) train, (10, 512) test
# - original_images: (90, 3, 224, 224) train, (10, 3, 224, 224) test
```

### **🚀 Phase 2: LDM Reconstruction (Current)**
```python
# Planned implementation
from digit69_ldm import Digit69LDM

# Initialize LDM
ldm = Digit69LDM()

# Train LDM with fMRI conditioning
ldm.train(embeddings_data)

# Generate digit from fMRI embedding
fmri_emb = embeddings_data['test']['fmri_embeddings'][0]
reconstructed_digit = ldm.generate(fmri_emb)
```

### **📊 Phase 3: Evaluation (Planned)**
```python
# Planned evaluation framework
from evaluate_reconstruction import ReconstructionEvaluator

evaluator = ReconstructionEvaluator()
metrics = evaluator.evaluate(original_images, reconstructed_images)
# Returns: SSIM, MSE, LPIPS, perceptual similarity
```

## 🔍 Technical Approach

### **🎯 LDM Architecture (Planned)**
```python
class Digit69LDM:
    def __init__(self):
        self.unet = UNet2D(
            in_channels=4,          # Latent space
            out_channels=4,
            condition_dim=512       # fMRI embedding dimension
        )
        self.vae = AutoencoderKL()  # Encode/decode images
        self.scheduler = DDPMScheduler()

    def forward(self, fmri_embedding, timestep, latents):
        # Condition UNet with fMRI embedding
        return self.unet(latents, timestep, fmri_embedding)
```

### **📊 Training Strategy**
1. **Data Preparation**: Load fMRI embeddings + target images
2. **VAE Encoding**: Encode target images to latent space
3. **Noise Addition**: Add noise according to diffusion schedule
4. **Conditioning**: Use fMRI embeddings as conditioning input
5. **Denoising**: Train UNet to predict noise given fMRI condition
6. **Generation**: Sample from noise using trained model

### **🎨 Comparison with Miyawaki4**
| Aspect | **Miyawaki4** | **Digit69_3** |
|--------|---------------|---------------|
| Input | fMRI signals | fMRI embeddings |
| Method | Binary patterns → LDM | Direct fMRI → LDM |
| Output | Geometric shapes | Digit images |
| Conditioning | Binary patterns | CLIP embeddings |
| Complexity | Multi-stage | Single-stage |

## 📊 Expected Results

### **🎯 Reconstruction Quality Metrics**
- **SSIM**: Structural similarity with original digits
- **MSE**: Pixel-level reconstruction error
- **LPIPS**: Perceptual similarity (human-like evaluation)
- **Digit Classification**: Accuracy of reconstructed digits
- **Cross-modal Retrieval**: fMRI → correct digit matching

### **📈 Success Criteria**
- **Visual Quality**: Recognizable digit shapes
- **Class Accuracy**: Correct digit class reconstruction
- **Perceptual Similarity**: High LPIPS scores
- **Comparison**: Competitive with miyawaki4 approach

## 🔗 Integration & Applications

### **🎯 Research Applications**
- **Brain-Computer Interfaces**: Real-time digit visualization from brain
- **Neuroscience Research**: Understanding digit representation in brain
- **Cross-modal Learning**: fMRI-visual correspondence
- **Cognitive Studies**: Mental imagery reconstruction

### **🔧 Technical Integration**
- **Compatible format**: Same as miyawaki4_embeddings.pkl
- **Modular design**: Easy integration with other pipelines
- **Extensible**: Can adapt to other visual categories
- **Benchmarking**: Direct comparison with miyawaki4 results

## 🎯 Development Timeline

### **✅ Completed (Phase 1)**
- [x] fMRI embedding generation
- [x] CLIP-aligned embeddings (512D)
- [x] Dataset preprocessing (90 train, 10 test)
- [x] Embedding quality validation

### **🚀 Current Development (Phase 2)**
- [ ] LDM architecture design
- [ ] fMRI conditioning implementation
- [ ] Training pipeline setup
- [ ] Initial reconstruction experiments

### **📊 Planned (Phase 3)**
- [ ] Comprehensive evaluation metrics
- [ ] Comparison with miyawaki4
- [ ] Optimization and fine-tuning
- [ ] Documentation and analysis

## 🎉 Project Vision

**"From Brain Signals to Visual Digits"**

Digit69_3 aims to demonstrate that **fMRI embeddings can be used to reconstruct visual digit images** using state-of-the-art Latent Diffusion Models. This approach:

- **Simplifies** the reconstruction pipeline compared to miyawaki4
- **Leverages** powerful CLIP embeddings for conditioning
- **Demonstrates** direct brain-to-image generation
- **Provides** a foundation for real-time BCI applications

### 🔗 **Connection to Miyawaki4**
While miyawaki4 uses binary patterns as intermediate representation, digit69_3 explores **direct conditioning** of LDM with fMRI embeddings, potentially offering:
- Simpler pipeline
- Better semantic preservation
- More flexible generation
- Easier integration with modern generative models

**🚀 Ready to revolutionize brain-to-image reconstruction with digits!**