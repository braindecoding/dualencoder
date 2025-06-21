# Analisis Dataset fMRI-Image untuk Dual Encoder Project

## Overview
Proyek ini menggunakan 4 dataset utama yang berisi pasangan data fMRI dan image untuk pembelajaran korelasi menggunakan arsitektur dual encoder dengan CLIP-style correlation learning.

## Dataset Summary

### 1. CRELL Dataset (`crell.mat`)
- **Ukuran file**: 15.58 MB
- **Jenis**: Dataset digit MNIST dengan fMRI
- **Training samples**: 576
- **Test samples**: 64
- **Classes**: 10 digit (0-9), distribusi seimbang (~57-58 samples per class)

**Struktur Data:**
- `fmriTrn`: (576, 3092) - fMRI signals training
- `fmriTest`: (64, 3092) - fMRI signals test  
- `stimTrn`: (576, 784) - Image data training (28x28 pixels, 0-255)
- `stimTest`: (64, 784) - Image data test (28x28 pixels, 0-255)
- `labelTrn`: (576, 1) - Labels training (0-9)
- `labelTest`: (64, 1) - Labels test (0-9)

**Karakteristik:**
- fMRI: 3092 features, range [-0.22, 0.37], mean ~0.048
- Images: Inverted MNIST (background putih, digit hitam), mean pixel ~236
- Balanced dataset dengan distribusi class yang merata

### 2. DIGIT69 Dataset (`digit69_28x28.mat`)
- **Ukuran file**: 2.28 MB
- **Jenis**: Binary classification (digit 6 vs 9)
- **Training samples**: 90
- **Test samples**: 10
- **Classes**: 2 classes (1=digit 6, 2=digit 9), distribusi seimbang (45 samples each)

**Struktur Data:**
- `fmriTrn`: (90, 3092) - fMRI signals training
- `fmriTest`: (10, 3092) - fMRI signals test
- `stimTrn`: (90, 784) - Image data training (28x28 pixels, 0-255)
- `stimTest`: (10, 784) - Image data test (28x28 pixels, 0-255)
- `labelTrn`: (90, 1) - Labels training (1-2)
- `labelTest`: (10, 1) - Labels test (1-2)

**Karakteristik:**
- fMRI: 3092 features, range [-0.33, 0.58], mean ~0.011
- Images: Standard MNIST style (background hitam, digit putih), mean pixel ~33
- Small dataset, cocok untuk proof-of-concept binary classification

### 3. MINDBIGDATA Dataset (`mindbigdata.mat`)
- **Ukuran file**: 29.21 MB (terbesar)
- **Jenis**: Extended MNIST dataset
- **Training samples**: 1080
- **Test samples**: 120
- **Classes**: 10 digit (0-9), distribusi hampir seimbang (~97-115 samples per class)

**Struktur Data:**
- `fmriTrn`: (1080, 3092) - fMRI signals training
- `fmriTest`: (120, 3092) - fMRI signals test
- `stimTrn`: (1080, 784) - Image data training (28x28 pixels, 0-255)
- `stimTest`: (120, 784) - Image data test (28x28 pixels, 0-255)
- `labelTrn`: (1080, 1) - Labels training (0-9)
- `labelTest`: (120, 1) - Labels test (0-9)

**Karakteristik:**
- fMRI: 3092 features, range [-0.74, 0.68], mean ~0.013
- Images: Standard MNIST style, mean pixel ~31
- Largest dataset, ideal untuk training model yang robust

### 4. MIYAWAKI Dataset (`miyawaki_structured_28x28.mat`)
- **Ukuran file**: 1.59 MB
- **Jenis**: Structured visual stimuli (bukan MNIST)
- **Training samples**: 107
- **Test samples**: 12
- **Classes**: 4 classes (2, 3, 4, 5), distribusi tidak seimbang

**Struktur Data:**
- `fmriTrn`: (107, 967) - fMRI signals training (berbeda dimensi!)
- `fmriTest`: (12, 967) - fMRI signals test
- `stimTrn`: (107, 784) - Image data training (28x28 pixels, 0-1 normalized)
- `stimTest`: (12, 784) - Image data test (28x28 pixels, 0-1 normalized)
- `labelTrn`: (107, 1) - Labels training (2-5)
- `labelTest`: (12, 1) - Labels test (2-5)
- `train_indices`: (1, 107) - Training indices
- `test_indices`: (1, 12) - Test indices
- `metadata`: Structured metadata dengan informasi preprocessing

**Karakteristik:**
- fMRI: **967 features** (berbeda dari dataset lain!), range [-5.76, 4.99]
- Images: Normalized ke [0,1], mean pixel ~0.27
- Smallest dataset, mungkin untuk specific visual patterns

## Key Findings

### 1. Konsistensi Struktur
- **Image data**: Semua menggunakan 784 features (28x28 pixels)
- **fMRI data**: 3 dataset menggunakan 3092 features, Miyawaki menggunakan 967 features
- **Format**: Semua menggunakan train/test split dengan struktur variabel yang konsisten

### 2. Preprocessing Differences
- **CRELL**: Inverted MNIST (background putih)
- **DIGIT69 & MINDBIGDATA**: Standard MNIST (background hitam)
- **MIYAWAKI**: Normalized ke [0,1], bukan MNIST data

### 3. Dataset Characteristics
- **Size progression**: DIGIT69 (smallest) → MIYAWAKI → CRELL → MINDBIGDATA (largest)
- **Complexity**: Binary (DIGIT69) → Multi-class (others)
- **fMRI dimensionality**: Mayoritas 3092, Miyawaki 967

### 4. Implications untuk Dual Encoder
- **fMRI Encoder**: Perlu handle 2 dimensi input (3092 dan 967)
- **Image Encoder**: Konsisten 784 → latent space
- **CLIP Correlation**: Perlu robust terhadap variasi data distribution
- **Generative Backend**: Output 28x28 images dengan normalization yang berbeda

## Recommendations

### 1. Data Preprocessing
```python
# Standardize image normalization
def normalize_images(images):
    if images.max() > 1:
        return images / 255.0
    return images

# Handle different fMRI dimensions
def get_fmri_encoder(fmri_dim):
    if fmri_dim == 3092:
        return fMRI_Encoder_3092()
    elif fmri_dim == 967:
        return fMRI_Encoder_967()
```

### 2. Training Strategy
- **Start with DIGIT69**: Binary classification, small dataset
- **Scale to CRELL**: Multi-class, medium size
- **Validate on MINDBIGDATA**: Large dataset untuk robustness
- **Test generalization on MIYAWAKI**: Different data distribution

### 3. Architecture Considerations
- Flexible fMRI encoder untuk handle different input dimensions
- Consistent latent space dimension (512) untuk semua dataset
- Adaptive normalization untuk different image ranges

## Dataset Usage dalam Codebase
Berdasarkan analisis kode, dataset ini digunakan dalam pipeline:
1. **fMRI_Encoder**: `fmriTrn/fmriTest` → latent space
2. **Shape_Encoder**: `stimTrn/stimTest` → latent space  
3. **CLIP_Correlation**: Learn correlation between latent spaces
4. **Diffusion/GAN**: Generate images dari correlation + test fMRI

## Next Steps
1. Implement data loader yang handle semua 4 dataset
2. Create unified preprocessing pipeline
3. Test dual encoder architecture pada setiap dataset
4. Evaluate cross-dataset generalization
