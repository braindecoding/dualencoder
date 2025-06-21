# Digit69 Embedding Converter

## 🎯 Purpose

**Digit69 Embedding Converter** mengkonversi dataset digit69 (fMRI + digit images) menjadi CLIP-aligned embeddings yang siap digunakan untuk downstream tasks. Mirip dengan miyawaki4 embedding converter, tapi untuk digit dataset.

**"Convert fMRI + digit images → embeddings untuk training downstream models"**

## 🔧 Main Component

**Digit69EmbeddingConverter**
- Loads trained model: `digit69_contrastive_clip.pth`
- Processes dataset: `digit69_28x28.mat`
- Converts fMRI (3092 voxels) + digit images → CLIP embeddings (512D)
- Saves embeddings dalam format yang sama dengan miyawaki4

## 📊 Generated Files

- **`digit69_embeddings.pkl`** - Complete embeddings data (58.99 MB)
- **`digit69_embeddings.npz`** - Numpy arrays (1.45 MB)
- **`digit69_embeddings_metadata.json`** - Metadata
- **`digit69_embedding_analysis.png`** - Analysis visualization

## 🚀 Usage

**Simple Usage:**
```python
from digit69_embedding_converter import Digit69EmbeddingConverter

# Initialize converter
converter = Digit69EmbeddingConverter("digit69_contrastive_clip.pth")

# Convert dataset to embeddings
embeddings_data = converter.convert_dataset_to_embeddings()

# Analyze embeddings
analysis_results = converter.analyze_embeddings(embeddings_data)
```

**Output Structure:**
```python
embeddings_data = {
    'train': {
        'fmri_embeddings': (90, 512),      # fMRI → CLIP embeddings
        'image_embeddings': (90, 512),     # Images → CLIP embeddings
        'original_fmri': (90, 3092),       # Original fMRI data
        'original_images': (90, 3, 224, 224)  # Original images
    },
    'test': {
        'fmri_embeddings': (10, 512),
        'image_embeddings': (10, 512),
        'original_fmri': (10, 3092),
        'original_images': (10, 3, 224, 224)
    },
    'metadata': {...}
}
```

## 📊 Results

**Dataset Statistics:**
- Training: 90 samples (fMRI + digit images)
- Testing: 10 samples (fMRI + digit images)
- fMRI dimension: 3092 → 512 embeddings
- Image dimension: 28×28 digits → 512 CLIP embeddings

**Similarity Analysis:**
- Training similarities: Mean 0.195 ± 0.036
- Test similarities: Mean 0.075 ± 0.025

## 🔗 Integration

**Compatible dengan miyawaki4:**
```python
# Load digit69 embeddings
embeddings_data = converter.load_embeddings("digit69_embeddings.pkl")

# Use dengan miyawaki4 pipeline
# Same format dengan miyawaki4_embeddings.pkl
```

**Downstream Applications:**
- Cross-modal retrieval dengan digit images
- Digit reconstruction dari fMRI
- Transfer learning ke datasets lain
- Comparison dengan geometric patterns (miyawaki4)

## 📁 File Structure

```
digit69_2/
├── digit69_embedding_converter.py    # Main converter
├── digit69_contrastive_clip.pth      # Trained model (94MB)
├── digit69_embeddings.pkl           # Generated embeddings (59MB)
├── digit69_embeddings.npz           # Numpy format (1.5MB)
├── digit69_embeddings_metadata.json # Metadata
├── digit69_embedding_analysis.png   # Analysis plot
└── README.md                        # This file
```

## ✅ Ready to Use

File `digit69_embeddings.pkl` sudah siap digunakan untuk downstream tasks dengan format yang sama seperti miyawaki4!