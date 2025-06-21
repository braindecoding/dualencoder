# CLIP Embedding
## Kita mengajarkan model: "fMRI ini artinya sama dengan image ini"
fmri_embedding = fmri_encoder(fmri_signal)      # "Kata asing"
image_embedding = clip_model.encode_image(img)  # "Terjemahan yang benar"

## Loss: Paksa fMRI embedding mirip dengan CLIP embedding
loss = contrastive_loss(fmri_embedding, image_embedding)

## Komponen Utama

**1. MiyawakiDataset**
- Handles loading fMRI (967 voxels) dan stimuli (28x28 images)
- Converts stimuli ke format yang kompatibel dengan CLIP
- Applies proper preprocessing untuk CLIP input

**2. fMRIEncoder** 
- Neural network yang map fMRI signals ke CLIP embedding space (512-dim)
- Architecture: 967 → 2048 → 1024 → 512 dengan BatchNorm dan Dropout
- Output di-normalize untuk contrastive learning

**3. ContrastiveLoss**
- Symmetric contrastive loss untuk align fMRI dan image embeddings
- Temperature parameter untuk control learning dynamics

## Cara Penggunaan

**Step 1: Persiapan Data**
```python
# Pastikan file .mat Anda memiliki struktur:
# - fmriTrn: (n_samples, 967)
# - stimTrn: (n_samples, 784) 
# - fmriTest: (n_samples, 967)
# - stimTest: (n_samples, 784)

decoder = MiyawakiDecoder()
decoder.load_data("path/to/miyawaki_data.mat")
```

**Step 2: Training**
```python
# Initialize models
decoder.initialize_models()

# Create dataloaders  
train_loader, test_loader = decoder.create_dataloaders(batch_size=32)

# Train (akan butuh GPU untuk speed optimal)
train_losses = decoder.train(train_loader, epochs=100, lr=1e-3)
```

**Step 3: Evaluation**
```python
# Evaluate dengan retrieval metrics
results, similarity_matrix = decoder.evaluate(test_loader)

# Hasil akan show Top-1, Top-5, Top-10 accuracy
# Top-1 accuracy = berapa % fMRI bisa retrieve correct image di rank 1
```

## Metrics yang Digunakan

**Retrieval Accuracy:**
- **Top-k Accuracy**: Persentase fMRI signals yang bisa retrieve correct image dalam top-k candidates
- **Similarity Matrix**: Cosine similarity antara fMRI embeddings dan image embeddings

**Expected Performance:**
- Top-1: ~15-25% (good result untuk 28x28 images)
- Top-5: ~35-50% 
- Top-10: ~50-65%

## Visualisasi

Code akan generate:
1. **Training loss curve** - monitoring convergence
2. **Retrieval examples** - comparison original vs retrieved images
3. **Similarity distributions** - ranking patterns

## Tips Optimization

**Hyperparameter Tuning:**
```python
# Experiment dengan:
temperature = [0.05, 0.07, 0.1]  # Contrastive loss temperature
lr = [1e-4, 1e-3, 5e-3]         # Learning rate
hidden_dims = [[1024, 512], [2048, 1024], [4096, 2048, 1024]]
```

**Architecture Variants:**
```python
# Untuk better performance, coba:
1. Residual connections
2. Attention mechanisms  
3. Multi-head projections
4. Different CLIP models (ViT-L/14, RN50x4)
```

## Troubleshooting

**Memory Issues:**
- Reduce batch_size ke 16 atau 8
- Use gradient checkpointing
- Process data dalam chunks

**Convergence Issues:**
- Lower learning rate
- Increase temperature
- Add warm-up schedule

**Poor Retrieval:**
- Check data normalization
- Verify CLIP preprocessing
- Try different architectures

Apakah Anda sudah punya file .mat dataset Miyawaki? Jika ya, kita bisa langsung test implementasi ini! Jika belum, saya bisa bantu generate synthetic data untuk testing pipeline.