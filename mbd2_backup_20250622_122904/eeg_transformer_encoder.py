#!/usr/bin/env python3
"""
EEG Transformer Encoder for MindBigData Dataset
Optimized for correctly preprocessed EEG signals (14 channels, 256 timepoints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EmbeddingProjector(nn.Module):
    """
    Final embedding generation layer for EEG features
    Projects transformer features to normalized embeddings
    """
    def __init__(self, input_dim=128, embedding_dim=128, normalize=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.normalize = normalize

        # Progressive projection layers
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(input_dim, embedding_dim),
        )

        # Final normalization layer
        if normalize:
            self.final_norm = nn.Tanh()  # Normalize to [-1, 1]
        else:
            self.final_norm = nn.Identity()

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Project features to final embeddings

        Args:
            x: [batch, input_dim] - transformer features

        Returns:
            embeddings: [batch, embedding_dim] - final normalized embeddings
        """
        # Project to embedding space
        embeddings = self.projector(x)  # [batch, embedding_dim]

        # Layer normalization
        embeddings = self.layer_norm(embeddings)

        # Final normalization
        embeddings = self.final_norm(embeddings)

        return embeddings

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :]

class EEGTransformerEncoder(nn.Module):
    """
    Transformer-based EEG encoder for MindBigData dataset
    
    Input: [batch, 14, 256] - 14 electrodes × 256 timepoints (correctly preprocessed)
    Output: [batch, d_model] - encoded representation
    """
    def __init__(self, n_channels=14, seq_len=256, d_model=128, nhead=8, 
                 num_layers=6, patch_size=16, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size  # 256 // 16 = 16 patches
        
        print(f"🧠 EEG Transformer Encoder Configuration:")
        print(f"   Input: [{n_channels} channels, {seq_len} timepoints]")
        print(f"   Patches: {self.num_patches} patches of size {patch_size}")
        print(f"   Model dimension: {d_model}")
        print(f"   Attention heads: {nhead}")
        print(f"   Transformer layers: {num_layers}")
        
        # STAGE 1: Spatial preprocessing (mix electrode information)
        self.spatial_projection = nn.Sequential(
            nn.Linear(n_channels, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # STAGE 2: Patch embedding (temporal patching)
        # Convert 256 timepoints into patches for transformer processing
        self.patch_embed = nn.Conv1d(d_model, d_model, 
                                   kernel_size=patch_size, 
                                   stride=patch_size)
        
        # STAGE 3: Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # STAGE 4: Self-attention processing (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False  # [seq, batch, features]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # STAGE 5: Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final embedding projector
        self.embedding_projector = EmbeddingProjector(
            input_dim=d_model,
            embedding_dim=d_model,
            normalize=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass through EEG Transformer Encoder
        
        Args:
            x: [batch, 14, 256] - correctly preprocessed EEG signals
            
        Returns:
            encoded: [batch, d_model] - encoded EEG representation
        """
        # Input validation
        assert x.size(1) == self.n_channels, f"Expected {self.n_channels} channels, got {x.size(1)}"
        assert x.size(2) == self.seq_len, f"Expected {self.seq_len} timepoints, got {x.size(2)}"
        
        # STAGE 1: Spatial mixing (mix electrode information)
        # [batch, 14, 256] -> [batch, 256, 14] -> [batch, 256, d_model]
        x = x.transpose(-1, -2)  # [batch, 256, 14]
        x = self.spatial_projection(x)  # [batch, 256, d_model]
        x = x.transpose(-1, -2)  # [batch, d_model, 256]
        
        # STAGE 2: Temporal patching (create patches for transformer)
        # [batch, d_model, 256] -> [batch, d_model, num_patches]
        x = self.patch_embed(x)  # [batch, d_model, 16]
        
        # Prepare for transformer: [seq, batch, features]
        x = x.transpose(-1, -2)  # [batch, 16, d_model]
        x = x.transpose(0, 1)    # [16, batch, d_model]
        
        # STAGE 3: Add positional encoding
        x = self.pos_encoding(x)  # [16, batch, d_model]
        
        # STAGE 4: Self-attention processing
        x = self.transformer(x)  # [16, batch, d_model]
        
        # STAGE 5: Global aggregation
        # [16, batch, d_model] -> [batch, d_model, 16] -> [batch, d_model, 1] -> [batch, d_model]
        x = x.transpose(0, 1)  # [batch, 16, d_model]
        x = x.transpose(-1, -2)  # [batch, d_model, 16]
        x = self.global_pool(x)  # [batch, d_model, 1]
        x = x.squeeze(-1)  # [batch, d_model]
        
        # Final processing
        x = self.layer_norm(x)

        # Generate final embeddings
        embeddings = self.embedding_projector(x)

        return embeddings

class MultiScaleEEGTransformerEncoder(nn.Module):
    """
    Multi-scale EEG Transformer Encoder
    Processes EEG at different temporal scales for better representation
    """
    def __init__(self, n_channels=14, seq_len=256, d_model=128, nhead=8, 
                 num_layers=4, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Multiple scales of patch sizes
        self.patch_sizes = [8, 16, 32]  # Different temporal resolutions
        self.encoders = nn.ModuleList()
        
        for patch_size in self.patch_sizes:
            # Ensure d_model is divisible by nhead
            scale_d_model = d_model // len(self.patch_sizes)
            scale_d_model = (scale_d_model // nhead) * nhead  # Make divisible by nhead

            encoder = EEGTransformerEncoder(
                n_channels=n_channels,
                seq_len=seq_len,
                d_model=scale_d_model,
                nhead=nhead,
                num_layers=num_layers,
                patch_size=patch_size,
                dropout=dropout
            )
            self.encoders.append(encoder)
        
        # Calculate actual concatenated dimension
        total_dim = sum((d_model // len(self.patch_sizes) // nhead) * nhead
                       for _ in self.patch_sizes)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Final embedding projector
        self.embedding_projector = EmbeddingProjector(
            input_dim=d_model,
            embedding_dim=d_model,
            normalize=True
        )
        
    def forward(self, x):
        """
        Multi-scale encoding
        
        Args:
            x: [batch, 14, 256] - correctly preprocessed EEG signals
            
        Returns:
            fused: [batch, d_model] - multi-scale encoded representation
        """
        # Encode at different scales
        scale_features = []
        for encoder in self.encoders:
            features = encoder(x)
            scale_features.append(features)
        
        # Concatenate multi-scale features
        fused = torch.cat(scale_features, dim=-1)  # [batch, total_dim]

        # Final fusion
        fused = self.fusion(fused)  # [batch, d_model]

        # Generate final embeddings
        embeddings = self.embedding_projector(fused)  # [batch, d_model]

        return embeddings

class EEGToEmbeddingModel(nn.Module):
    """
    Complete EEG-to-Embedding model
    Combines EEG Transformer Encoder with specialized embedding generation
    """
    def __init__(self, n_channels=14, seq_len=256, d_model=128, embedding_dim=128,
                 encoder_type='single', nhead=8, num_layers=6, patch_size=16, dropout=0.1):
        super().__init__()

        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim

        # Choose encoder type
        if encoder_type == 'single':
            self.encoder = EEGTransformerEncoder(
                n_channels=n_channels,
                seq_len=seq_len,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                patch_size=patch_size,
                dropout=dropout
            )
        elif encoder_type == 'multi':
            self.encoder = MultiScaleEEGTransformerEncoder(
                n_channels=n_channels,
                seq_len=seq_len,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Additional embedding refinement (optional)
        if embedding_dim != d_model:
            self.embedding_adapter = nn.Sequential(
                nn.Linear(d_model, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Tanh()
            )
        else:
            self.embedding_adapter = nn.Identity()

        print(f"🧠 EEG-to-Embedding Model Configuration:")
        print(f"   Encoder type: {encoder_type}")
        print(f"   Input: [{n_channels} channels, {seq_len} timepoints]")
        print(f"   Model dimension: {d_model}")
        print(f"   Final embedding dimension: {embedding_dim}")

    def forward(self, x):
        """
        Generate embeddings from EEG signals

        Args:
            x: [batch, 14, 256] - correctly preprocessed EEG signals

        Returns:
            embeddings: [batch, embedding_dim] - final normalized embeddings
        """
        # Encode EEG signals
        features = self.encoder(x)  # [batch, d_model]

        # Adapt to final embedding dimension if needed
        embeddings = self.embedding_adapter(features)  # [batch, embedding_dim]

        return embeddings

    def get_embeddings(self, eeg_signals, batch_size=32):
        """
        Generate embeddings for a batch of EEG signals

        Args:
            eeg_signals: [N, 14, 256] - batch of EEG signals
            batch_size: batch size for processing

        Returns:
            embeddings: [N, embedding_dim] - generated embeddings
        """
        self.eval()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(eeg_signals), batch_size):
                batch = eeg_signals[i:i+batch_size]
                if torch.is_tensor(batch):
                    batch_embeddings = self(batch)
                else:
                    batch_tensor = torch.FloatTensor(batch).to(next(self.parameters()).device)
                    batch_embeddings = self(batch_tensor)

                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

def test_eeg_transformer_encoder():
    """
    Test function for EEG Transformer Encoder with GPU support
    """
    print("🧪 Testing EEG Transformer Encoder...")

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test parameters
    batch_size = 8
    n_channels = 14
    seq_len = 256
    d_model = 128

    # Create test data (simulating correctly preprocessed EEG)
    test_eeg = torch.randn(batch_size, n_channels, seq_len).to(device)
    print(f"📊 Test input shape: {test_eeg.shape}")
    print(f"📊 Test input device: {test_eeg.device}")
    
    # Test single-scale encoder
    print("\n🔧 Testing Single-Scale Encoder...")
    encoder = EEGTransformerEncoder(
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=d_model,
        nhead=8,
        num_layers=6,
        patch_size=16
    ).to(device)

    with torch.no_grad():
        encoded = encoder(test_eeg)
        print(f"✅ Single-scale output shape: {encoded.shape}")
        print(f"✅ Single-scale output device: {encoded.device}")
        print(f"   Expected: [{batch_size}, {d_model}]")
    
    # Test multi-scale encoder
    print("\n🔧 Testing Multi-Scale Encoder...")
    multi_encoder = MultiScaleEEGTransformerEncoder(
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=d_model,
        nhead=8,
        num_layers=4
    ).to(device)

    with torch.no_grad():
        multi_encoded = multi_encoder(test_eeg)
        print(f"✅ Multi-scale output shape: {multi_encoded.shape}")
        print(f"✅ Multi-scale output device: {multi_encoded.device}")
        print(f"   Expected: [{batch_size}, {d_model}]")
    
    # Model complexity analysis
    single_params = sum(p.numel() for p in encoder.parameters())
    multi_params = sum(p.numel() for p in multi_encoder.parameters())
    
    # Test complete EEG-to-Embedding models
    print("\n🔧 Testing Complete EEG-to-Embedding Models...")

    # Single-scale embedding model
    single_embedding_model = EEGToEmbeddingModel(
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=d_model,
        embedding_dim=128,
        encoder_type='single'
    ).to(device)

    # Multi-scale embedding model
    multi_embedding_model = EEGToEmbeddingModel(
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=d_model,
        embedding_dim=128,
        encoder_type='multi'
    ).to(device)

    with torch.no_grad():
        single_embeddings = single_embedding_model(test_eeg)
        multi_embeddings = multi_embedding_model(test_eeg)

        print(f"✅ Single-scale embeddings shape: {single_embeddings.shape}")
        print(f"✅ Single-scale embeddings device: {single_embeddings.device}")
        print(f"✅ Single-scale embeddings range: [{single_embeddings.min():.3f}, {single_embeddings.max():.3f}]")

        print(f"✅ Multi-scale embeddings shape: {multi_embeddings.shape}")
        print(f"✅ Multi-scale embeddings device: {multi_embeddings.device}")
        print(f"✅ Multi-scale embeddings range: [{multi_embeddings.min():.3f}, {multi_embeddings.max():.3f}]")

    # Model complexity analysis
    single_params = sum(p.numel() for p in encoder.parameters())
    multi_params = sum(p.numel() for p in multi_encoder.parameters())
    single_embed_params = sum(p.numel() for p in single_embedding_model.parameters())
    multi_embed_params = sum(p.numel() for p in multi_embedding_model.parameters())

    print(f"\n📊 Model Complexity:")
    print(f"   Single-scale encoder: {single_params:,} parameters")
    print(f"   Multi-scale encoder: {multi_params:,} parameters")
    print(f"   Single-scale embedding model: {single_embed_params:,} parameters")
    print(f"   Multi-scale embedding model: {multi_embed_params:,} parameters")

    print(f"\n🎉 EEG Transformer Encoder with Embedding Generation tests completed successfully!")

    return {
        'encoder': encoder,
        'multi_encoder': multi_encoder,
        'single_embedding_model': single_embedding_model,
        'multi_embedding_model': multi_embedding_model
    }

if __name__ == "__main__":
    # Run tests
    results = test_eeg_transformer_encoder()
