#!/usr/bin/env python3
"""
Enhanced EEG Transformer Encoder with Improved Architecture
- Deeper layers (8 vs 6)
- Larger model dimension (256 vs 128)
- Multi-head attention improvements
- Better positional encoding
- Advanced embedding projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AdvancedEmbeddingProjector(nn.Module):
    """
    Advanced embedding generation with residual connections and attention
    """
    def __init__(self, input_dim=256, embedding_dim=512, normalize=True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Multi-layer projection with residual connections
        self.projector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(input_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, embedding_dim)
        
        # Self-attention for embedding refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final normalization
        if normalize:
            self.final_norm = nn.Tanh()
        else:
            self.final_norm = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Advanced embedding projection with residual connections
        
        Args:
            x: [batch, input_dim] - transformer features
            
        Returns:
            embeddings: [batch, embedding_dim] - refined embeddings
        """
        # Store input for residual connection
        residual = self.residual_proj(x)  # [batch, embedding_dim]
        
        # Progressive projection
        out = x
        for layer in self.projector:
            out = layer(out)
        
        # Add residual connection
        out = out + residual
        
        # Self-attention refinement
        # Reshape for attention: [batch, 1, embedding_dim]
        out_expanded = out.unsqueeze(1)
        attended, _ = self.self_attention(out_expanded, out_expanded, out_expanded)
        out = attended.squeeze(1)  # [batch, embedding_dim]
        
        # Layer normalization and final activation
        out = self.layer_norm(out)
        out = self.final_norm(out)
        
        return out

class EnhancedPositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with learnable components
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Learnable positional embedding
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        
        # Mixing weight
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: [seq_len, batch, d_model]
        seq_len = x.size(0)
        
        # Combine sinusoidal and learnable encodings
        sinusoidal = self.pe[:seq_len, :]
        learnable = self.learnable_pe[:seq_len, :].unsqueeze(1)
        
        # Weighted combination
        pos_encoding = torch.sigmoid(self.mix_weight) * sinusoidal + \
                      (1 - torch.sigmoid(self.mix_weight)) * learnable
        
        return self.dropout(x + pos_encoding)

class EnhancedEEGTransformerEncoder(nn.Module):
    """
    Enhanced EEG Transformer with improved architecture
    """
    def __init__(self, n_channels=14, seq_len=256, d_model=256, nhead=8, 
                 num_layers=8, patch_size=16, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        print(f"🧠 Enhanced EEG Transformer Configuration:")
        print(f"   Input: [{n_channels} channels, {seq_len} timepoints]")
        print(f"   Patches: {self.num_patches} patches of size {patch_size}")
        print(f"   Model dimension: {d_model} (enhanced)")
        print(f"   Attention heads: {nhead}")
        print(f"   Transformer layers: {num_layers} (enhanced)")
        
        # Enhanced spatial preprocessing
        self.spatial_projection = nn.Sequential(
            nn.Linear(n_channels, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Enhanced patch embedding with depthwise convolution
        self.patch_embed = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=patch_size, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        self.patch_norm = nn.LayerNorm(d_model)
        
        # Enhanced positional encoding
        self.pos_encoding = EnhancedPositionalEncoding(d_model, max_len=self.num_patches, dropout=dropout)
        
        # Enhanced transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',  # GELU instead of ReLU
            batch_first=False,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention-based pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Advanced embedding projector
        self.embedding_projector = AdvancedEmbeddingProjector(
            input_dim=d_model,
            embedding_dim=d_model,
            normalize=True
        )
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Enhanced forward pass
        
        Args:
            x: [batch, 14, 256] - EEG signals
            
        Returns:
            encoded: [batch, d_model] - enhanced encoded representation
        """
        batch_size = x.size(0)
        
        # Input validation
        assert x.size(1) == self.n_channels, f"Expected {self.n_channels} channels, got {x.size(1)}"
        assert x.size(2) == self.seq_len, f"Expected {self.seq_len} timepoints, got {x.size(2)}"
        
        # Enhanced spatial mixing
        x = x.transpose(-1, -2)  # [batch, 256, 14]
        x = self.spatial_projection(x)  # [batch, 256, d_model]
        x = x.transpose(-1, -2)  # [batch, d_model, 256]
        
        # Enhanced temporal patching
        x = self.patch_embed(x)  # [batch, d_model, num_patches]

        # Prepare for transformer
        x = x.transpose(-1, -2)  # [batch, num_patches, d_model]
        x = self.patch_norm(x)   # Apply layer norm after transpose
        x = x.transpose(0, 1)    # [num_patches, batch, d_model]
        
        # Enhanced positional encoding (before adding CLS token)
        x = self.pos_encoding(x)

        # Add CLS token for attention pooling
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # [1, batch, d_model]
        x = torch.cat([cls_tokens, x], dim=0)  # [num_patches+1, batch, d_model]
        
        # Enhanced transformer processing
        x = self.transformer(x)  # [num_patches+1, batch, d_model]
        
        # Enhanced global aggregation
        cls_output = x[0]  # [batch, d_model] - CLS token output
        patch_outputs = x[1:]  # [num_patches, batch, d_model]
        
        # Multiple pooling strategies
        patch_outputs = patch_outputs.transpose(0, 1)  # [batch, num_patches, d_model]
        patch_outputs = patch_outputs.transpose(-1, -2)  # [batch, d_model, num_patches]
        
        avg_pool = self.global_pool(patch_outputs).squeeze(-1)  # [batch, d_model]
        max_pool = self.global_max_pool(patch_outputs).squeeze(-1)  # [batch, d_model]
        
        # Combine different representations
        combined = (cls_output + avg_pool + max_pool) / 3
        
        # Final processing
        combined = self.layer_norm(combined)
        
        # Generate enhanced embeddings
        embeddings = self.embedding_projector(combined)
        
        return embeddings

class EnhancedEEGToEmbeddingModel(nn.Module):
    """
    Complete Enhanced EEG-to-Embedding model
    """
    def __init__(self, n_channels=14, seq_len=256, d_model=256, embedding_dim=512,
                 nhead=8, num_layers=8, patch_size=16, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Enhanced encoder
        self.encoder = EnhancedEEGTransformerEncoder(
            n_channels=n_channels,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout=dropout
        )
        
        # Embedding adaptation if needed
        if embedding_dim != d_model:
            self.embedding_adapter = nn.Sequential(
                nn.Linear(d_model, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Tanh()
            )
        else:
            self.embedding_adapter = nn.Identity()
        
        print(f"🧠 Enhanced EEG-to-Embedding Model:")
        print(f"   Input: [{n_channels} channels, {seq_len} timepoints]")
        print(f"   Model dimension: {d_model} (enhanced)")
        print(f"   Final embedding dimension: {embedding_dim}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, x):
        """
        Generate enhanced embeddings from EEG signals
        
        Args:
            x: [batch, 14, 256] - EEG signals
            
        Returns:
            embeddings: [batch, embedding_dim] - enhanced embeddings
        """
        # Enhanced encoding
        features = self.encoder(x)  # [batch, d_model]
        
        # Adapt to final embedding dimension
        embeddings = self.embedding_adapter(features)  # [batch, embedding_dim]
        
        return embeddings
    
    def get_embeddings(self, eeg_signals, batch_size=32):
        """
        Generate embeddings for a batch of EEG signals
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

def test_enhanced_model():
    """
    Test enhanced model architecture
    """
    print("🧪 Testing Enhanced EEG Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Test parameters
    batch_size = 8
    n_channels = 14
    seq_len = 256
    
    # Create test data
    test_eeg = torch.randn(batch_size, n_channels, seq_len).to(device)
    print(f"📊 Test input shape: {test_eeg.shape}")
    
    # Test enhanced model
    model = EnhancedEEGToEmbeddingModel(
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=256,
        embedding_dim=512,
        nhead=8,
        num_layers=8,
        patch_size=16,
        dropout=0.1
    ).to(device)
    
    with torch.no_grad():
        embeddings = model(test_eeg)
        print(f"✅ Enhanced model output shape: {embeddings.shape}")
        print(f"✅ Output device: {embeddings.device}")
        print(f"✅ Output range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    print(f"🎉 Enhanced model test completed successfully!")
    return model

if __name__ == "__main__":
    test_enhanced_model()
