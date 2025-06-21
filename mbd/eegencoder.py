# Typical EEG data structure
EEG_raw = {
    'data': np.array([channels, timepoints]),  # e.g., [64, 5000]
    'channels': 64,     # Number of electrodes
    'sampling_rate': 1000,  # Hz
    'duration': 5.0,    # seconds
    'events': [...],    # Stimulus markers
}

# Example for single trial:
#shape: [64 channels × 1000 timepoints] = 64,000 raw values


def preprocess_eeg(raw_eeg):
    # 1. Bandpass filtering
    filtered = butter_bandpass_filter(raw_eeg, 0.1, 100, fs=1000)
    
    # 2. Artifact removal (ICA)
    clean_eeg = remove_artifacts(filtered)
    
    # 3. Epoching (extract relevant time windows)
    epochs = extract_epochs(clean_eeg, onset=0, duration=1.0)  # 0-1000ms
    
    # 4. Baseline correction
    baseline_corrected = baseline_correction(epochs, baseline=(-200, 0))
    
    # 5. Normalization
    normalized = zscore_normalize(baseline_corrected)
    
    return normalized  # Shape: [64, 1000]

def train_eeg_encoder(eeg_encoder, dataloader):
    for batch in dataloader:
        eeg_signals, images = batch
        
        # Generate EEG embeddings
        eeg_embeddings = eeg_encoder(eeg_signals)
        
        # Get CLIP embeddings for corresponding images
        with torch.no_grad():
            clip_embeddings = clip_model.encode_image(images)
        
        # Contrastive loss
        loss = contrastive_loss(eeg_embeddings, clip_embeddings)
        
        # Backward pass
        loss.backward()
        optimizer.step()

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Separable convolution untuk temporal patterns
        self.temporal_conv = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(1, 25),  # 25ms temporal kernel
            padding=(0, 12),
            groups=40,  # Depthwise convolution
            bias=False
        )
        self.pointwise_conv = nn.Conv2d(40, 40, 1)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d((1, 4))  # Downsample temporal
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Input: [batch, 40, 1, 1000]
        x = self.temporal_conv(x)
        x = self.pointwise_conv(x)
        x = self.elu(x)
        x = self.avgpool(x)  # [batch, 40, 1, 250]
        x = self.dropout(x)
        return x
    
class DeepFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(40, 80, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(80),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
            
            # Layer 2
            nn.Conv2d(80, 160, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        
    def forward(self, x):
        # Progressive feature abstraction
        x = self.conv_layers(x)  # [batch, 160, 1, 15]
        return x
    
class EmbeddingProjector(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(160 * 15, 512),  # 160 features × 15 timepoints
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),  # Final embedding
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
    def forward(self, x):
        x = self.flatten(x)  # [batch, 2400]
        embeddings = self.fc_layers(x)  # [batch, 128]
        return embeddings
    

class EEGTransformerEncoder(nn.Module):
    def __init__(self, n_channels=64, seq_len=1000, d_model=128):
        super().__init__()
        # Patch embedding for EEG
        self.patch_embed = nn.Conv1d(n_channels, d_model, kernel_size=50, stride=25)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch, 64, 1000]
        x = self.patch_embed(x)  # [batch, 128, 39] - 39 patches
        x = x.transpose(1, 2)  # [batch, 39, 128]
        x = self.pos_encoding(x)
        
        # Transformer processing
        x = x.transpose(0, 1)  # [39, batch, 128] for transformer
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch, 39, 128]
        
        # Global representation
        x = x.transpose(1, 2)  # [batch, 128, 39]
        x = self.global_pool(x).squeeze(-1)  # [batch, 128]
        
        return x