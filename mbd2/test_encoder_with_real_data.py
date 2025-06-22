#!/usr/bin/env python3
"""
Test EEG Transformer Encoder with Real Preprocessed Data
Load correctly preprocessed MindBigData EEG and test encoder performance
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from eeg_transformer_encoder import EEGTransformerEncoder, MultiScaleEEGTransformerEncoder

def load_preprocessed_data():
    """
    Load correctly preprocessed EEG data
    """
    print("📂 Loading correctly preprocessed EEG data...")
    
    with open('mbd2/correctly_preprocessed_eeg_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    processed_data = dataset['correctly_processed_eeg_data']
    stats = dataset['processing_statistics']
    
    print(f"✅ Loaded preprocessed data:")
    print(f"   Total processed signals: {stats['processed_signals']:,}")
    print(f"   Rejection rate: {stats['rejection_rate']:.1%}")
    
    return processed_data, stats

def create_training_dataset(processed_data, max_samples_per_digit=400):
    """
    Create training dataset from preprocessed EEG data
    """
    print(f"🔧 Creating training dataset (max {max_samples_per_digit} per digit)...")
    
    electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    multi_electrode_signals = []
    labels = []
    
    for digit in range(10):  # Digits 0-9
        print(f"   Processing digit {digit}...")
        
        # Find minimum samples across all electrodes for this digit
        min_samples = min(len(processed_data[electrode][digit]) 
                         for electrode in electrodes 
                         if digit in processed_data[electrode])
        
        num_samples = min(min_samples, max_samples_per_digit)
        
        for sample_idx in range(num_samples):
            # Stack signals from all 14 electrodes
            multi_electrode_signal = []
            for electrode in electrodes:
                signal = processed_data[electrode][digit][sample_idx]
                multi_electrode_signal.append(signal)
            
            multi_electrode_signals.append(np.stack(multi_electrode_signal))  # Shape: (14, 256)
            labels.append(digit)
    
    eeg_signals = np.array(multi_electrode_signals)
    labels = np.array(labels)
    
    print(f"✅ Created training dataset:")
    print(f"   EEG signals shape: {eeg_signals.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Samples per digit: {len(labels) // 10}")
    
    return eeg_signals, labels

def test_encoder_classification(encoder, eeg_signals, labels, test_name="Encoder"):
    """
    Test encoder for EEG classification task with GPU support
    """
    print(f"\n🧪 Testing {test_name} for EEG Classification...")

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Convert to tensors and move to device
    eeg_tensor = torch.FloatTensor(eeg_signals).to(device)
    labels_tensor = torch.LongTensor(labels).to(device)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        eeg_tensor, labels_tensor, test_size=0.2, random_state=42, stratify=labels_tensor
    )
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create simple classifier on top of encoder
    class EEGClassifier(nn.Module):
        def __init__(self, encoder, num_classes=10):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Linear(encoder.d_model, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            features = self.encoder(x)
            return self.classifier(features)
    
    # Initialize model and move to device
    model = EEGClassifier(encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    model.train()
    num_epochs = 20
    batch_size = 32
    
    train_losses = []
    
    print(f"🚀 Training {test_name} classifier...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train accuracy
        train_outputs = model(X_train.to(device))
        train_preds = torch.argmax(train_outputs, dim=1)
        train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())

        # Test accuracy
        test_outputs = model(X_test.to(device))
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = accuracy_score(y_test.cpu().numpy(), test_preds.cpu().numpy())
    
    print(f"✅ {test_name} Classification Results:")
    print(f"   Train Accuracy: {train_acc:.3f}")
    print(f"   Test Accuracy: {test_acc:.3f}")
    
    # Detailed classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test.cpu().numpy(), test_preds.cpu().numpy(),
                              target_names=[f'Digit {i}' for i in range(10)]))
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_losses': train_losses,
        'model': model
    }

def visualize_encoder_features(encoder, eeg_signals, labels, encoder_name="Encoder"):
    """
    Visualize encoder features using t-SNE
    """
    print(f"\n🎨 Visualizing {encoder_name} features...")
    
    try:
        from sklearn.manifold import TSNE
        
        # Extract features
        device = next(encoder.parameters()).device
        encoder.eval()
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_signals[:1000]).to(device)  # Sample for visualization
            features = encoder(eeg_tensor).cpu().numpy()
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for digit in range(10):
            mask = labels[:1000] == digit
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.7)
        
        plt.title(f'{encoder_name} Feature Visualization (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'{encoder_name.lower().replace(" ", "_")}_features_tsne.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature visualization saved as {filename}")
        
    except ImportError:
        print("⚠️  scikit-learn not available for t-SNE visualization")

def main():
    """
    Main function to test EEG Transformer Encoder with real data
    """
    print("🧠 EEG TRANSFORMER ENCODER REAL DATA TEST")
    print("=" * 60)

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")

    # Load preprocessed data
    processed_data, _ = load_preprocessed_data()
    
    # Create training dataset
    eeg_signals, labels = create_training_dataset(processed_data, max_samples_per_digit=400)
    
    # Test Single-Scale Encoder
    print(f"\n{'='*60}")
    print("🔧 TESTING SINGLE-SCALE TRANSFORMER ENCODER")
    print("=" * 60)
    
    single_encoder = EEGTransformerEncoder(
        n_channels=14,
        seq_len=256,
        d_model=128,
        nhead=8,
        num_layers=6,
        patch_size=16,
        dropout=0.1
    ).to(device)
    
    single_results = test_encoder_classification(
        single_encoder, eeg_signals, labels, "Single-Scale Transformer"
    )
    
    # Test Multi-Scale Encoder
    print(f"\n{'='*60}")
    print("🔧 TESTING MULTI-SCALE TRANSFORMER ENCODER")
    print("=" * 60)
    
    multi_encoder = MultiScaleEEGTransformerEncoder(
        n_channels=14,
        seq_len=256,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    multi_results = test_encoder_classification(
        multi_encoder, eeg_signals, labels, "Multi-Scale Transformer"
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print("📊 ENCODER COMPARISON RESULTS")
    print("=" * 60)
    print(f"Single-Scale Transformer:")
    print(f"   Train Accuracy: {single_results['train_accuracy']:.3f}")
    print(f"   Test Accuracy: {single_results['test_accuracy']:.3f}")
    print(f"Multi-Scale Transformer:")
    print(f"   Train Accuracy: {multi_results['train_accuracy']:.3f}")
    print(f"   Test Accuracy: {multi_results['test_accuracy']:.3f}")
    
    # Feature visualization
    visualize_encoder_features(single_encoder, eeg_signals, labels, "Single-Scale Transformer")
    visualize_encoder_features(multi_encoder, eeg_signals, labels, "Multi-Scale Transformer")
    
    print(f"\n🎉 EEG Transformer Encoder real data testing completed!")
    
    return {
        'single_encoder': single_encoder,
        'multi_encoder': multi_encoder,
        'single_results': single_results,
        'multi_results': multi_results,
        'data': (eeg_signals, labels)
    }

if __name__ == "__main__":
    results = main()
