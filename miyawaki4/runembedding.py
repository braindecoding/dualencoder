import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import clip
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

class MiyawakiDataset(Dataset):
    """Dataset class untuk data Miyawaki"""
    def __init__(self, fmri_data, stim_data, transform=None):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stim_data = stim_data
        self.transform = transform
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        
        # Convert stimuli (784,) ke (28, 28) kemudian ke PIL Image
        stim = self.stim_data[idx].reshape(28, 28)
        # Normalize ke 0-255 dan convert ke PIL
        stim = ((stim - stim.min()) / (stim.max() - stim.min()) * 255).astype(np.uint8)
        stim_img = Image.fromarray(stim).convert('RGB')
        
        if self.transform:
            stim_img = self.transform(stim_img)
            
        return fmri, stim_img

class fMRIEncoder(nn.Module):
    """Encoder untuk mapping fMRI ke CLIP embedding space"""
    def __init__(self, fmri_dim=967, clip_dim=512, hidden_dims=[2048, 1024]):
        super().__init__()
        
        layers = []
        prev_dim = fmri_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final projection ke CLIP space
        layers.append(nn.Linear(prev_dim, clip_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, fmri):
        # L2 normalize output untuk contrastive learning
        encoded = self.encoder(fmri)
        return F.normalize(encoded, dim=-1)

class ContrastiveLoss(nn.Module):
    """Contrastive loss untuk alignment fMRI dan image embeddings"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, fmri_features, image_features):
        # Compute cosine similarity
        logits = torch.matmul(fmri_features, image_features.T) / self.temperature
        
        # Labels adalah diagonal (setiap fMRI match dengan image yang sama)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size).to(logits.device)
        
        # Symmetric loss
        loss_fmri_to_image = F.cross_entropy(logits, labels)
        loss_image_to_fmri = F.cross_entropy(logits.T, labels)
        
        return (loss_fmri_to_image + loss_image_to_fmri) / 2

class MiyawakiDecoder:
    """Main class untuk training dan evaluation"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.fmri_encoder = None
        self.clip_model = None
        self.clip_preprocess = None
        self.scaler = StandardScaler()
        
        print(f"Using device: {self.device}")
        
    def load_data(self, mat_file_path):
        """Load data dari file .mat"""
        print("Loading data from .mat file...")
        data = loadmat(mat_file_path)
        
        # Extract data
        fmri_train = data['fmriTrn']  # (n_samples, 967)
        stim_train = data['stimTrn']  # (n_samples, 784)
        fmri_test = data['fmriTest']  # (n_samples, 967)
        stim_test = data['stimTest']  # (n_samples, 784)
        
        print(f"Training: fMRI {fmri_train.shape}, Stimuli {stim_train.shape}")
        print(f"Testing: fMRI {fmri_test.shape}, Stimuli {stim_test.shape}")
        
        # Normalize fMRI data
        fmri_train_norm = self.scaler.fit_transform(fmri_train)
        fmri_test_norm = self.scaler.transform(fmri_test)
        
        self.fmri_train = fmri_train_norm
        self.stim_train = stim_train
        self.fmri_test = fmri_test_norm
        self.stim_test = stim_test
        
        return fmri_train_norm, stim_train, fmri_test_norm, stim_test
    
    def initialize_models(self):
        """Initialize CLIP dan fMRI encoder"""
        print("Initializing CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        print("Initializing fMRI encoder...")
        self.fmri_encoder = fMRIEncoder(
            fmri_dim=967, 
            clip_dim=512,
            hidden_dims=[2048, 1024]
        ).to(self.device)
        
        print(f"fMRI Encoder parameters: {sum(p.numel() for p in self.fmri_encoder.parameters())}")
    
    def create_dataloaders(self, batch_size=32):
        """Create train dan test dataloaders"""
        
        # CLIP preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP input size
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        train_dataset = MiyawakiDataset(self.fmri_train, self.stim_train, transform)
        test_dataset = MiyawakiDataset(self.fmri_test, self.stim_test, transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, epochs=100, lr=1e-3):
        """Training loop"""
        print("Starting training...")
        
        optimizer = optim.Adam(self.fmri_encoder.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = ContrastiveLoss(temperature=0.07)
        
        train_losses = []
        
        for epoch in range(epochs):
            self.fmri_encoder.train()
            epoch_loss = 0.0
            
            for batch_idx, (fmri, images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                fmri = fmri.to(self.device)
                images = images.to(self.device)
                
                # Get embeddings
                fmri_emb = self.fmri_encoder(fmri)
                
                with torch.no_grad():
                    image_emb = self.clip_model.encode_image(images)
                    image_emb = F.normalize(image_emb.float(), dim=-1)
                
                # Compute loss
                loss = criterion(fmri_emb, image_emb)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        return train_losses
    
    def evaluate(self, test_loader, top_k=[1, 5, 10]):
        """Evaluation dengan retrieval metrics"""
        print("Evaluating model...")
        
        self.fmri_encoder.eval()
        
        all_fmri_emb = []
        all_image_emb = []
        
        with torch.no_grad():
            for fmri, images in tqdm(test_loader, desc="Computing embeddings"):
                fmri = fmri.to(self.device)
                images = images.to(self.device)
                
                # Get embeddings
                fmri_emb = self.fmri_encoder(fmri)
                image_emb = self.clip_model.encode_image(images)
                image_emb = F.normalize(image_emb.float(), dim=-1)
                
                all_fmri_emb.append(fmri_emb.cpu())
                all_image_emb.append(image_emb.cpu())
        
        # Concatenate all embeddings
        fmri_embeddings = torch.cat(all_fmri_emb, dim=0)
        image_embeddings = torch.cat(all_image_emb, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(fmri_embeddings, image_embeddings.T)
        
        # Compute retrieval metrics
        results = {}
        n_samples = similarity_matrix.shape[0]
        
        for k in top_k:
            # Top-k accuracy: berapa persen fMRI yang bisa retrieve correct image di top-k
            _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
            correct_retrieval = 0
            
            for i in range(n_samples):
                if i in top_k_indices[i]:
                    correct_retrieval += 1
            
            accuracy = correct_retrieval / n_samples
            results[f'top_{k}_accuracy'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.4f}")
        
        return results, similarity_matrix
    
    def visualize_results(self, test_loader, similarity_matrix, n_examples=5):
        """Visualisasi hasil reconstruction"""
        print("Creating visualizations...")
        
        # Get original stimuli untuk comparison
        test_stimuli = []
        for _, images in test_loader:
            # Convert balik ke numpy untuk visualization
            for img in images:
                # Denormalize
                img_denorm = img * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                img_denorm += torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                test_stimuli.append(img_denorm.permute(1, 2, 0).numpy())
        
        # Plot examples
        _, axes = plt.subplots(n_examples, 3, figsize=(12, 4*n_examples))
        
        for i in range(n_examples):
            # Top retrieved image
            _, top_idx = torch.topk(similarity_matrix[i], 1)
            retrieved_idx = top_idx[0].item()
            
            # Original stimulus (target)
            axes[i, 0].imshow(test_stimuli[i])
            axes[i, 0].set_title(f"Original Stimulus {i}")
            axes[i, 0].axis('off')
            
            # Retrieved image
            axes[i, 1].imshow(test_stimuli[retrieved_idx])
            axes[i, 1].set_title(f"Retrieved (idx: {retrieved_idx})")
            axes[i, 1].axis('off')
            
            # Similarity scores
            similarities = similarity_matrix[i].numpy()
            axes[i, 2].bar(range(min(10, len(similarities))), 
                          sorted(similarities, reverse=True)[:10])
            axes[i, 2].set_title("Top 10 Similarities")
            axes[i, 2].set_xlabel("Rank")
            axes[i, 2].set_ylabel("Similarity")
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'fmri_encoder_state_dict': self.fmri_encoder.state_dict(),
            'scaler': self.scaler
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        if self.fmri_encoder is None:
            self.initialize_models()

        self.fmri_encoder.load_state_dict(checkpoint['fmri_encoder_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"Model loaded from {filepath}")

def test_dataset_path():
    """Test if dataset path is accessible"""
    from pathlib import Path

    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")

    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path.absolute()}")
        return True
    else:
        print(f"❌ Dataset not found: {dataset_path.absolute()}")
        print("💡 Available files in dataset folder:")
        dataset_dir = Path("../dataset")
        if dataset_dir.exists():
            for file in dataset_dir.glob("*.mat"):
                print(f"   - {file.name}")
        return False

# Usage example
def main():
    """Main function untuk menjalankan training dan evaluation"""

    # Test dataset path first
    if not test_dataset_path():
        print("❌ Cannot proceed without dataset. Please check dataset path.")
        return None, None

    # Initialize decoder
    decoder = MiyawakiDecoder()

    # Load data
    mat_file_path = "../dataset/miyawaki_structured_28x28.mat"  # Updated path
    decoder.load_data(mat_file_path)
    
    # Initialize models
    decoder.initialize_models()
    
    # Create dataloaders
    train_loader, test_loader = decoder.create_dataloaders(batch_size=32)
    
    # Train model
    train_losses = decoder.train(train_loader, epochs=100, lr=1e-3)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.show()
    
    # Evaluate model
    results, similarity_matrix = decoder.evaluate(test_loader)
    
    # Visualize results
    decoder.visualize_results(test_loader, similarity_matrix)
    
    # Save model
    decoder.save_model("miyawaki_contrastive_clip.pth")
    
    return decoder, results

if __name__ == "__main__":
    # Run the main function
    print("🚀 Starting Miyawaki3 CLIP-based Training")
    print("=" * 50)

    try:
        decoder, results = main()
        print("\n✅ Training and evaluation completed successfully!")
        print(f"📊 Results: {results}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install torch torchvision clip-by-openai pillow tqdm scipy scikit-learn matplotlib")