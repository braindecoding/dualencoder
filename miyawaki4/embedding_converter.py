#!/usr/bin/env python3
"""
Embedding Converter for Miyawaki4
Convert training data to embeddings for downstream tasks
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import clip
from PIL import Image
import torchvision.transforms as transforms
from runembedding import MiyawakiDecoder
import pickle
import json
from tqdm import tqdm

class EmbeddingConverter:
    """Convert fMRI and images to CLIP embeddings"""
    
    def __init__(self, model_path="miyawaki_contrastive_clip.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.decoder = None
        self.embeddings_cache = {}
        
        print(f"🔧 Initializing Embedding Converter")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_path}")
        
    def load_trained_model(self):
        """Load the trained miyawaki model"""
        print("\n🔍 Loading Trained Model")
        print("=" * 40)
        
        if not Path(self.model_path).exists():
            print(f"❌ Model file not found: {self.model_path}")
            print("💡 Please train the model first using runembedding.py")
            return False
        
        # Initialize decoder
        self.decoder = MiyawakiDecoder()
        
        # Load data to get proper dimensions
        mat_file_path = "../dataset/miyawaki_structured_28x28.mat"
        self.decoder.load_data(mat_file_path)
        
        # Initialize models (this loads CLIP)
        self.decoder.initialize_models()
        
        # Load trained weights
        self.decoder.load_model(self.model_path)
        
        print(f"✅ Model loaded successfully")
        return True
    
    def convert_dataset_to_embeddings(self, save_embeddings=True):
        """Convert entire dataset to embeddings"""
        print("\n🔄 Converting Dataset to Embeddings")
        print("=" * 50)
        
        if self.decoder is None:
            if not self.load_trained_model():
                return None
        
        # Create dataloaders
        train_loader, test_loader = self.decoder.create_dataloaders(batch_size=32)
        
        # Convert training data
        train_embeddings = self._convert_dataloader_to_embeddings(train_loader, "Training")
        
        # Convert test data
        test_embeddings = self._convert_dataloader_to_embeddings(test_loader, "Testing")
        
        # Combine results
        embeddings_data = {
            'train': train_embeddings,
            'test': test_embeddings,
            'metadata': {
                'model_path': self.model_path,
                'fmri_dim': 967,
                'clip_dim': 512,
                'device': self.device,
                'train_samples': len(train_embeddings['fmri_embeddings']),
                'test_samples': len(test_embeddings['fmri_embeddings'])
            }
        }
        
        if save_embeddings:
            self._save_embeddings(embeddings_data)
        
        return embeddings_data
    
    def _convert_dataloader_to_embeddings(self, dataloader, split_name):
        """Convert a dataloader to embeddings"""
        print(f"\n📊 Converting {split_name} Data")
        
        self.decoder.fmri_encoder.eval()
        
        fmri_embeddings = []
        image_embeddings = []
        original_fmri = []
        original_images = []
        
        with torch.no_grad():
            for batch_idx, (fmri_batch, image_batch) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
                fmri_batch = fmri_batch.to(self.device)
                image_batch = image_batch.to(self.device)
                
                # Get fMRI embeddings (from trained encoder)
                fmri_emb = self.decoder.fmri_encoder(fmri_batch)
                fmri_emb = F.normalize(fmri_emb, dim=1)
                
                # Get CLIP image embeddings
                image_emb = self.decoder.clip_model.encode_image(image_batch)
                image_emb = F.normalize(image_emb.float(), dim=1)
                
                # Store embeddings
                fmri_embeddings.append(fmri_emb.cpu().numpy())
                image_embeddings.append(image_emb.cpu().numpy())
                
                # Store original data
                original_fmri.append(fmri_batch.cpu().numpy())
                
                # Convert images back to numpy for storage
                images_np = []
                for img in image_batch:
                    # Denormalize CLIP preprocessing
                    img_denorm = img * torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img.device)
                    img_denorm += torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img.device)
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    images_np.append(img_denorm.cpu().numpy())
                
                original_images.extend(images_np)
        
        # Concatenate all batches
        fmri_embeddings = np.concatenate(fmri_embeddings, axis=0)
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        original_fmri = np.concatenate(original_fmri, axis=0)
        original_images = np.array(original_images)
        
        print(f"✅ {split_name} conversion completed:")
        print(f"   📊 fMRI embeddings: {fmri_embeddings.shape}")
        print(f"   📊 Image embeddings: {image_embeddings.shape}")
        print(f"   📊 Original fMRI: {original_fmri.shape}")
        print(f"   📊 Original images: {original_images.shape}")
        
        return {
            'fmri_embeddings': fmri_embeddings,
            'image_embeddings': image_embeddings,
            'original_fmri': original_fmri,
            'original_images': original_images
        }
    
    def _save_embeddings(self, embeddings_data):
        """Save embeddings to files"""
        print("\n💾 Saving Embeddings")
        print("=" * 30)
        
        # Save as pickle (complete data)
        pickle_path = "miyawaki4_embeddings.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"✅ Complete embeddings saved: {pickle_path}")
        
        # Save as numpy arrays (for easy loading)
        np.savez_compressed("miyawaki4_embeddings.npz",
                           train_fmri_emb=embeddings_data['train']['fmri_embeddings'],
                           train_image_emb=embeddings_data['train']['image_embeddings'],
                           test_fmri_emb=embeddings_data['test']['fmri_embeddings'],
                           test_image_emb=embeddings_data['test']['image_embeddings'],
                           train_original_fmri=embeddings_data['train']['original_fmri'],
                           test_original_fmri=embeddings_data['test']['original_fmri'])
        print(f"✅ Numpy embeddings saved: miyawaki4_embeddings.npz")
        
        # Save metadata
        metadata_path = "miyawaki4_embeddings_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(embeddings_data['metadata'], f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Print file sizes
        for file_path in [pickle_path, "miyawaki4_embeddings.npz", metadata_path]:
            if Path(file_path).exists():
                size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                print(f"   📁 {file_path}: {size_mb:.2f} MB")
    
    def load_embeddings(self, file_path="miyawaki4_embeddings.pkl"):
        """Load saved embeddings"""
        print(f"\n📥 Loading Embeddings from {file_path}")
        
        if not Path(file_path).exists():
            print(f"❌ Embeddings file not found: {file_path}")
            return None
        
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                embeddings_data = pickle.load(f)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            embeddings_data = {
                'train': {
                    'fmri_embeddings': data['train_fmri_emb'],
                    'image_embeddings': data['train_image_emb'],
                    'original_fmri': data['train_original_fmri']
                },
                'test': {
                    'fmri_embeddings': data['test_fmri_emb'],
                    'image_embeddings': data['test_image_emb'],
                    'original_fmri': data['test_original_fmri']
                }
            }
        else:
            print(f"❌ Unsupported file format: {file_path}")
            return None
        
        print(f"✅ Embeddings loaded successfully")
        print(f"   📊 Training samples: {len(embeddings_data['train']['fmri_embeddings'])}")
        print(f"   📊 Test samples: {len(embeddings_data['test']['fmri_embeddings'])}")
        
        return embeddings_data
    
    def analyze_embeddings(self, embeddings_data):
        """Analyze the converted embeddings"""
        print("\n🔍 Analyzing Embeddings")
        print("=" * 40)
        
        train_fmri = embeddings_data['train']['fmri_embeddings']
        train_image = embeddings_data['train']['image_embeddings']
        test_fmri = embeddings_data['test']['fmri_embeddings']
        test_image = embeddings_data['test']['image_embeddings']
        
        # Compute similarities
        train_similarities = np.sum(train_fmri * train_image, axis=1)
        test_similarities = np.sum(test_fmri * test_image, axis=1)
        
        print(f"📊 Embedding Statistics:")
        print(f"   Training fMRI embeddings: {train_fmri.shape}")
        print(f"   Training image embeddings: {train_image.shape}")
        print(f"   Test fMRI embeddings: {test_fmri.shape}")
        print(f"   Test image embeddings: {test_image.shape}")
        
        print(f"\n🔗 Similarity Statistics:")
        print(f"   Training similarities - Mean: {train_similarities.mean():.3f}, Std: {train_similarities.std():.3f}")
        print(f"   Test similarities - Mean: {test_similarities.mean():.3f}, Std: {test_similarities.std():.3f}")
        
        # Visualize similarity distributions
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(train_similarities, bins=20, alpha=0.7, label='Training', color='blue')
        plt.hist(test_similarities, bins=20, alpha=0.7, label='Test', color='red')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('fMRI-Image Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(train_similarities, range(len(train_similarities)), alpha=0.6, label='Training', s=20)
        plt.scatter(test_similarities, range(len(test_similarities)), alpha=0.6, label='Test', s=20)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Sample Index')
        plt.title('Similarity by Sample')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('miyawaki4_embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Analysis plot saved: miyawaki4_embedding_analysis.png")
        
        return {
            'train_similarities': train_similarities,
            'test_similarities': test_similarities,
            'train_stats': {'mean': train_similarities.mean(), 'std': train_similarities.std()},
            'test_stats': {'mean': test_similarities.mean(), 'std': test_similarities.std()}
        }
    
    def create_embedding_dataset(self, embeddings_data, output_format='pytorch'):
        """Create ready-to-use dataset from embeddings"""
        print(f"\n🎯 Creating {output_format.upper()} Dataset")
        print("=" * 40)
        
        if output_format == 'pytorch':
            return self._create_pytorch_dataset(embeddings_data)
        elif output_format == 'numpy':
            return self._create_numpy_dataset(embeddings_data)
        else:
            print(f"❌ Unsupported format: {output_format}")
            return None
    
    def _create_pytorch_dataset(self, embeddings_data):
        """Create PyTorch dataset"""
        from torch.utils.data import Dataset, DataLoader
        
        class EmbeddingDataset(Dataset):
            def __init__(self, fmri_emb, image_emb, original_fmri=None):
                self.fmri_emb = torch.FloatTensor(fmri_emb)
                self.image_emb = torch.FloatTensor(image_emb)
                self.original_fmri = torch.FloatTensor(original_fmri) if original_fmri is not None else None
                
            def __len__(self):
                return len(self.fmri_emb)
            
            def __getitem__(self, idx):
                if self.original_fmri is not None:
                    return self.fmri_emb[idx], self.image_emb[idx], self.original_fmri[idx]
                else:
                    return self.fmri_emb[idx], self.image_emb[idx]
        
        train_dataset = EmbeddingDataset(
            embeddings_data['train']['fmri_embeddings'],
            embeddings_data['train']['image_embeddings'],
            embeddings_data['train']['original_fmri']
        )
        
        test_dataset = EmbeddingDataset(
            embeddings_data['test']['fmri_embeddings'],
            embeddings_data['test']['image_embeddings'],
            embeddings_data['test']['original_fmri']
        )
        
        print(f"✅ PyTorch datasets created:")
        print(f"   📊 Training dataset: {len(train_dataset)} samples")
        print(f"   📊 Test dataset: {len(test_dataset)} samples")
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'train_loader': DataLoader(train_dataset, batch_size=32, shuffle=True),
            'test_loader': DataLoader(test_dataset, batch_size=32, shuffle=False)
        }
    
    def _create_numpy_dataset(self, embeddings_data):
        """Create numpy dataset"""
        dataset = {
            'X_train_fmri': embeddings_data['train']['fmri_embeddings'],
            'X_train_image': embeddings_data['train']['image_embeddings'],
            'X_test_fmri': embeddings_data['test']['fmri_embeddings'],
            'X_test_image': embeddings_data['test']['image_embeddings'],
            'original_train_fmri': embeddings_data['train']['original_fmri'],
            'original_test_fmri': embeddings_data['test']['original_fmri']
        }
        
        print(f"✅ Numpy dataset created:")
        print(f"   📊 Training fMRI: {dataset['X_train_fmri'].shape}")
        print(f"   📊 Training images: {dataset['X_train_image'].shape}")
        print(f"   📊 Test fMRI: {dataset['X_test_fmri'].shape}")
        print(f"   📊 Test images: {dataset['X_test_image'].shape}")
        
        return dataset

def main():
    """Main function to convert dataset to embeddings"""
    print("🔧 Miyawaki4 Embedding Converter")
    print("=" * 50)
    
    # Initialize converter
    converter = EmbeddingConverter("miyawaki_contrastive_clip.pth")
    
    # Convert dataset to embeddings
    embeddings_data = converter.convert_dataset_to_embeddings(save_embeddings=True)
    
    if embeddings_data is None:
        print("❌ Failed to convert embeddings")
        return
    
    # Analyze embeddings
    analysis_results = converter.analyze_embeddings(embeddings_data)
    
    # Create PyTorch dataset
    pytorch_dataset = converter.create_embedding_dataset(embeddings_data, 'pytorch')
    
    # Create numpy dataset
    numpy_dataset = converter.create_embedding_dataset(embeddings_data, 'numpy')
    
    print("\n" + "=" * 50)
    print("✅ EMBEDDING CONVERSION COMPLETED!")
    print("=" * 50)
    
    print(f"📁 Generated Files:")
    print(f"   - miyawaki4_embeddings.pkl (complete data)")
    print(f"   - miyawaki4_embeddings.npz (numpy arrays)")
    print(f"   - miyawaki4_embeddings_metadata.json (metadata)")
    print(f"   - miyawaki4_embedding_analysis.png (analysis plot)")
    
    print(f"\n🎯 Ready for Downstream Tasks:")
    print(f"   - Cross-modal retrieval")
    print(f"   - Image generation from fMRI")
    print(f"   - Embedding space analysis")
    print(f"   - Transfer learning")
    
    return embeddings_data, pytorch_dataset, numpy_dataset

if __name__ == "__main__":
    main()
