#!/usr/bin/env python3
"""
Simple LDM Demo - Test Integration without Heavy Model Downloads
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from pathlib import Path

def load_miyawaki4_embeddings():
    """Load miyawaki4 embeddings for testing"""
    print("📥 Loading Miyawaki4 Embeddings")
    print("=" * 40)
    
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        return None
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    print(f"✅ Embeddings loaded successfully")
    print(f"   📊 Training samples: {len(embeddings_data['train']['fmri_embeddings'])}")
    print(f"   📊 Test samples: {len(embeddings_data['test']['fmri_embeddings'])}")
    
    return embeddings_data

class SimplefMRIToImageMapper(nn.Module):
    """Simple mapping network from fMRI embeddings to image space"""

    def __init__(self, fmri_dim=512, image_size=128):
        super().__init__()

        self.image_size = image_size
        image_dim = image_size * image_size * 3  # 128x128x3 = 49,152

        self.mapper = nn.Sequential(
            nn.Linear(fmri_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, image_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, fmri_embedding):
        # Map fMRI to image space
        image_flat = self.mapper(fmri_embedding)
        # Reshape to image format
        image = image_flat.view(-1, 3, self.image_size, self.image_size)
        return image

class SimpleLDMDemo:
    """Simple demonstration of fMRI-to-image mapping"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mapper = SimplefMRIToImageMapper().to(device)
        
        print(f"🔧 Simple LDM Demo initialized on {device}")
        print(f"📊 Mapper parameters: {sum(p.numel() for p in self.mapper.parameters()):,}")
    
    def generate_dummy_training_data(self, embeddings_data):
        """Generate dummy training data for demonstration"""
        print("\n🎯 Generating Dummy Training Data")
        print("=" * 40)
        
        # Get fMRI embeddings
        train_fmri = torch.FloatTensor(embeddings_data['train']['fmri_embeddings']).to(self.device)
        
        # Create dummy target images (random patterns based on fMRI)
        dummy_images = []
        for i, fmri in enumerate(train_fmri):
            # Create deterministic pattern based on fMRI
            np.random.seed(int(torch.sum(fmri).item() * 1000) % 2**32)
            
            # Generate colorful pattern (smaller size)
            pattern = np.random.rand(128, 128, 3)

            # Add some structure based on fMRI values
            for j in range(0, 128, 16):
                for k in range(0, 128, 16):
                    if j//16 < 8 and k//16 < 8:
                        idx = (j//16) * 8 + (k//16)
                        if idx < len(fmri):
                            intensity = float(fmri[idx % len(fmri)])
                            pattern[j:j+16, k:k+16] *= (0.5 + intensity)
            
            pattern = np.clip(pattern, 0, 1)
            dummy_images.append(pattern)
        
        dummy_images = torch.FloatTensor(np.array(dummy_images)).permute(0, 3, 1, 2).to(self.device)
        
        print(f"✅ Generated {len(dummy_images)} dummy training images")
        print(f"   📊 fMRI shape: {train_fmri.shape}")
        print(f"   📊 Images shape: {dummy_images.shape}")
        
        return train_fmri, dummy_images
    
    def train_simple_mapper(self, train_fmri, train_images, epochs=50):
        """Train the simple mapper"""
        print(f"\n🏋️ Training Simple Mapper for {epochs} epochs")
        print("=" * 50)
        
        optimizer = torch.optim.Adam(self.mapper.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            batch_size = 16
            for i in range(0, len(train_fmri), batch_size):
                batch_fmri = train_fmri[i:i+batch_size]
                batch_images = train_images[i:i+batch_size]
                
                # Forward pass
                predicted_images = self.mapper(batch_fmri)
                loss = criterion(predicted_images, batch_images)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_fmri) // batch_size + 1)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print(f"✅ Training completed!")
        print(f"   📊 Final loss: {losses[-1]:.6f}")
        
        return losses
    
    def generate_from_fmri(self, fmri_embedding):
        """Generate image from fMRI embedding"""
        self.mapper.eval()
        
        with torch.no_grad():
            if len(fmri_embedding.shape) == 1:
                fmri_embedding = fmri_embedding.unsqueeze(0)
            
            generated_image = self.mapper(fmri_embedding)
            
            # Convert to PIL Image
            image_np = generated_image[0].permute(1, 2, 0).cpu().numpy()
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image_np)
    
    def test_generation(self, embeddings_data):
        """Test image generation"""
        print("\n🎨 Testing Image Generation")
        print("=" * 40)
        
        test_fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings']).to(self.device)
        
        generated_images = []
        
        for i in range(min(6, len(test_fmri))):
            print(f"   Generating image {i+1}...")
            generated_image = self.generate_from_fmri(test_fmri[i])
            generated_images.append(generated_image)
        
        print(f"✅ Generated {len(generated_images)} test images")
        
        return generated_images
    
    def visualize_results(self, embeddings_data, generated_images, training_losses):
        """Visualize results"""
        print("\n📊 Creating Visualization")
        print("=" * 30)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Training loss plot
        plt.subplot(2, 4, 1)
        plt.plot(training_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
        
        # Original vs Generated images
        original_images = embeddings_data['test']['original_images']

        for i in range(min(3, len(generated_images))):
            # Original image
            plt.subplot(2, 4, i+2)
            if i < len(original_images):
                original = original_images[i].transpose(1, 2, 0)  # CHW -> HWC
                plt.imshow(original)
                plt.title(f'Original {i+1}')
                plt.axis('off')

        for i in range(min(3, len(generated_images))):
            # Generated image
            plt.subplot(2, 4, i+6)
            plt.imshow(generated_images[i])
            plt.title(f'Generated {i+1}')
            plt.axis('off')
        
        # Performance metrics
        plt.subplot(2, 4, 5)
        final_loss = training_losses[-1]
        improvement = (training_losses[0] - final_loss) / training_losses[0] * 100

        plt.text(0.1, 0.8, 'Performance Metrics', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Final Loss: {final_loss:.6f}', fontsize=10)
        plt.text(0.1, 0.6, f'Improvement: {improvement:.1f}%', fontsize=10)
        plt.text(0.1, 0.5, f'Epochs: {len(training_losses)}', fontsize=10)
        plt.text(0.1, 0.4, f'Device: {self.device}', fontsize=10)
        plt.text(0.1, 0.3, f'Parameters: {sum(p.numel() for p in self.mapper.parameters()):,}', fontsize=10)
        plt.text(0.1, 0.2, f'Status: ✅ Completed', fontsize=10, color='green')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('simple_ldm_demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("💾 Results saved as 'simple_ldm_demo_results.png'")

def main():
    """Main demo function"""
    print("🎯 Simple LDM Demo with Miyawaki4 Embeddings")
    print("=" * 60)
    
    # Load embeddings
    embeddings_data = load_miyawaki4_embeddings()
    if embeddings_data is None:
        print("❌ Cannot proceed without embeddings")
        return
    
    # Initialize demo
    demo = SimpleLDMDemo()
    
    # Generate dummy training data
    train_fmri, train_images = demo.generate_dummy_training_data(embeddings_data)
    
    # Train mapper
    training_losses = demo.train_simple_mapper(train_fmri, train_images, epochs=50)
    
    # Test generation
    generated_images = demo.test_generation(embeddings_data)
    
    # Visualize results
    demo.visualize_results(embeddings_data, generated_images, training_losses)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SIMPLE LDM DEMO SUMMARY")
    print("=" * 60)
    
    print(f"🔧 Architecture: Simple Neural Network Mapper")
    print(f"📊 Parameters: {sum(p.numel() for p in demo.mapper.parameters()):,}")
    print(f"🎯 Task: fMRI (512D) → Image (512x512x3)")
    print(f"🏋️ Training: {len(training_losses)} epochs")
    print(f"📈 Final Loss: {training_losses[-1]:.6f}")
    print(f"🎨 Generated: {len(generated_images)} test images")
    
    improvement = (training_losses[0] - training_losses[-1]) / training_losses[0] * 100
    print(f"📊 Improvement: {improvement:.1f}%")
    
    print(f"\n💡 This is a simplified demonstration of:")
    print(f"   - fMRI embedding to image mapping")
    print(f"   - Neural network training pipeline")
    print(f"   - Image generation from brain signals")
    print(f"   - Integration with miyawaki4 embeddings")
    
    print(f"\n🚀 For production use:")
    print(f"   - Use actual Stable Diffusion models")
    print(f"   - Implement proper conditioning mechanisms")
    print(f"   - Add evaluation metrics (CLIP similarity, FID)")
    print(f"   - Scale to larger datasets")

if __name__ == "__main__":
    main()
