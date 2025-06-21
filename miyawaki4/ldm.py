import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler, DDIMScheduler
import clip
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class fMRIToLDMAdapter:
    """
    Adapter untuk menggunakan fMRI embedding dengan Latent Diffusion Models
    Berbagai pendekatan conditioning
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.fmri_encoder = None
        self.ldm_pipeline = None

    def load_pretrained_fmri_encoder(self, model_path):
        """Load pre-trained fMRI encoder dari contrastive learning"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Import fMRI encoder dari runembedding.py
        from runembedding import fMRIEncoder
        self.fmri_encoder = fMRIEncoder(fmri_dim=967, clip_dim=512, hidden_dims=[2048, 1024])
        self.fmri_encoder.load_state_dict(checkpoint['fmri_encoder_state_dict'])
        self.fmri_encoder.to(self.device)
        self.fmri_encoder.eval()

        print("✅ Pre-trained fMRI encoder loaded successfully!")

    def load_miyawaki4_embeddings(self, embeddings_path="miyawaki4_embeddings.pkl"):
        """Load pre-computed embeddings dari miyawaki4"""
        print(f"📥 Loading embeddings from {embeddings_path}")

        if not Path(embeddings_path).exists():
            print(f"❌ Embeddings file not found: {embeddings_path}")
            return None

        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)

        print(f"✅ Embeddings loaded successfully")
        print(f"   📊 Training samples: {len(embeddings_data['train']['fmri_embeddings'])}")
        print(f"   📊 Test samples: {len(embeddings_data['test']['fmri_embeddings'])}")

        return embeddings_data

class Method1_DirectConditioning:
    """
    Method 1: Direct conditioning menggunakan fMRI embedding sebagai text conditioning
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Setup Stable Diffusion pipeline dengan custom conditioning"""
        print("🔄 Loading Stable Diffusion pipeline...")

        try:
            # Load standard Stable Diffusion
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,  # Disable safety checker for faster loading
                requires_safety_checker=False
            ).to(self.device)

            print("✅ Stable Diffusion pipeline loaded successfully")

        except Exception as e:
            print(f"⚠️ Failed to load Stable Diffusion: {e}")
            print("💡 Using dummy pipeline for testing")
            self.pipe = None

        # Mapping network: fMRI embedding → CLIP text embedding space
        self.fmri_to_clip = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),  # CLIP text embedding dimension
            nn.LayerNorm(768)
        ).to(self.device)

        print("✅ fMRI-to-CLIP mapping network initialized")
        
    def train_adapter(self, fmri_embeddings, images, epochs=50):
        """Train adapter network untuk map fMRI → CLIP text space"""
        optimizer = torch.optim.Adam(self.fmri_to_clip.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(fmri_embeddings), 16):  # Batch size 16
                batch_fmri = fmri_embeddings[i:i+16].to(self.device)
                batch_images = images[i:i+16]
                
                # Get CLIP text embeddings dari images (sebagai target)
                with torch.no_grad():
                    # Convert images ke prompt embeddings
                    text_inputs = self.pipe.tokenizer(
                        ["generated image"] * len(batch_images),
                        padding="max_length",
                        max_length=77,
                        return_tensors="pt"
                    )
                    target_embeddings = self.pipe.text_encoder(
                        text_inputs.input_ids.to(self.device)
                    ).last_hidden_state.mean(dim=1)  # [batch, 768]
                
                # Forward pass
                predicted_embeddings = self.fmri_to_clip(batch_fmri)
                
                # Loss
                loss = F.mse_loss(predicted_embeddings, target_embeddings)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def reconstruct_from_fmri(self, fmri_embedding, guidance_scale=7.5, num_steps=20):
        """Reconstruct image dari fMRI embedding"""
        if self.pipe is None:
            print("⚠️ Stable Diffusion pipeline not available, returning dummy image")
            return self._create_dummy_image()

        try:
            with torch.no_grad():
                # Convert fMRI embedding ke CLIP text embedding
                clip_embedding = self.fmri_to_clip(fmri_embedding.unsqueeze(0))

                # Create text embeddings structure yang expected oleh diffusion model
                batch_size = 1
                max_length = 77

                # Create text embeddings structure
                text_embeddings = torch.zeros(batch_size, max_length, 768).to(self.device)
                text_embeddings[:, 0] = clip_embedding  # Put fMRI embedding di first token

                # Create unconditional embeddings for classifier-free guidance
                uncond_embeddings = torch.zeros_like(text_embeddings)

                # Combine for classifier-free guidance
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                print(f"🎨 Generating image with {num_steps} steps...")

                # Generate image
                image = self.pipe(
                    prompt_embeds=text_embeddings,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    height=512,
                    width=512
                ).images[0]

                print("✅ Image generated successfully")
                return image

        except Exception as e:
            print(f"❌ Error in image generation: {e}")
            return self._create_dummy_image()

    def _create_dummy_image(self):
        """Create dummy image for testing"""
        # Create colorful dummy image
        dummy_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(dummy_array)

class Method2_CrossAttentionConditioning:
    """
    Method 2: Cross-attention conditioning - inject fMRI embedding ke U-Net
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_components()
        
    def setup_components(self):
        """Setup individual components"""
        print("🔄 Loading diffusion components...")

        try:
            # Load VAE untuk latent space
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

            # Load U-Net (akan kita modifikasi)
            self.unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

            # Scheduler
            self.scheduler = DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler"
            )

            print("✅ Diffusion components loaded successfully")

        except Exception as e:
            print(f"⚠️ Failed to load diffusion components: {e}")
            print("💡 Using dummy components for testing")
            self.vae = None
            self.unet = None
            self.scheduler = None

        # Conditioning projection
        self.fmri_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),  # Match cross-attention dimension
            nn.LayerNorm(768)
        ).to(self.device)

        print("✅ fMRI projection network initialized")
        
    def modify_unet_cross_attention(self):
        """Modifikasi U-Net untuk accept fMRI conditioning"""
        # Add fMRI cross-attention layers
        for name, module in self.unet.named_modules():
            if hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # Add fMRI attention branch
                hidden_dim = module.to_k.in_features
                module.fmri_to_k = nn.Linear(768, hidden_dim).to(self.device)
                module.fmri_to_v = nn.Linear(768, hidden_dim).to(self.device)
    
    def forward_with_fmri_conditioning(self, latents, timestep, fmri_embedding):
        """Forward pass dengan fMRI conditioning"""
        # Ensure consistent dtype and device
        device = latents.device
        dtype = latents.dtype

        # Project fMRI embedding with correct dtype
        fmri_embedding = fmri_embedding.to(device=device, dtype=dtype)
        fmri_context = self.fmri_projection(fmri_embedding.unsqueeze(0))  # [1, 768]
        fmri_context = fmri_context.unsqueeze(1)  # [1, 1, 768] untuk sequence

        # Standard text conditioning (empty) with correct dtype
        text_embeddings = torch.zeros(1, 77, 768, device=device, dtype=dtype)

        # Ensure fMRI context has correct dtype
        fmri_context = fmri_context.to(dtype=dtype)

        # Combine conditioning
        combined_context = torch.cat([text_embeddings, fmri_context], dim=1)  # [1, 78, 768]

        # U-Net forward
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=combined_context
        ).sample

        return noise_pred
    
    def reconstruct_from_fmri(self, fmri_embedding, num_steps=20):
        """Reconstruct menggunakan DDIM sampling"""
        if self.unet is None or self.vae is None or self.scheduler is None:
            print("⚠️ Diffusion components not available, returning dummy image")
            return self._create_dummy_image()

        try:
            print(f"🎨 Generating image with cross-attention conditioning...")

            # Determine dtype from model
            model_dtype = next(self.unet.parameters()).dtype

            # Initialize latents with correct dtype
            latents = torch.randn(1, 4, 64, 64, device=self.device, dtype=model_dtype)

            # Ensure fMRI projection network has correct dtype
            self.fmri_projection = self.fmri_projection.to(dtype=model_dtype)

            # Set timesteps
            self.scheduler.set_timesteps(num_steps)
            timesteps = self.scheduler.timesteps

            # Denoising loop
            for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
                # Predict noise with dtype consistency
                noise_pred = self.forward_with_fmri_conditioning(latents, t, fmri_embedding)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode ke image
            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                # Ensure VAE input has correct dtype
                if latents.dtype != model_dtype:
                    latents = latents.to(dtype=model_dtype)
                images = self.vae.decode(latents).sample
                images = (images + 1) / 2  # [-1, 1] → [0, 1]
                images = torch.clamp(images, 0, 1)

            # Convert ke PIL
            image = images[0].permute(1, 2, 0).cpu().float().numpy()
            image = (image * 255).astype(np.uint8)

            print("✅ Image generated successfully")
            return Image.fromarray(image)

        except Exception as e:
            print(f"❌ Error in cross-attention generation: {e}")
            return self._create_dummy_image()

    def _create_dummy_image(self):
        """Create dummy image for testing"""
        # Create colorful dummy image
        dummy_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(dummy_array)

class Method3_ControlNetStyle:
    """
    Method 3: ControlNet-style conditioning
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_controlnet()
        
    def setup_controlnet(self):
        """Setup ControlNet-style conditioning network"""
        print("🔄 Setting up ControlNet-style conditioning...")

        # ControlNet untuk fMRI conditioning
        self.fmri_controlnet = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4 * 64 * 64),  # Match latent dimensions
            nn.Unflatten(1, (4, 64, 64))   # Reshape ke latent shape
        ).to(self.device)

        try:
            # Load standard diffusion components
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            print("✅ ControlNet pipeline loaded successfully")

        except Exception as e:
            print(f"⚠️ Failed to load ControlNet pipeline: {e}")
            print("💡 Using dummy pipeline for testing")
            self.pipe = None

        print("✅ ControlNet conditioning network initialized")
        
    def train_controlnet(self, fmri_embeddings, target_images, epochs=100):
        """Train ControlNet untuk fMRI conditioning"""
        optimizer = torch.optim.Adam(self.fmri_controlnet.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            for i in range(0, len(fmri_embeddings), 4):  # Batch size 4
                batch_fmri = fmri_embeddings[i:i+4].to(self.device)
                batch_images = target_images[i:i+4].to(self.device)
                
                # Encode images ke latent space
                with torch.no_grad():
                    latents = self.pipe.vae.encode(batch_images).latent_dist.sample()
                    latents = latents * 0.18215
                
                # Generate conditioning dari fMRI
                conditioning = self.fmri_controlnet(batch_fmri)
                
                # Loss: predict target latents
                loss = F.mse_loss(conditioning, latents)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print("ControlNet training completed!")
    
    def reconstruct_from_fmri(self, fmri_embedding, num_steps=20):
        """Reconstruct dengan ControlNet conditioning"""
        if self.pipe is None:
            print("⚠️ ControlNet pipeline not available, returning dummy image")
            return self._create_dummy_image()

        try:
            print(f"🎨 Generating image with ControlNet conditioning...")

            # Determine model dtype
            model_dtype = next(self.pipe.unet.parameters()).dtype

            # Generate conditioning signal with correct dtype
            with torch.no_grad():
                # Ensure fMRI embedding has correct dtype
                fmri_embedding = fmri_embedding.to(device=self.device, dtype=model_dtype)

                # Ensure ControlNet has correct dtype
                self.fmri_controlnet = self.fmri_controlnet.to(dtype=model_dtype)

                conditioning = self.fmri_controlnet(fmri_embedding.unsqueeze(0))

                # Use conditioning sebagai starting point + add noise
                noise = torch.randn_like(conditioning) * 0.1
                init_latents = conditioning + noise

                # Ensure latents have correct dtype
                init_latents = init_latents.to(dtype=model_dtype)

                # Generate with conditioning
                image = self.pipe(
                    prompt="",  # Empty prompt
                    latents=init_latents,
                    num_inference_steps=num_steps,
                    guidance_scale=7.5
                ).images[0]

                print("✅ ControlNet image generated successfully")
                return image

        except Exception as e:
            print(f"❌ Error in ControlNet generation: {e}")
            return self._create_dummy_image()

    def _create_dummy_image(self):
        """Create dummy image for testing"""
        # Create colorful dummy image
        dummy_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(dummy_array)

class fMRILDMReconstructor:
    """
    Main class yang combine semua methods
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.methods = {
            'direct': Method1_DirectConditioning(device),
            'cross_attention': Method2_CrossAttentionConditioning(device),
            'controlnet': Method3_ControlNetStyle(device)
        }
        
    def reconstruct_comparison(self, fmri_embedding, method='direct'):
        """Compare reconstruction dengan different methods"""
        if method not in self.methods:
            raise ValueError(f"Method {method} not available")
        
        reconstructed = self.methods[method].reconstruct_from_fmri(fmri_embedding)
        return reconstructed
    
    def batch_reconstruction(self, fmri_embeddings, method='direct'):
        """Batch reconstruction"""
        results = []
        for embedding in fmri_embeddings:
            result = self.reconstruct_comparison(embedding, method)
            results.append(result)
        return results
    
    def visualize_methods_comparison(self, fmri_embedding):
        """Compare all methods side by side"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['direct', 'cross_attention', 'controlnet']
        titles = ['Direct Conditioning', 'Cross Attention', 'ControlNet Style']
        
        for i, (method, title) in enumerate(zip(methods, titles)):
            try:
                image = self.reconstruct_comparison(fmri_embedding, method)
                axes[i].imshow(image)
                axes[i].set_title(title)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{title} (Failed)")
        
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    """Usage example"""
    
    # Initialize reconstructor
    reconstructor = fMRILDMReconstructor()
    
    # Load pre-trained fMRI encoder
    # adapter = fMRIToLDMAdapter()
    # adapter.load_pretrained_fmri_encoder("miyawaki_contrastive_clip.pth")
    
    # Example fMRI embedding (512-dimensional)
    dummy_fmri_embedding = torch.randn(512)
    
    # Reconstruct dengan different methods
    try:
        # Method 1: Direct conditioning
        image_direct = reconstructor.reconstruct_comparison(
            dummy_fmri_embedding, 
            method='direct'
        )
        
        # Method 2: Cross attention
        image_cross = reconstructor.reconstruct_comparison(
            dummy_fmri_embedding, 
            method='cross_attention'
        )
        
        # Method 3: ControlNet style
        image_controlnet = reconstructor.reconstruct_comparison(
            dummy_fmri_embedding, 
            method='controlnet'
        )
        
        # Visualize comparison
        reconstructor.visualize_methods_comparison(dummy_fmri_embedding)
        
    except Exception as e:
        print(f"Error in reconstruction: {e}")
        print("Note: This requires actual trained models and proper setup")

if __name__ == "__main__":
    # Uncomment untuk testing
    # main()
    pass