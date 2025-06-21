import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ImprovedCLIPGuidedDiffusionModel(nn.Module):
    def __init__(self, unet, num_timesteps=1000, clip_guidance_weight=0.05, clip_model_name="ViT-B/32"):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.base_clip_guidance_weight = clip_guidance_weight  # Much smaller default
        self.current_clip_weight = 0.0  # Start with no CLIP guidance
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_model.eval()
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Noise schedule
        self.register_noise_schedule()
        
        # Training phase tracking
        self.training_phase = "diffusion_only"  # "diffusion_only" or "clip_guided"
        self.warmup_epochs = 100  # Pure diffusion training epochs
        
    def register_noise_schedule(self):
        betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def update_training_phase(self, epoch):
        """Update training phase and CLIP weight based on curriculum learning"""
        if epoch < self.warmup_epochs:
            # Phase 1: Pure diffusion training
            self.training_phase = "diffusion_only"
            self.current_clip_weight = 0.0
        else:
            # Phase 2: Gradual CLIP introduction
            self.training_phase = "clip_guided"
            # Gradually increase CLIP weight from 0 to base_clip_guidance_weight
            progress = min(1.0, (epoch - self.warmup_epochs) / 50)  # 50 epochs to reach full weight
            self.current_clip_weight = progress * self.base_clip_guidance_weight
    
    def prepare_image_for_clip(self, images):
        """Convert grayscale 28x28 images to RGB 224x224 for CLIP"""
        # images: [batch, 1, 28, 28] in range [-1, 1]
        
        # Convert to [0, 1] range
        images = (images + 1.0) / 2.0
        
        # Convert grayscale to RGB
        images = images.repeat(1, 3, 1, 1)  # [batch, 3, 28, 28]
        
        # Resize to 224x224 using bilinear interpolation
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize for CLIP (ImageNet normalization)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        return images
    
    def get_clip_text_features(self, labels):
        """Get CLIP text features for digit labels"""
        # Convert labels to text descriptions
        text_descriptions = [f"a handwritten digit {label.item()}" for label in labels]
        
        # Tokenize text
        text_tokens = clip.tokenize(text_descriptions).to(next(self.clip_model.parameters()).device)
        
        # Get text features
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def compute_clip_loss(self, images, labels):
        """Compute CLIP loss between generated images and text descriptions"""
        if self.current_clip_weight == 0.0:
            return torch.tensor(0.0, device=images.device)
        
        # Prepare images for CLIP
        clip_images = self.prepare_image_for_clip(images)
        
        # Get CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_images)
            image_features = F.normalize(image_features, dim=-1)
        
        text_features = self.get_clip_text_features(labels)
        
        # Compute cosine similarity
        similarity = torch.sum(image_features * text_features, dim=-1)
        
        # CLIP loss: maximize similarity (minimize negative similarity)
        clip_loss = -similarity.mean()
        
        return clip_loss
    
    def forward(self, x_t, t, condition, labels=None):
        """Forward pass with optional CLIP guidance"""
        # Standard diffusion forward pass
        noise_pred = self.unet(x_t, t, condition)
        
        return noise_pred
    
    def training_step(self, x_0, condition, labels, epoch):
        """Complete training step with curriculum learning"""
        # Update training phase
        self.update_training_phase(epoch)
        
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to images
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.forward(x_t, t, condition, labels)
        
        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # CLIP loss (only if in CLIP guided phase)
        if self.training_phase == "clip_guided" and labels is not None:
            # Reconstruct x_0 from noise prediction
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            
            # Clamp to valid range
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            # Compute CLIP loss
            clip_loss = self.compute_clip_loss(x_0_pred, labels)
        else:
            clip_loss = torch.tensor(0.0, device=device)
        
        # Total loss with current CLIP weight
        total_loss = diffusion_loss + self.current_clip_weight * clip_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'clip_loss': clip_loss,
            'clip_weight': self.current_clip_weight,
            'training_phase': self.training_phase
        }

    def sample(self, condition, num_samples=1, num_timesteps=None):
        """Sample images from the model"""
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        
        device = next(self.parameters()).device
        
        # Start from random noise
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(num_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(x, t, condition)
            
            # Compute x_{t-1}
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            alpha_cumprod_prev = self.alphas_cumprod_prev[i] if i > 0 else torch.tensor(1.0)
            beta_t = self.betas[i]
            
            # Compute mean
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred)
            
            # Compute variance
            if i > 0:
                variance = self.posterior_variance[i]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        
        return torch.clamp(x, -1.0, 1.0)

def test_improved_clip_guided_model():
    """Test the improved CLIP guided model"""
    print("Testing Improved CLIP Guided Diffusion Model...")
    
    # Import UNet
    from improved_unet import ImprovedUNet
    
    # Create UNet
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=32,  # Smaller for stability
        num_res_blocks=1    # Smaller for stability
    )
    
    # Create improved CLIP guided model
    model = ImprovedCLIPGuidedDiffusionModel(
        unet=unet,
        num_timesteps=1000,
        clip_guidance_weight=0.05  # Much smaller weight
    )
    
    # Test data
    batch_size = 2
    x_0 = torch.randn(batch_size, 1, 28, 28)
    condition = torch.randn(batch_size, 512)
    labels = torch.tensor([3, 7])
    
    print(f"Input shape: {x_0.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Labels: {labels}")
    
    # Test training step (diffusion only phase)
    print("\n=== Testing Diffusion Only Phase (Epoch 50) ===")
    result = model.training_step(x_0, condition, labels, epoch=50)
    print(f"Training phase: {result['training_phase']}")
    print(f"CLIP weight: {result['clip_weight']:.4f}")
    print(f"Total loss: {result['total_loss']:.4f}")
    print(f"Diffusion loss: {result['diffusion_loss']:.4f}")
    print(f"CLIP loss: {result['clip_loss']:.4f}")
    
    # Test training step (CLIP guided phase)
    print("\n=== Testing CLIP Guided Phase (Epoch 120) ===")
    result = model.training_step(x_0, condition, labels, epoch=120)
    print(f"Training phase: {result['training_phase']}")
    print(f"CLIP weight: {result['clip_weight']:.4f}")
    print(f"Total loss: {result['total_loss']:.4f}")
    print(f"Diffusion loss: {result['diffusion_loss']:.4f}")
    print(f"CLIP loss: {result['clip_loss']:.4f}")
    
    # Test sampling
    print("\n=== Testing Sampling ===")
    with torch.no_grad():
        samples = model.sample(condition, num_samples=2, num_timesteps=100)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print("\n✅ Improved CLIP Guided Model test completed successfully!")

if __name__ == "__main__":
    test_improved_clip_guided_model()
