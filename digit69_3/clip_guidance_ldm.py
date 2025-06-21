#!/usr/bin/env python3
"""
CLIP Guidance Enhanced LDM for Digit69_3
Menggunakan CLIP model untuk guidance dalam generasi digit dari fMRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import matplotlib.pyplot as plt
from enhanced_training_pipeline import EnhancedDiffusionModel
from improved_unet import ImprovedUNet

class CLIPGuidanceLoss(nn.Module):
    """CLIP Guidance Loss untuk mengarahkan generasi"""
    
    def __init__(self, clip_model_name="ViT-B/32", device='cuda'):
        super().__init__()
        self.device = device
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Digit class descriptions
        self.digit_descriptions = [
            "a handwritten digit zero",
            "a handwritten digit one", 
            "a handwritten digit two",
            "a handwritten digit three",
            "a handwritten digit four",
            "a handwritten digit five",
            "a handwritten digit six",
            "a handwritten digit seven",
            "a handwritten digit eight",
            "a handwritten digit nine"
        ]
        
        # Pre-compute text embeddings
        self.text_embeddings = self._compute_text_embeddings()
        
        print(f"📊 CLIP Guidance Loss initialized:")
        print(f"   Model: {clip_model_name}")
        print(f"   Device: {device}")
        print(f"   Text embeddings: {self.text_embeddings.shape}")
    
    def _compute_text_embeddings(self):
        """Pre-compute text embeddings untuk semua digit classes"""
        text_tokens = clip.tokenize(self.digit_descriptions).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        return text_embeddings
    
    def _preprocess_image(self, image):
        """Preprocess image untuk CLIP (convert ke RGB, resize, normalize)"""
        # image shape: (batch, 1, 28, 28) range [-1, 1]
        
        # Convert to [0, 1]
        image = (image + 1) / 2
        
        # Convert grayscale to RGB
        image_rgb = image.repeat(1, 3, 1, 1)  # (batch, 3, 28, 28)
        
        # Resize to 224x224 (CLIP input size)
        image_resized = F.interpolate(image_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
        
        image_normalized = (image_resized - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        return image_normalized
    
    def forward(self, generated_images, target_classes=None, target_text=None):
        """
        Compute CLIP guidance loss
        
        Args:
            generated_images: (batch, 1, 28, 28) generated digit images
            target_classes: (batch,) digit class indices [0-9]
            target_text: list of text descriptions (optional)
        """
        batch_size = generated_images.shape[0]
        
        # Preprocess images for CLIP
        clip_images = self._preprocess_image(generated_images)
        
        # Get image embeddings
        image_embeddings = self.clip_model.encode_image(clip_images)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        if target_classes is not None:
            # Use pre-computed text embeddings for digit classes
            target_classes_cpu = target_classes.cpu()  # Move to CPU for indexing
            target_embeddings = self.text_embeddings[target_classes_cpu].to(generated_images.device)
        elif target_text is not None:
            # Use custom text descriptions
            text_tokens = clip.tokenize(target_text).to(self.device)
            target_embeddings = self.clip_model.encode_text(text_tokens)
            target_embeddings = F.normalize(target_embeddings, dim=-1)
        else:
            raise ValueError("Either target_classes or target_text must be provided")
        
        # Compute cosine similarity (CLIP score)
        clip_scores = torch.sum(image_embeddings * target_embeddings, dim=-1)
        
        # Convert to loss (maximize similarity = minimize negative similarity)
        clip_loss = -clip_scores.mean()
        
        return clip_loss, clip_scores

class CLIPGuidedDiffusionModel(EnhancedDiffusionModel):
    """Enhanced Diffusion Model dengan CLIP Guidance"""
    
    def __init__(self, unet, num_timesteps=1000, clip_guidance_weight=1.0,
                 clip_model_name="ViT-B/32", **kwargs):
        super().__init__(unet, num_timesteps, **kwargs)

        self.clip_guidance_weight = clip_guidance_weight
        device = next(unet.parameters()).device
        self.clip_loss = CLIPGuidanceLoss(clip_model_name, device=device)

        # Move noise schedule tensors to device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, attr_value.to(device))

        print(f"📊 CLIP Guided Diffusion Model:")
        print(f"   CLIP guidance weight: {clip_guidance_weight}")
        print(f"   Base timesteps: {num_timesteps}")
        print(f"   Device: {device}")
    
    def p_losses_with_clip(self, x_start, t, condition, target_classes=None, 
                          noise=None, loss_type="huber"):
        """Training losses dengan CLIP guidance"""
        
        # Standard diffusion loss
        diffusion_loss = self.p_losses(x_start, t, condition, noise, loss_type)
        
        # CLIP guidance loss (only on clean images)
        if target_classes is not None:
            clip_loss, clip_scores = self.clip_loss(x_start, target_classes)
            
            # Combined loss
            total_loss = diffusion_loss + self.clip_guidance_weight * clip_loss
            
            return {
                'total_loss': total_loss,
                'diffusion_loss': diffusion_loss,
                'clip_loss': clip_loss,
                'clip_scores': clip_scores.mean()
            }
        else:
            return {
                'total_loss': diffusion_loss,
                'diffusion_loss': diffusion_loss,
                'clip_loss': torch.tensor(0.0),
                'clip_scores': torch.tensor(0.0)
            }
    
    @torch.no_grad()
    def clip_guided_sampling(self, condition, target_classes, image_size=28, 
                           guidance_scale=7.5, num_steps=50):
        """CLIP-guided sampling dengan classifier-free guidance"""
        
        batch_size = condition.shape[0]
        shape = (batch_size, 1, image_size, image_size)
        
        # Initialize noise
        img = torch.randn(shape, device=condition.device)
        
        # DDIM sampling schedule
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_steps, dtype=torch.long)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=condition.device, dtype=torch.long)
            
            # Unconditional prediction (null condition)
            null_condition = torch.zeros_like(condition)
            noise_pred_uncond = self.unet(img, t_batch, null_condition)
            
            # Conditional prediction
            noise_pred_cond = self.unet(img, t_batch, condition)
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # DDIM step
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # CLIP guidance on predicted x0
            if target_classes is not None and i % 5 == 0:  # Apply every 5 steps
                pred_x0_guided = pred_x0.clone().detach().requires_grad_(True)
                clip_loss, _ = self.clip_loss(pred_x0_guided, target_classes)

                # Gradient-based guidance
                if pred_x0_guided.grad is not None:
                    pred_x0_guided.grad.zero_()
                grad = torch.autograd.grad(clip_loss, pred_x0_guided, retain_graph=False)[0]
                pred_x0 = pred_x0_guided - 0.1 * grad  # Guidance step
                pred_x0 = pred_x0.detach()
            
            # Continue DDIM
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
        
        return img

class CLIPGuidedTrainer:
    """Trainer untuk CLIP Guided LDM"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = {
            'train_total': [], 'train_diffusion': [], 'train_clip': [],
            'test_total': [], 'test_diffusion': [], 'test_clip': [],
            'clip_scores': []
        }
        
        print(f"📊 CLIP Guided Trainer initialized")
        print(f"   Device: {device}")
        print(f"   CLIP guidance weight: {model.clip_guidance_weight}")
    
    def train_epoch(self, digit_classes=None):
        """Train one epoch dengan CLIP guidance"""
        self.model.train()
        epoch_losses = {'total': 0, 'diffusion': 0, 'clip': 0, 'clip_scores': 0}
        num_batches = 0
        
        for batch_idx, (fmri_emb, images) in enumerate(self.train_loader):
            fmri_emb = fmri_emb.to(self.device)
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
            
            # Get digit classes (if available)
            if digit_classes is not None:
                target_classes = digit_classes[batch_idx*batch_size:(batch_idx+1)*batch_size]
                target_classes = torch.tensor(target_classes).to(self.device)
            else:
                target_classes = None
            
            # Forward pass with CLIP guidance
            loss_dict = self.model.p_losses_with_clip(
                images, t, fmri_emb, target_classes
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['diffusion'] += loss_dict['diffusion_loss'].item()
            epoch_losses['clip'] += loss_dict['clip_loss'].item()
            epoch_losses['clip_scores'] += loss_dict['clip_scores'].item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def generate_samples(self, num_samples=4, target_classes=None):
        """Generate samples dengan CLIP guidance"""
        self.model.eval()
        
        with torch.no_grad():
            # Get test conditions
            test_batch = next(iter(self.test_loader))
            fmri_emb, target_images = test_batch
            fmri_emb = fmri_emb[:num_samples].to(self.device)
            
            if target_classes is None:
                target_classes = torch.arange(num_samples).to(self.device)
            
            # Generate with CLIP guidance
            generated_images = self.model.clip_guided_sampling(
                fmri_emb, target_classes, guidance_scale=7.5
            )
            
            return generated_images.cpu(), target_images[:num_samples]

def test_clip_guidance():
    """Test CLIP Guidance implementation"""
    print("🧪 TESTING CLIP GUIDANCE")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )
    
    model = CLIPGuidedDiffusionModel(
        unet, 
        num_timesteps=1000,
        clip_guidance_weight=1.0
    ).to(device)
    
    # Test inputs
    batch_size = 4
    fmri_emb = torch.randn(batch_size, 512).to(device)
    images = torch.randn(batch_size, 1, 28, 28).to(device)
    target_classes = torch.tensor([0, 1, 2, 3]).to(device)
    t = torch.randint(0, 1000, (batch_size,), device=device)  # Fix: specify device directly
    
    # Test CLIP loss directly
    print("🔍 Testing CLIP guidance loss...")
    clip_loss, clip_scores = model.clip_loss(images, target_classes)

    print(f"   CLIP loss: {clip_loss.item():.4f}")
    print(f"   CLIP scores: {clip_scores.mean().item():.4f}")
    print(f"   Individual scores: {clip_scores.tolist()}")
    
    # Test CLIP preprocessing
    print("🔍 Testing CLIP preprocessing...")
    with torch.no_grad():
        clip_images = model.clip_loss._preprocess_image(images)
        print(f"   Original images shape: {images.shape}")
        print(f"   CLIP processed shape: {clip_images.shape}")
        print(f"   CLIP processed range: [{clip_images.min():.3f}, {clip_images.max():.3f}]")

    # Test text embeddings
    print("🔍 Testing text embeddings...")
    print(f"   Text embeddings shape: {model.clip_loss.text_embeddings.shape}")
    print(f"   Sample descriptions: {model.clip_loss.digit_descriptions[:3]}")

    # Test different digit classes
    print("🔍 Testing different digit classes...")
    for digit in range(3):  # Test first 3 digits
        single_class = torch.tensor([digit]).to(device)
        single_image = images[0:1]  # Take first image

        clip_loss_single, clip_scores_single = model.clip_loss(single_image, single_class)
        print(f"   Digit {digit}: CLIP loss={clip_loss_single.item():.4f}, score={clip_scores_single.item():.4f}")
    
    print("✅ CLIP Guidance test completed!")

def main():
    """Main training function untuk CLIP Guided LDM"""
    print("🚀 CLIP GUIDED LDM TRAINING")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    # Load datasets
    from simple_baseline_model import Digit69BaselineDataset
    from torch.utils.data import DataLoader

    train_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "train", target_size=28)
    test_dataset = Digit69BaselineDataset("digit69_embeddings.pkl", "test", target_size=28)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch for CLIP
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Create CLIP guided model
    unet = ImprovedUNet(
        in_channels=1,
        out_channels=1,
        condition_dim=512,
        model_channels=64,
        num_res_blocks=2
    )

    model = CLIPGuidedDiffusionModel(
        unet,
        num_timesteps=1000,
        clip_guidance_weight=2.0  # Strong CLIP guidance
    )

    # Create trainer
    trainer = CLIPGuidedTrainer(model, train_loader, test_loader, device)

    # Training loop
    num_epochs = 100  # Shorter training dengan CLIP guidance

    print(f"🚀 Starting CLIP guided training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_losses = trainer.train_epoch()

        # Update scheduler
        trainer.scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1:3d}: "
              f"Total={train_losses['total']:.4f}, "
              f"Diff={train_losses['diffusion']:.4f}, "
              f"CLIP={train_losses['clip']:.4f}, "
              f"Score={train_losses['clip_scores']:.4f}")

        # Save losses
        trainer.losses['train_total'].append(train_losses['total'])
        trainer.losses['train_diffusion'].append(train_losses['diffusion'])
        trainer.losses['train_clip'].append(train_losses['clip'])
        trainer.losses['clip_scores'].append(train_losses['clip_scores'])

        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            generated, targets = trainer.generate_samples()

            # Save samples
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'CLIP Guided Samples - Epoch {epoch+1}', fontsize=16)

            for i in range(4):
                # Target
                axes[0, i].imshow(targets[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
                axes[0, i].set_title(f'Target {i}')
                axes[0, i].axis('off')

                # Generated
                axes[1, i].imshow(generated[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
                axes[1, i].set_title(f'CLIP Guided {i}')
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(f'clip_guided_samples_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Save best model
        if train_losses['total'] < trainer.best_loss:
            trainer.best_loss = train_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_loss': trainer.best_loss,
                'losses': trainer.losses
            }, 'clip_guided_ldm_best.pth')
            print(f"   💾 New best model saved (loss: {train_losses['total']:.4f})")

    print(f"\n✅ CLIP Guided training completed!")
    print(f"   Best loss: {trainer.best_loss:.4f}")

    # Plot training curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(trainer.losses['train_total'], label='Total Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Diffusion vs CLIP loss
    axes[0, 1].plot(trainer.losses['train_diffusion'], label='Diffusion Loss', color='blue')
    axes[0, 1].plot(trainer.losses['train_clip'], label='CLIP Loss', color='red')
    axes[0, 1].set_title('Diffusion vs CLIP Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # CLIP scores
    axes[1, 0].plot(trainer.losses['clip_scores'], label='CLIP Scores', color='green')
    axes[1, 0].set_title('CLIP Similarity Scores')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Loss ratios
    diffusion_losses = np.array(trainer.losses['train_diffusion'])
    clip_losses = np.array(trainer.losses['train_clip'])
    ratios = clip_losses / (diffusion_losses + 1e-8)

    axes[1, 1].plot(ratios, label='CLIP/Diffusion Ratio', color='purple')
    axes[1, 1].set_title('Loss Component Ratios')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('clip_guided_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_clip_guidance()
    else:
        main()
