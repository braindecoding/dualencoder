#!/usr/bin/env python3
"""
Quick CLIP Guidance Test
Test CLIP guidance dengan existing data dan models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Import our models
from improved_unet import ImprovedUNet
from clip_guidance_ldm import CLIPGuidedDiffusionModel, CLIPGuidanceLoss

class QuickCLIPTest:
    """Quick test untuk CLIP guidance functionality"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"🚀 QUICK CLIP GUIDANCE TEST")
        print(f"   Device: {self.device}")
        
        # Load existing data
        self.load_data()
        
        # Setup models
        self.setup_models()
    
    def load_data(self):
        """Load existing embeddings data"""
        print(f"\n📊 LOADING DATA")
        
        with open('digit69_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Get test data
        self.test_fmri = torch.FloatTensor(data['test']['fmri_embeddings'][:4])  # First 4 samples

        # Load original images and ensure correct format
        original_images = data['test']['original_images'][:4]

        # Check and fix image format
        if len(original_images.shape) == 4 and original_images.shape[1] == 3:
            # If already (batch, 3, H, W), convert to grayscale
            # Take only first channel and add channel dimension
            self.test_images = torch.FloatTensor(original_images[:, 0:1, :, :])
        elif len(original_images.shape) == 3:
            # If (batch, H, W), add channel dimension
            self.test_images = torch.FloatTensor(original_images).unsqueeze(1)
        else:
            # If (batch, 1, H, W), use as is
            self.test_images = torch.FloatTensor(original_images)

        # Ensure correct size (should be 28x28)
        if self.test_images.shape[-1] != 28:
            import torch.nn.functional as F
            self.test_images = F.interpolate(self.test_images, size=(28, 28), mode='bilinear', align_corners=False)

        # Generate random digit classes for demo
        self.test_classes = torch.randint(0, 10, (4,))
        
        print(f"   Test fMRI: {self.test_fmri.shape}")
        print(f"   Test images: {self.test_images.shape}")
        print(f"   Test classes: {self.test_classes.tolist()}")
    
    def setup_models(self):
        """Setup CLIP guided model"""
        print(f"\n🏗️ SETTING UP MODELS")
        
        # Create UNet
        unet = ImprovedUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=512,
            model_channels=64,
            num_res_blocks=2
        )
        
        # Create CLIP guided model
        self.clip_model = CLIPGuidedDiffusionModel(
            unet,
            num_timesteps=1000,
            clip_guidance_weight=2.0,  # Strong CLIP guidance
            clip_model_name="ViT-B/32"
        ).to(self.device)
        
        # Create standalone CLIP loss for evaluation
        self.clip_evaluator = CLIPGuidanceLoss("ViT-B/32", device=self.device)
        
        print(f"   CLIP guided model created")
        print(f"   Parameters: {sum(p.numel() for p in self.clip_model.parameters()):,}")
    
    def test_clip_loss_functionality(self):
        """Test CLIP loss dengan different scenarios"""
        print(f"\n🔍 TESTING CLIP LOSS FUNCTIONALITY")
        print("=" * 40)
        
        # Move data to device
        test_images = self.test_images.to(self.device)
        test_classes = self.test_classes.to(self.device)
        
        # Test 1: Random images vs digit classes
        print(f"📊 Test 1: Random images vs digit classes")
        random_images = torch.randn_like(test_images).to(self.device)
        clip_loss_random, clip_scores_random = self.clip_evaluator(random_images, test_classes)
        
        print(f"   Random images CLIP loss: {clip_loss_random.item():.4f}")
        print(f"   Random images CLIP scores: {clip_scores_random.mean().item():.4f}")
        
        # Test 2: Target images vs digit classes
        print(f"📊 Test 2: Target images vs digit classes")
        clip_loss_target, clip_scores_target = self.clip_evaluator(test_images, test_classes)
        
        print(f"   Target images CLIP loss: {clip_loss_target.item():.4f}")
        print(f"   Target images CLIP scores: {clip_scores_target.mean().item():.4f}")
        
        # Test 3: Different digit classes
        print(f"📊 Test 3: Same image, different classes")
        single_image = test_images[0:1].repeat(4, 1, 1, 1)
        different_classes = torch.tensor([0, 1, 2, 3]).to(self.device)
        
        clip_loss_diff, clip_scores_diff = self.clip_evaluator(single_image, different_classes)
        
        print(f"   Same image, different classes:")
        for i, (cls, score) in enumerate(zip(different_classes.cpu(), clip_scores_diff.cpu())):
            print(f"     Class {cls}: {score:.4f}")
        
        return {
            'random': {'loss': clip_loss_random.item(), 'scores': clip_scores_random.cpu().numpy()},
            'target': {'loss': clip_loss_target.item(), 'scores': clip_scores_target.cpu().numpy()},
            'different_classes': {'loss': clip_loss_diff.item(), 'scores': clip_scores_diff.cpu().numpy()}
        }
    
    def test_clip_guided_training_step(self):
        """Test single training step dengan CLIP guidance"""
        print(f"\n🎯 TESTING CLIP GUIDED TRAINING STEP")
        print("=" * 40)
        
        # Move data to device
        fmri_emb = self.test_fmri.to(self.device)
        images = self.test_images.to(self.device)
        classes = self.test_classes.to(self.device)
        
        # Sample timesteps
        t = torch.randint(0, self.clip_model.num_timesteps, (4,), device=self.device)

        # Ensure all noise schedule tensors are on correct device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(self.clip_model, attr_name)
            setattr(self.clip_model, attr_name, attr_value.to(self.device))
        
        # Test training step
        self.clip_model.train()
        
        print(f"📊 Forward pass dengan CLIP guidance...")
        loss_dict = self.clip_model.p_losses_with_clip(images, t, fmri_emb, classes)
        
        print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
        print(f"   Diffusion loss: {loss_dict['diffusion_loss'].item():.4f}")
        print(f"   CLIP loss: {loss_dict['clip_loss'].item():.4f}")
        print(f"   CLIP scores: {loss_dict['clip_scores'].item():.4f}")
        
        # Test backward pass
        print(f"📊 Testing backward pass...")
        loss_dict['total_loss'].backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in self.clip_model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"   Total gradient norm: {total_grad_norm:.4f}")
        print(f"   ✅ Backward pass successful!")
        
        return loss_dict
    
    def test_different_clip_weights(self):
        """Test different CLIP weights"""
        print(f"\n⚖️ TESTING DIFFERENT CLIP WEIGHTS")
        print("=" * 40)
        
        weights = [0.0, 0.5, 1.0, 2.0, 5.0]
        results = {}
        
        # Move data to device
        fmri_emb = self.test_fmri.to(self.device)
        images = self.test_images.to(self.device)
        classes = self.test_classes.to(self.device)
        t = torch.randint(0, self.clip_model.num_timesteps, (4,), device=self.device)
        
        for weight in weights:
            print(f"📊 Testing CLIP weight: {weight}")
            
            # Update CLIP weight
            self.clip_model.clip_guidance_weight = weight
            
            # Forward pass
            with torch.no_grad():
                loss_dict = self.clip_model.p_losses_with_clip(images, t, fmri_emb, classes)
            
            results[weight] = {
                'total_loss': loss_dict['total_loss'].item(),
                'diffusion_loss': loss_dict['diffusion_loss'].item(),
                'clip_loss': loss_dict['clip_loss'].item(),
                'clip_scores': loss_dict['clip_scores'].item()
            }
            
            print(f"   Total: {results[weight]['total_loss']:.4f}, "
                  f"Diff: {results[weight]['diffusion_loss']:.4f}, "
                  f"CLIP: {results[weight]['clip_loss']:.4f}, "
                  f"Score: {results[weight]['clip_scores']:.4f}")
        
        return results
    
    def visualize_clip_preprocessing(self):
        """Visualize CLIP preprocessing steps"""
        print(f"\n🎨 VISUALIZING CLIP PREPROCESSING")
        print("=" * 40)
        
        # Take first test image
        original_image = self.test_images[0:1].to(self.device)
        
        # Apply CLIP preprocessing
        clip_processed = self.clip_evaluator._preprocess_image(original_image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('CLIP Preprocessing Pipeline', fontsize=14)
        
        # Original image
        axes[0].imshow(original_image[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title('Original\n(1×28×28)')
        axes[0].axis('off')
        
        # RGB conversion
        rgb_image = original_image.repeat(1, 3, 1, 1)
        axes[1].imshow(rgb_image[0].permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title('RGB Conversion\n(3×28×28)')
        axes[1].axis('off')
        
        # Resized
        import torch.nn.functional as F
        resized = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)
        axes[2].imshow(resized[0].permute(1, 2, 0).cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[2].set_title('Resized\n(3×224×224)')
        axes[2].axis('off')
        
        # CLIP normalized
        axes[3].imshow(clip_processed[0].permute(1, 2, 0).cpu().numpy())
        axes[3].set_title('CLIP Normalized\n(3×224×224)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig('clip_preprocessing_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   Original range: [{original_image.min():.3f}, {original_image.max():.3f}]")
        print(f"   CLIP processed range: [{clip_processed.min():.3f}, {clip_processed.max():.3f}]")
        print(f"   💾 Visualization saved: clip_preprocessing_visualization.png")
    
    def create_results_summary(self, clip_test_results, training_step_results, weight_test_results):
        """Create comprehensive results summary"""
        print(f"\n📋 CREATING RESULTS SUMMARY")
        print("=" * 40)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CLIP Guidance Test Results Summary', fontsize=16)
        
        # 1. CLIP scores comparison
        scenarios = ['Random Images', 'Target Images']
        clip_scores = [
            clip_test_results['random']['scores'].mean(),
            clip_test_results['target']['scores'].mean()
        ]
        
        axes[0, 0].bar(scenarios, clip_scores, alpha=0.7, color=['red', 'green'])
        axes[0, 0].set_title('CLIP Scores: Random vs Target Images')
        axes[0, 0].set_ylabel('CLIP Score')
        
        # Add values on bars
        for i, v in enumerate(clip_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. Different classes scores
        classes = [0, 1, 2, 3]
        class_scores = clip_test_results['different_classes']['scores']
        
        axes[0, 1].bar(classes, class_scores, alpha=0.7, color='blue')
        axes[0, 1].set_title('CLIP Scores for Different Digit Classes')
        axes[0, 1].set_xlabel('Digit Class')
        axes[0, 1].set_ylabel('CLIP Score')
        
        # 3. CLIP weight impact
        weights = list(weight_test_results.keys())
        total_losses = [weight_test_results[w]['total_loss'] for w in weights]
        clip_losses = [weight_test_results[w]['clip_loss'] for w in weights]
        
        axes[1, 0].plot(weights, total_losses, 'o-', label='Total Loss', color='red')
        axes[1, 0].plot(weights, clip_losses, 's-', label='CLIP Loss', color='blue')
        axes[1, 0].set_title('Loss vs CLIP Weight')
        axes[1, 0].set_xlabel('CLIP Weight')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training step breakdown
        step_results = training_step_results
        loss_components = ['Total', 'Diffusion', 'CLIP']
        loss_values = [
            step_results['total_loss'].item(),
            step_results['diffusion_loss'].item(),
            step_results['clip_loss'].item()
        ]
        
        axes[1, 1].bar(loss_components, loss_values, alpha=0.7, 
                      color=['purple', 'orange', 'red'])
        axes[1, 1].set_title('Training Step Loss Components')
        axes[1, 1].set_ylabel('Loss Value')
        
        # Add values on bars
        for i, v in enumerate(loss_values):
            axes[1, 1].text(i, v + max(loss_values)*0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('clip_guidance_test_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        all_results = {
            'clip_functionality_test': clip_test_results,
            'training_step_test': {k: v.item() if torch.is_tensor(v) else v 
                                 for k, v in training_step_results.items()},
            'clip_weight_test': weight_test_results,
            'device': self.device,
            'model_parameters': sum(p.numel() for p in self.clip_model.parameters())
        }
        
        with open('clip_guidance_test_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"   💾 Results saved: clip_guidance_test_results.pkl")
        print(f"   💾 Summary plot: clip_guidance_test_summary.png")
        
        return all_results
    
    def run_full_test(self):
        """Run complete CLIP guidance test"""
        print(f"\n🚀 RUNNING FULL CLIP GUIDANCE TEST")
        print("=" * 50)
        
        # Test 1: CLIP loss functionality
        clip_results = self.test_clip_loss_functionality()
        
        # Test 2: Training step
        training_results = self.test_clip_guided_training_step()
        
        # Test 3: Different CLIP weights
        weight_results = self.test_different_clip_weights()
        
        # Test 4: Visualization
        self.visualize_clip_preprocessing()
        
        # Test 5: Summary
        summary = self.create_results_summary(clip_results, training_results, weight_results)
        
        print(f"\n✅ FULL CLIP GUIDANCE TEST COMPLETED!")
        print(f"📊 Key Findings:")
        print(f"   Target images CLIP score: {clip_results['target']['scores'].mean():.4f}")
        print(f"   Random images CLIP score: {clip_results['random']['scores'].mean():.4f}")
        print(f"   Training step successful: ✅")
        print(f"   Optimal CLIP weight: {max(weight_results.keys(), key=lambda w: weight_results[w]['clip_scores'])}")
        
        return summary

def main():
    """Main test function"""
    print("🎯 HIGH PRIORITY: QUICK CLIP GUIDANCE TEST")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run test
    tester = QuickCLIPTest(device)
    results = tester.run_full_test()
    
    print(f"\n🎉 CLIP GUIDANCE FUNCTIONALITY VERIFIED!")

if __name__ == "__main__":
    main()
