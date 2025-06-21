#!/usr/bin/env python3
"""
Modular Training with Fixed Diffusion Model
Compare original vs fixed diffusion performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Import components
from fmriencoder import fMRI_Encoder
from stimencoder import Shape_Encoder
from clipcorrtrain import CLIP_Correlation
from diffusion import Diffusion_Decoder as OriginalDiffusion
from diffusion_fixed import FixedDiffusion_Decoder, SimpleCNN_Decoder
from gan import GAN_Decoder
from miyawakidataset import load_miyawaki_dataset_corrected, create_dataloaders_corrected
from metriks import evaluate_decoding_performance

class FixedModularTrainer:
    """Modular trainer with fixed diffusion"""
    
    def __init__(self, device='cuda', use_fixed_diffusion=True):
        self.device = device
        self.use_fixed_diffusion = use_fixed_diffusion
        
        print(f"🏗️ Initializing {'Fixed' if use_fixed_diffusion else 'Original'} Modular Trainer")
        print("=" * 60)
        
        # Initialize components
        self.fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512).to(device)
        self.stim_encoder = Shape_Encoder(latent_dim=512).to(device)
        self.clip_correlation = CLIP_Correlation(latent_dim=512).to(device)
        
        # Choose diffusion implementation
        if use_fixed_diffusion:
            self.diffusion_decoder = FixedDiffusion_Decoder(correlation_dim=512).to(device)
            print("✅ Using Fixed Diffusion Decoder")
        else:
            self.diffusion_decoder = OriginalDiffusion(correlation_dim=512).to(device)
            print("✅ Using Original Diffusion Decoder")
        
        self.gan_decoder = GAN_Decoder(correlation_dim=512).to(device)
        
        # Print parameter counts
        self._print_parameter_counts()
    
    def _print_parameter_counts(self):
        """Print parameter counts for each component"""
        components = [
            ('fMRI Encoder', self.fmri_encoder),
            ('Stimulus Encoder', self.stim_encoder),
            ('CLIP Correlation', self.clip_correlation),
            ('Diffusion Decoder', self.diffusion_decoder),
            ('GAN Decoder', self.gan_decoder)
        ]
        
        total_params = 0
        for name, model in components:
            params = sum(p.numel() for p in model.parameters())
            total_params += params
            print(f"✅ {name}: {params:,} params")
        
        print(f"📊 Total Parameters: {total_params:,}")
    
    def train_encoders(self, train_loader, test_loader, epochs=15, lr=1e-3):
        """Phase 1: Train encoders"""
        print(f"\n🎯 Phase 1: Training Encoders ({epochs} epochs)")
        print("=" * 50)
        
        # Setup optimizer
        params = list(self.fmri_encoder.parameters()) + \
                list(self.stim_encoder.parameters()) + \
                list(self.clip_correlation.parameters())
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        losses = []
        
        for epoch in range(epochs):
            # Training
            self.fmri_encoder.train()
            self.stim_encoder.train()
            self.clip_correlation.train()
            
            train_loss = 0.0
            for batch in train_loader:
                fmri = batch['fmri'].to(self.device)
                stimulus = batch['stimulus'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                fmri_latent = self.fmri_encoder(fmri)
                stim_latent = self.stim_encoder(stimulus)
                
                # Contrastive loss
                loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_encoders(test_loader)
            
            # Log progress
            lr_current = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr_current:.6f}")
            
            scheduler.step()
        
        print(f"  ✅ Encoder training completed")
        return losses
    
    def train_decoders(self, train_loader, test_loader, epochs=10, lr=1e-3):
        """Phase 2: Train decoders with proper loss functions"""
        print(f"\n🎯 Phase 2: Training Decoders ({epochs} epochs)")
        print("=" * 50)
        
        # Freeze encoders
        for param in self.fmri_encoder.parameters():
            param.requires_grad = False
        for param in self.stim_encoder.parameters():
            param.requires_grad = False
        for param in self.clip_correlation.parameters():
            param.requires_grad = False
        
        # Setup optimizer for decoders only
        params = list(self.diffusion_decoder.parameters()) + \
                list(self.gan_decoder.parameters())
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        losses = []
        
        for epoch in range(epochs):
            # Training
            self.diffusion_decoder.train()
            self.gan_decoder.train()
            
            epoch_loss = 0.0
            diff_loss_total = 0.0
            gan_loss_total = 0.0
            
            for batch in train_loader:
                fmri = batch['fmri'].to(self.device)
                stimulus = batch['stimulus'].to(self.device)
                
                optimizer.zero_grad()
                
                # Get frozen representations
                with torch.no_grad():
                    fmri_latent = self.fmri_encoder(fmri)
                    stim_latent = self.stim_encoder(stimulus)
                    correlation = self.clip_correlation(fmri_latent, stim_latent)
                
                # Diffusion training (proper loss)
                if self.use_fixed_diffusion:
                    # Fixed diffusion uses noise prediction loss
                    _, diff_loss = self.diffusion_decoder(correlation, fmri_latent, stimulus)
                else:
                    # Original diffusion uses MSE loss
                    diff_pred = self.diffusion_decoder(correlation, fmri_latent)
                    diff_loss = nn.MSELoss()(diff_pred, stimulus)
                
                # GAN training (MSE loss)
                gan_pred = self.gan_decoder(correlation, fmri_latent)
                gan_loss = nn.MSELoss()(gan_pred, stimulus)
                
                # Combined loss
                total_loss = diff_loss + gan_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                diff_loss_total += diff_loss.item()
                gan_loss_total += gan_loss.item()
            
            epoch_loss /= len(train_loader)
            diff_loss_total /= len(train_loader)
            gan_loss_total /= len(train_loader)
            
            losses.append(epoch_loss)
            
            # Validation
            val_loss = self._validate_decoders(test_loader)
            
            # Log progress
            print(f"  Epoch {epoch+1:3d}: Total={epoch_loss:.4f}, Diff={diff_loss_total:.4f}, GAN={gan_loss_total:.4f}, Val={val_loss:.4f}")
            
            scheduler.step()
        
        print(f"  ✅ Decoder training completed")
        return losses
    
    def _validate_encoders(self, test_loader):
        """Validate encoder training"""
        self.fmri_encoder.eval()
        self.stim_encoder.eval()
        self.clip_correlation.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(self.device)
                stimulus = batch['stimulus'].to(self.device)
                
                fmri_latent = self.fmri_encoder(fmri)
                stim_latent = self.stim_encoder(stimulus)
                
                loss = self.clip_correlation.compute_contrastive_loss(fmri_latent, stim_latent)
                val_loss += loss.item()
        
        return val_loss / len(test_loader)
    
    def _validate_decoders(self, test_loader):
        """Validate decoder training"""
        self.diffusion_decoder.eval()
        self.gan_decoder.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(self.device)
                stimulus = batch['stimulus'].to(self.device)
                
                fmri_latent = self.fmri_encoder(fmri)
                stim_latent = self.stim_encoder(stimulus)
                correlation = self.clip_correlation(fmri_latent, stim_latent)
                
                # Diffusion validation
                if self.use_fixed_diffusion:
                    diff_pred = self.diffusion_decoder(correlation, fmri_latent, num_inference_steps=5)
                    diff_loss = nn.MSELoss()(diff_pred, stimulus)
                else:
                    diff_pred = self.diffusion_decoder(correlation, fmri_latent)
                    diff_loss = nn.MSELoss()(diff_pred, stimulus)
                
                # GAN validation
                gan_pred = self.gan_decoder(correlation, fmri_latent)
                gan_loss = nn.MSELoss()(gan_pred, stimulus)
                
                val_loss += (diff_loss + gan_loss).item()
        
        return val_loss / len(test_loader)
    
    def evaluate(self, test_loader):
        """Comprehensive evaluation"""
        print(f"\n📊 Evaluating {'Fixed' if self.use_fixed_diffusion else 'Original'} Diffusion Model")
        print("=" * 60)
        
        # Set all models to eval mode
        self.fmri_encoder.eval()
        self.stim_encoder.eval()
        self.clip_correlation.eval()
        self.diffusion_decoder.eval()
        self.gan_decoder.eval()
        
        all_diff_preds = []
        all_gan_preds = []
        all_targets = []
        all_fmri_latents = []
        all_stim_latents = []
        
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(self.device)
                stimulus = batch['stimulus'].to(self.device)
                
                # Get representations
                fmri_latent = self.fmri_encoder(fmri)
                stim_latent = self.stim_encoder(stimulus)
                correlation = self.clip_correlation(fmri_latent, stim_latent)
                
                # Generate predictions
                if self.use_fixed_diffusion:
                    diff_pred = self.diffusion_decoder(correlation, fmri_latent, num_inference_steps=10)
                else:
                    diff_pred = self.diffusion_decoder(correlation, fmri_latent)
                
                gan_pred = self.gan_decoder(correlation, fmri_latent)
                
                all_diff_preds.append(diff_pred.cpu())
                all_gan_preds.append(gan_pred.cpu())
                all_targets.append(stimulus.cpu())
                all_fmri_latents.append(fmri_latent.cpu())
                all_stim_latents.append(stim_latent.cpu())
        
        # Concatenate results
        diff_preds = torch.cat(all_diff_preds, dim=0)
        gan_preds = torch.cat(all_gan_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        fmri_latents = torch.cat(all_fmri_latents, dim=0)
        stim_latents = torch.cat(all_stim_latents, dim=0)
        
        # Evaluate metrics
        print("🔍 Diffusion Decoder Metrics:")
        diff_metrics = evaluate_decoding_performance(diff_preds, targets)
        for metric, value in diff_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n🔍 GAN Decoder Metrics:")
        gan_metrics = evaluate_decoding_performance(gan_preds, targets)
        for metric, value in gan_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Cross-modal retrieval
        N = fmri_latents.shape[0]
        similarity_matrix = torch.mm(fmri_latents, stim_latents.t())
        
        fmri_to_stim_correct = sum(1 for i in range(N) if torch.argmax(similarity_matrix[i, :]).item() == i)
        stim_to_fmri_correct = sum(1 for i in range(N) if torch.argmax(similarity_matrix[:, i]).item() == i)
        
        retrieval_accuracy = ((fmri_to_stim_correct + stim_to_fmri_correct) / (2 * N)) * 100
        
        print(f"\n🎯 Cross-Modal Retrieval: {retrieval_accuracy:.1f}%")
        
        return {
            'diffusion_metrics': diff_metrics,
            'gan_metrics': gan_metrics,
            'retrieval_accuracy': retrieval_accuracy,
            'predictions': {
                'diffusion': diff_preds,
                'gan': gan_preds,
                'targets': targets
            }
        }

def compare_diffusion_implementations():
    """Compare original vs fixed diffusion"""
    print("🔄 Comparing Original vs Fixed Diffusion")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    train_dataset, test_dataset, _ = load_miyawaki_dataset_corrected(dataset_path)
    train_loader, test_loader = create_dataloaders_corrected(
        train_dataset, test_dataset, batch_size=8
    )
    
    results = {}
    
    # Test both implementations
    for use_fixed in [False, True]:
        name = "Fixed" if use_fixed else "Original"
        print(f"\n{'='*20} {name} Diffusion {'='*20}")
        
        # Initialize trainer
        trainer = FixedModularTrainer(device=device, use_fixed_diffusion=use_fixed)
        
        # Quick training
        start_time = time.time()
        
        # Phase 1: Encoders
        encoder_losses = trainer.train_encoders(train_loader, test_loader, epochs=10, lr=1e-3)
        
        # Phase 2: Decoders
        decoder_losses = trainer.train_decoders(train_loader, test_loader, epochs=8, lr=1e-3)
        
        training_time = time.time() - start_time
        
        # Evaluation
        eval_results = trainer.evaluate(test_loader)
        
        results[name] = {
            'eval_results': eval_results,
            'training_time': training_time,
            'encoder_losses': encoder_losses,
            'decoder_losses': decoder_losses
        }
        
        print(f"\n📊 {name} Results:")
        print(f"  Training Time: {training_time/60:.1f} minutes")
        print(f"  Retrieval Accuracy: {eval_results['retrieval_accuracy']:.1f}%")
        print(f"  Diffusion MSE: {eval_results['diffusion_metrics']['mse']:.4f}")
        print(f"  GAN MSE: {eval_results['gan_metrics']['mse']:.4f}")
    
    # Compare results
    print(f"\n" + "="*70)
    print(f"🏆 COMPARISON RESULTS")
    print(f"="*70)
    
    orig_results = results['Original']['eval_results']
    fixed_results = results['Fixed']['eval_results']
    
    print(f"📊 Diffusion MSE Comparison:")
    orig_mse = orig_results['diffusion_metrics']['mse']
    fixed_mse = fixed_results['diffusion_metrics']['mse']
    improvement = (orig_mse - fixed_mse) / orig_mse * 100
    print(f"  Original: {orig_mse:.4f}")
    print(f"  Fixed:    {fixed_mse:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\n🎯 Retrieval Accuracy Comparison:")
    orig_acc = orig_results['retrieval_accuracy']
    fixed_acc = fixed_results['retrieval_accuracy']
    print(f"  Original: {orig_acc:.1f}%")
    print(f"  Fixed:    {fixed_acc:.1f}%")
    print(f"  Difference: {fixed_acc - orig_acc:.1f}%")
    
    print(f"\n⏱️ Training Time Comparison:")
    orig_time = results['Original']['training_time']
    fixed_time = results['Fixed']['training_time']
    print(f"  Original: {orig_time/60:.1f} minutes")
    print(f"  Fixed:    {fixed_time/60:.1f} minutes")
    
    # Save results
    torch.save(results, 'diffusion_comparison_results.pth')
    print(f"\n💾 Results saved to 'diffusion_comparison_results.pth'")
    
    return results

if __name__ == "__main__":
    results = compare_diffusion_implementations()
