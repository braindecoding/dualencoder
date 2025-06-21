#!/usr/bin/env python3
"""
Modular Training Pipeline - Easy Tracking
Step-by-step training with clear progress tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Import our components
from fmriencoder import fMRI_Encoder
from stimencoder import Shape_Encoder
from clipcorrtrain import CLIP_Correlation
from diffusion import Diffusion_Decoder
from gan import GAN_Decoder
from miyawakidataset import load_miyawaki_dataset_corrected, create_dataloaders_corrected
from metriks import evaluate_decoding_performance

class ModularTracker:
    """Track training progress and metrics"""
    
    def __init__(self):
        self.phase_losses = {}
        self.phase_times = {}
        self.metrics_history = {}
        self.current_phase = None
    
    def start_phase(self, phase_name):
        """Start tracking a new phase"""
        self.current_phase = phase_name
        self.phase_losses[phase_name] = []
        self.phase_times[phase_name] = time.time()
        print(f"\n🎯 Starting {phase_name}")
        print("=" * 50)
    
    def log_epoch(self, epoch, loss, additional_info=""):
        """Log epoch progress"""
        self.phase_losses[self.current_phase].append(loss)
        print(f"  Epoch {epoch+1:3d}: Loss = {loss:.6f} {additional_info}")
    
    def end_phase(self):
        """End current phase tracking"""
        if self.current_phase:
            duration = time.time() - self.phase_times[self.current_phase]
            self.phase_times[self.current_phase] = duration
            final_loss = self.phase_losses[self.current_phase][-1]
            print(f"  ✅ {self.current_phase} completed in {duration:.1f}s")
            print(f"  📊 Final loss: {final_loss:.6f}")
            self.current_phase = None
    
    def save_progress_plot(self, save_path='training_progress.png'):
        """Save training progress plot"""
        fig, axes = plt.subplots(1, len(self.phase_losses), figsize=(15, 4))
        if len(self.phase_losses) == 1:
            axes = [axes]
        
        for i, (phase, losses) in enumerate(self.phase_losses.items()):
            axes[i].plot(losses)
            axes[i].set_title(f'{phase} Loss')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training progress saved to {save_path}")

class ModularMiyawakiTrainer:
    """Modular trainer with easy tracking"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tracker = ModularTracker()
        
        print("🏗️ Initializing Modular Components")
        print("=" * 50)
        
        # Initialize components
        self.fmri_encoder = fMRI_Encoder(fmri_dim=967, latent_dim=512).to(device)
        self.stim_encoder = Shape_Encoder(latent_dim=512).to(device)
        self.clip_correlation = CLIP_Correlation(latent_dim=512).to(device)
        self.diffusion_decoder = Diffusion_Decoder(correlation_dim=512).to(device)
        self.gan_decoder = GAN_Decoder(correlation_dim=512).to(device)
        
        print(f"✅ fMRI Encoder: {sum(p.numel() for p in self.fmri_encoder.parameters()):,} params")
        print(f"✅ Stimulus Encoder: {sum(p.numel() for p in self.stim_encoder.parameters()):,} params")
        print(f"✅ CLIP Correlation: {sum(p.numel() for p in self.clip_correlation.parameters()):,} params")
        print(f"✅ Diffusion Decoder: {sum(p.numel() for p in self.diffusion_decoder.parameters()):,} params")
        print(f"✅ GAN Decoder: {sum(p.numel() for p in self.gan_decoder.parameters()):,} params")
        
        total_params = sum(p.numel() for model in [self.fmri_encoder, self.stim_encoder, 
                          self.clip_correlation, self.diffusion_decoder, self.gan_decoder] 
                          for p in model.parameters())
        print(f"📊 Total Parameters: {total_params:,}")
    
    def train_encoders(self, train_loader, test_loader, epochs=20, lr=1e-3):
        """Phase 1: Train encoders with correlation learning"""
        self.tracker.start_phase("Phase 1: Encoder Training")
        
        # Setup optimizer for encoders + correlation
        params = list(self.fmri_encoder.parameters()) + \
                list(self.stim_encoder.parameters()) + \
                list(self.clip_correlation.parameters())
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
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
            
            # Validation
            val_loss = self._validate_encoders(test_loader)
            
            # Log progress
            lr_current = scheduler.get_last_lr()[0]
            self.tracker.log_epoch(epoch, train_loss, 
                                 f"| Val: {val_loss:.6f} | LR: {lr_current:.6f}")
            
            scheduler.step()
        
        self.tracker.end_phase()
        return self.tracker.phase_losses["Phase 1: Encoder Training"]
    
    def train_decoders(self, train_loader, test_loader, epochs=15, lr=1e-3):
        """Phase 2: Train decoders (frozen encoders)"""
        self.tracker.start_phase("Phase 2: Decoder Training")
        
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
        
        for epoch in range(epochs):
            # Training
            self.diffusion_decoder.train()
            self.gan_decoder.train()
            
            train_loss = 0.0
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
                
                # Generate predictions
                diff_pred = self.diffusion_decoder(correlation, fmri_latent)
                gan_pred = self.gan_decoder(correlation, fmri_latent)
                
                # Reconstruction losses
                diff_loss = nn.MSELoss()(diff_pred, stimulus)
                gan_loss = nn.MSELoss()(gan_pred, stimulus)
                
                total_loss = diff_loss + gan_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                diff_loss_total += diff_loss.item()
                gan_loss_total += gan_loss.item()
            
            train_loss /= len(train_loader)
            diff_loss_total /= len(train_loader)
            gan_loss_total /= len(train_loader)
            
            # Validation
            val_loss = self._validate_decoders(test_loader)
            
            # Log progress
            self.tracker.log_epoch(epoch, train_loss, 
                                 f"| Diff: {diff_loss_total:.4f} | GAN: {gan_loss_total:.4f} | Val: {val_loss:.4f}")
            
            scheduler.step()
        
        self.tracker.end_phase()
        return self.tracker.phase_losses["Phase 2: Decoder Training"]
    
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
                
                diff_pred = self.diffusion_decoder(correlation, fmri_latent)
                gan_pred = self.gan_decoder(correlation, fmri_latent)
                
                diff_loss = nn.MSELoss()(diff_pred, stimulus)
                gan_loss = nn.MSELoss()(gan_pred, stimulus)
                
                val_loss += (diff_loss + gan_loss).item()
        
        return val_loss / len(test_loader)
    
    def evaluate_comprehensive(self, test_loader):
        """Comprehensive evaluation"""
        print("\n📊 Comprehensive Evaluation")
        print("=" * 50)
        
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
        print(f"  fMRI → Stimulus: {fmri_to_stim_correct}/{N} ({fmri_to_stim_correct/N*100:.1f}%)")
        print(f"  Stimulus → fMRI: {stim_to_fmri_correct}/{N} ({stim_to_fmri_correct/N*100:.1f}%)")
        
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

def main():
    """Main modular training function"""
    print("🚀 Modular Miyawaki Training - Easy Tracking")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    # Load dataset
    print("\n📂 Loading Dataset")
    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    train_dataset, test_dataset, fmri_stats = load_miyawaki_dataset_corrected(dataset_path)
    train_loader, test_loader = create_dataloaders_corrected(
        train_dataset, test_dataset, batch_size=8
    )
    
    print(f"✅ Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    # Initialize trainer
    trainer = ModularMiyawakiTrainer(device=device)
    
    # Training phases
    start_time = time.time()
    
    # Phase 1: Train encoders
    encoder_losses = trainer.train_encoders(train_loader, test_loader, epochs=20, lr=1e-3)
    
    # Phase 2: Train decoders
    decoder_losses = trainer.train_decoders(train_loader, test_loader, epochs=15, lr=1e-3)
    
    total_time = time.time() - start_time
    
    # Evaluation
    results = trainer.evaluate_comprehensive(test_loader)
    
    # Save progress plot
    trainer.tracker.save_progress_plot('modular_training_progress.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 MODULAR TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"⏱️ Total Training Time: {total_time/60:.1f} minutes")
    print(f"🎯 Cross-Modal Retrieval: {results['retrieval_accuracy']:.1f}%")
    print(f"🔍 Diffusion MSE: {results['diffusion_metrics']['mse']:.4f}")
    print(f"🔍 GAN MSE: {results['gan_metrics']['mse']:.4f}")
    print(f"🏆 Better Decoder: {'Diffusion' if results['diffusion_metrics']['mse'] < results['gan_metrics']['mse'] else 'GAN'}")
    
    # Save complete results
    torch.save({
        'results': results,
        'training_history': trainer.tracker.phase_losses,
        'training_times': trainer.tracker.phase_times,
        'total_time': total_time
    }, 'modular_training_results.pth')
    
    print(f"\n💾 Results saved to 'modular_training_results.pth'")
    print("✅ Modular training completed successfully!")
    
    return results

if __name__ == "__main__":
    results = main()
