#!/usr/bin/env python3
"""
Compare MiyawakaTrainer vs MiyawakiDualEncoderPipeline
Head-to-head comparison of both architectures
"""

import torch
import time
from pathlib import Path

# Import both architectures
from miyawakitrain import MiyawakaTrainer
from main_pipeline_fixed import MiyawakiDualEncoderPipeline
from miyawakidataset import load_miyawaki_dataset_corrected, create_dataloaders_corrected
from metriks import evaluate_decoding_performance, compare_decoders

def train_miyawaka_trainer(trainer, train_loader, test_loader, device='cuda'):
    """Train using MiyawakaTrainer (3-phase approach)"""
    print("🎯 Training MiyawakaTrainer (3-Phase Approach)")
    print("=" * 60)
    
    trainer = trainer.to(device)
    
    # Convert dataloader format for MiyawakaTrainer
    def convert_dataloader(loader):
        converted_batches = []
        for batch in loader:
            fmri = batch['fmri'].to(device)
            stimulus = batch['stimulus'].to(device)
            converted_batches.append((fmri, stimulus))
        return converted_batches
    
    train_batches = convert_dataloader(train_loader)
    test_batches = convert_dataloader(test_loader)
    
    start_time = time.time()
    
    # Phase 1: Train encoders (reduced epochs for comparison)
    print("Phase 1: Training Encoders...")
    trainer.train_phase1_encoders(train_batches, epochs=15)
    
    # Phase 2: Train correlation (reduced epochs)
    print("\nPhase 2: Training Correlation...")
    trainer.train_phase2_correlation(train_batches, epochs=10)
    
    # Phase 3: Train decoders (reduced epochs)
    print("\nPhase 3: Training Decoders...")
    trainer.train_phase3_decoders(train_batches, test_batches, epochs=10)
    
    training_time = time.time() - start_time
    print(f"\n⏱️ MiyawakaTrainer training time: {training_time/60:.1f} minutes")
    
    return trainer, training_time

def train_pipeline(pipeline, train_loader, test_loader, device='cuda'):
    """Train using MiyawakiDualEncoderPipeline (unified approach)"""
    print("\n🎯 Training MiyawakiDualEncoderPipeline (Unified Approach)")
    print("=" * 60)
    
    from main_pipeline_fixed import train_dual_encoder_phase, train_decoders_phase
    
    start_time = time.time()
    
    # Phase 1: Train dual encoder
    print("Phase 1: Training Dual Encoder + CLIP Correlation")
    train_dual_encoder_phase(pipeline, train_loader, test_loader, num_epochs=15, lr=1e-3)
    
    # Phase 2: Train decoders
    print("\nPhase 2: Training Generative Decoders")
    train_decoders_phase(pipeline, train_loader, test_loader, num_epochs=10, lr=1e-3)
    
    training_time = time.time() - start_time
    print(f"\n⏱️ MiyawakiDualEncoderPipeline training time: {training_time/60:.1f} minutes")
    
    return pipeline, training_time

def evaluate_trainer(trainer, test_loader, device='cuda'):
    """Evaluate MiyawakaTrainer"""
    print("\n📊 Evaluating MiyawakaTrainer")
    print("=" * 40)
    
    results = trainer.evaluate(test_loader)
    
    # Calculate metrics
    diffusion_metrics = evaluate_decoding_performance(results['diffusion_preds'], results['targets'])
    gan_metrics = evaluate_decoding_performance(results['gan_preds'], results['targets'])
    
    # Cross-modal retrieval
    fmri_latents = results['fmri_latents']
    stim_latents = results['stim_latents']
    N = fmri_latents.shape[0]
    
    similarity_matrix = torch.mm(fmri_latents, stim_latents.t())
    
    # fMRI → Stimulus retrieval
    fmri_to_stim_correct = sum(1 for i in range(N) if torch.argmax(similarity_matrix[i, :]).item() == i)
    stim_to_fmri_correct = sum(1 for i in range(N) if torch.argmax(similarity_matrix[:, i]).item() == i)
    
    retrieval_accuracy = ((fmri_to_stim_correct + stim_to_fmri_correct) / (2 * N)) * 100
    
    print(f"Cross-Modal Retrieval: {retrieval_accuracy:.1f}%")
    print(f"Diffusion MSE: {diffusion_metrics['mse']:.4f}")
    print(f"GAN MSE: {gan_metrics['mse']:.4f}")
    
    return {
        'diffusion_metrics': diffusion_metrics,
        'gan_metrics': gan_metrics,
        'retrieval_accuracy': retrieval_accuracy,
        'predictions': results
    }

def evaluate_pipeline(pipeline, test_loader, device='cuda'):
    """Evaluate MiyawakiDualEncoderPipeline"""
    print("\n📊 Evaluating MiyawakiDualEncoderPipeline")
    print("=" * 40)
    
    from evaluatepipeline import evaluate_model
    results = evaluate_model(pipeline, test_loader, device)
    
    print(f"Cross-Modal Retrieval: {results['retrieval_accuracy']:.1f}%")
    print(f"Diffusion MSE: {results['diffusion_metrics']['mse']:.4f}")
    print(f"GAN MSE: {results['gan_metrics']['mse']:.4f}")
    
    return results

def compare_results(trainer_results, pipeline_results):
    """Compare results from both architectures"""
    print("\n" + "=" * 80)
    print("🏆 ARCHITECTURE COMPARISON RESULTS")
    print("=" * 80)
    
    # Retrieval comparison
    trainer_retrieval = trainer_results['retrieval_accuracy']
    pipeline_retrieval = pipeline_results['retrieval_accuracy']
    
    print(f"\n🎯 Cross-Modal Retrieval Accuracy:")
    print(f"  MiyawakaTrainer:           {trainer_retrieval:.1f}%")
    print(f"  MiyawakiDualEncoderPipeline: {pipeline_retrieval:.1f}%")
    print(f"  Winner: {'MiyawakaTrainer' if trainer_retrieval > pipeline_retrieval else 'MiyawakiDualEncoderPipeline'}")
    
    # Reconstruction comparison
    print(f"\n🔍 Reconstruction Quality (MSE - Lower is Better):")
    
    print(f"\n  Diffusion Decoder:")
    trainer_diff_mse = trainer_results['diffusion_metrics']['mse']
    pipeline_diff_mse = pipeline_results['diffusion_metrics']['mse']
    print(f"    MiyawakaTrainer:           {trainer_diff_mse:.4f}")
    print(f"    MiyawakiDualEncoderPipeline: {pipeline_diff_mse:.4f}")
    print(f"    Winner: {'MiyawakaTrainer' if trainer_diff_mse < pipeline_diff_mse else 'MiyawakiDualEncoderPipeline'}")
    
    print(f"\n  GAN Decoder:")
    trainer_gan_mse = trainer_results['gan_metrics']['mse']
    pipeline_gan_mse = pipeline_results['gan_metrics']['mse']
    print(f"    MiyawakaTrainer:           {trainer_gan_mse:.4f}")
    print(f"    MiyawakiDualEncoderPipeline: {pipeline_gan_mse:.4f}")
    print(f"    Winner: {'MiyawakaTrainer' if trainer_gan_mse < pipeline_gan_mse else 'MiyawakiDualEncoderPipeline'}")
    
    # Overall winner
    trainer_score = 0
    pipeline_score = 0
    
    if trainer_retrieval > pipeline_retrieval:
        trainer_score += 1
    else:
        pipeline_score += 1
    
    if trainer_diff_mse < pipeline_diff_mse:
        trainer_score += 1
    else:
        pipeline_score += 1
    
    if trainer_gan_mse < pipeline_gan_mse:
        trainer_score += 1
    else:
        pipeline_score += 1
    
    print(f"\n🏆 OVERALL WINNER:")
    if trainer_score > pipeline_score:
        print(f"  🥇 MiyawakaTrainer ({trainer_score}/3 metrics)")
        print(f"  🥈 MiyawakiDualEncoderPipeline ({pipeline_score}/3 metrics)")
    else:
        print(f"  🥇 MiyawakiDualEncoderPipeline ({pipeline_score}/3 metrics)")
        print(f"  🥈 MiyawakaTrainer ({trainer_score}/3 metrics)")
    
    return {
        'trainer_score': trainer_score,
        'pipeline_score': pipeline_score,
        'winner': 'MiyawakaTrainer' if trainer_score > pipeline_score else 'MiyawakiDualEncoderPipeline'
    }

def main():
    """Main comparison function"""
    print("🚀 ARCHITECTURE COMPARISON: MiyawakaTrainer vs MiyawakiDualEncoderPipeline")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    
    # Load dataset
    print("📂 Loading Miyawaki dataset...")
    dataset_path = Path("../dataset/miyawaki_structured_28x28.mat")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    train_dataset, test_dataset, fmri_stats = load_miyawaki_dataset_corrected(dataset_path)
    train_loader, test_loader = create_dataloaders_corrected(
        train_dataset, test_dataset, batch_size=batch_size
    )
    
    # Initialize both architectures
    print("\n🏗️ Initializing architectures...")
    trainer = MiyawakaTrainer()
    pipeline = MiyawakiDualEncoderPipeline(fmri_dim=967, latent_dim=512, device=device)
    
    # Train both architectures
    trainer, trainer_time = train_miyawaka_trainer(trainer, train_loader, test_loader, device)
    pipeline, pipeline_time = train_pipeline(pipeline, train_loader, test_loader, device)
    
    # Evaluate both architectures
    trainer_results = evaluate_trainer(trainer, test_loader, device)
    pipeline_results = evaluate_pipeline(pipeline, test_loader, device)
    
    # Compare results
    comparison = compare_results(trainer_results, pipeline_results)
    
    # Training time comparison
    print(f"\n⏱️ Training Time Comparison:")
    print(f"  MiyawakaTrainer:           {trainer_time/60:.1f} minutes")
    print(f"  MiyawakiDualEncoderPipeline: {pipeline_time/60:.1f} minutes")
    print(f"  Faster: {'MiyawakaTrainer' if trainer_time < pipeline_time else 'MiyawakiDualEncoderPipeline'}")
    
    # Save results
    torch.save({
        'trainer_results': trainer_results,
        'pipeline_results': pipeline_results,
        'comparison': comparison,
        'training_times': {
            'trainer': trainer_time,
            'pipeline': pipeline_time
        }
    }, 'architecture_comparison_results.pth')
    
    print(f"\n💾 Complete comparison saved to 'architecture_comparison_results.pth'")
    print("✅ Architecture comparison completed!")
    
    return comparison

if __name__ == "__main__":
    results = main()
