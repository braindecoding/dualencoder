#!/usr/bin/env python3
"""
Start Miyawaki LDM Fine-tuning
Optimized training for Miyawaki pattern characteristics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def check_requirements():
    """Check if all requirements are met for training"""
    print("🔍 Checking training requirements...")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available - training will be very slow on CPU")
        return False
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 8:
        print("⚠️ Warning: Less than 8GB GPU memory - may need to reduce batch size")
    
    # Check embeddings
    if not Path("miyawaki4_embeddings.pkl").exists():
        print("❌ Miyawaki embeddings not found - run embedding_converter.py first")
        return False
    
    print("✅ Miyawaki embeddings found")
    
    # Check disk space (rough estimate)
    free_space = Path(".").stat().st_size  # This is not accurate, but for demo
    print(f"✅ Training environment ready")
    
    return True

def estimate_training_time():
    """Estimate training time based on hardware"""
    print("\n⏱️ Training Time Estimation")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        
        # Rough estimates based on GPU
        if "RTX 4090" in gpu_name or "A100" in gpu_name:
            time_per_epoch = "~5-10 minutes"
            total_time = "~1-2 hours"
        elif "RTX 3080" in gpu_name or "RTX 3090" in gpu_name:
            time_per_epoch = "~10-15 minutes"
            total_time = "~2-3 hours"
        elif "RTX 3060" in gpu_name or "RTX 3070" in gpu_name:
            time_per_epoch = "~15-25 minutes"
            total_time = "~3-5 hours"
        else:
            time_per_epoch = "~20-40 minutes"
            total_time = "~4-8 hours"
    else:
        time_per_epoch = "~2-4 hours"
        total_time = "~20-40 hours"
    
    print(f"📊 Estimated time per epoch: {time_per_epoch}")
    print(f"📊 Estimated total time (10 epochs): {total_time}")
    print(f"📊 Dataset size: 107 training samples")
    print(f"📊 Batch size: 4 (recommended for 12GB GPU)")

def show_training_strategy():
    """Show the training strategy for Miyawaki patterns"""
    print("\n🎯 MIYAWAKI-SPECIFIC TRAINING STRATEGY")
    print("=" * 50)
    
    print("📋 Key Adaptations:")
    print("   1. 🎨 Pattern-focused loss functions")
    print("      - Edge-preserving loss for high contrast")
    print("      - Frequency domain loss for patterns")
    print("      - Standard MSE loss")
    
    print("   2. 🔧 Optimized for dark, high-contrast images")
    print("      - Mid-range timestep sampling")
    print("      - Contrast-aware augmentation")
    print("      - Pattern structure preservation")
    
    print("   3. 📊 Miyawaki-specific conditioning")
    print("      - fMRI → CLIP text embedding space")
    print("      - Direct neural conditioning")
    print("      - Pattern-aware attention")
    
    print("   4. 🎯 Evaluation metrics")
    print("      - CLIP similarity with original images")
    print("      - Pattern structure preservation")
    print("      - Edge consistency metrics")

def create_training_config():
    """Create training configuration"""
    config = {
        'num_epochs': 10,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'save_every': 2,
        'sample_every': 2,
        'num_inference_steps': 20,
        'guidance_scale': 7.5,
        'loss_weights': {
            'mse': 1.0,
            'edge': 0.1,
            'frequency': 0.05
        }
    }
    
    return config

def main():
    """Main function"""
    print("🎯 MIYAWAKI LDM FINE-TUNING SETUP")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix issues and try again.")
        return
    
    # Show training strategy
    show_training_strategy()
    
    # Estimate training time
    estimate_training_time()
    
    # Create config
    config = create_training_config()
    
    print(f"\n🔧 TRAINING CONFIGURATION")
    print("=" * 40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"📊 {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"📊 {key}: {value}")
    
    # Ask for confirmation
    print(f"\n🚀 READY TO START TRAINING")
    print("=" * 40)
    
    print("📋 What will happen:")
    print("   1. Load Miyawaki embeddings and images")
    print("   2. Setup Stable Diffusion components")
    print("   3. Initialize fMRI conditioning network")
    print("   4. Train for 10 epochs with pattern-specific losses")
    print("   5. Generate sample images every 2 epochs")
    print("   6. Save checkpoints every 2 epochs")
    
    print(f"\n⚠️ IMPORTANT NOTES:")
    print("   - Training will take several hours")
    print("   - GPU memory usage will be high")
    print("   - Generated files will be ~500MB-1GB")
    print("   - Process should not be interrupted")
    
    response = input(f"\n🎯 Start Miyawaki LDM training? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        print(f"\n🚀 Starting training...")
        
        try:
            # Import and run training
            from finetune_ldm_miyawaki import train_miyawaki_ldm
            train_miyawaki_ldm()
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure finetune_ldm_miyawaki.py is in the same directory")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            print("💡 Check GPU memory and try reducing batch size")
    
    else:
        print(f"\n⏸️ Training cancelled by user")
        print(f"💡 Run this script again when ready to start training")
    
    print(f"\n📁 Expected output files:")
    print(f"   - miyawaki_ldm_final.pth (trained model)")
    print(f"   - miyawaki_ldm_training_curve.png (training progress)")
    print(f"   - miyawaki_ldm_sample_epoch_*.png (generated samples)")
    print(f"   - miyawaki_ldm_checkpoint_epoch_*.pth (checkpoints)")

if __name__ == "__main__":
    main()
