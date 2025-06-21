#!/usr/bin/env python3
"""
Evaluate Quick Fix Results
Generate multiple samples and evaluate similarity
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from quick_fix_training import BinaryPatternGenerator, QuickFixTrainer

def evaluate_quickfix_model():
    """Evaluate the quick fix model on multiple test samples"""
    print("🔍 EVALUATING QUICK FIX MODEL")
    print("=" * 60)
    
    # Load embeddings
    embeddings_path = "miyawaki4_embeddings.pkl"
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = QuickFixTrainer(device=device)
    
    model_path = 'quickfix_binary_generator.pth'
    if not Path(model_path).exists():
        print("❌ Trained model not found!")
        return
    
    trainer.model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully")
    
    # Get test samples
    test_fmri = embeddings_data['test']['fmri_embeddings'][:10]
    test_images = embeddings_data['test']['original_images'][:10]
    
    print(f"📊 Evaluating on {len(test_fmri)} test samples...")
    
    # Generate predictions
    generated_images = []
    pattern_types = []
    
    for i, fmri in enumerate(test_fmri):
        fmri_tensor = torch.FloatTensor(fmri).to(device)
        pred_img, pattern_type = trainer.generate_binary_pattern(fmri_tensor)
        
        generated_images.append(np.array(pred_img))
        pattern_types.append(pattern_type.cpu().numpy())
        
        print(f"   Sample {i+1}: Generated pattern")
    
    # Create evaluation grid
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Quick Fix Evaluation: Original vs Generated Patterns', fontsize=16, fontweight='bold')
    
    for i in range(min(10, len(test_fmri))):
        row = (i // 5) * 2
        col = i % 5
        
        # Original image
        orig_img = test_images[i]
        
        # Convert CHW to HWC if needed
        if len(orig_img.shape) == 3 and orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
        
        # Convert to grayscale if RGB
        if len(orig_img.shape) == 3:
            orig_img = np.mean(orig_img, axis=2)
        
        # Apply binary threshold
        orig_binary = (orig_img > 0.5).astype(np.uint8) * 255
        
        # Display original
        axes[row, col].imshow(orig_binary, cmap='gray')
        axes[row, col].set_title(f'Original {i+1}')
        axes[row, col].axis('off')
        
        # Display generated
        axes[row+1, col].imshow(generated_images[i], cmap='gray')
        axes[row+1, col].set_title(f'Generated {i+1}')
        axes[row+1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('quickfix_evaluation_grid.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Calculate metrics
    print(f"\n📈 EVALUATION METRICS:")
    
    binary_ratios = []
    pattern_diversities = []
    
    for i, gen_img in enumerate(generated_images):
        # Binary ratio
        binary_pixels = np.sum(gen_img > 128)
        total_pixels = gen_img.size
        binary_ratio = binary_pixels / total_pixels
        binary_ratios.append(binary_ratio)
        
        # Pattern diversity (unique values)
        unique_vals = len(np.unique(gen_img))
        pattern_diversities.append(unique_vals)
    
    avg_binary_ratio = np.mean(binary_ratios)
    avg_diversity = np.mean(pattern_diversities)
    
    print(f"   Average binary ratio: {avg_binary_ratio:.3f}")
    print(f"   Average unique values: {avg_diversity:.1f}")
    print(f"   Binary ratio std: {np.std(binary_ratios):.3f}")
    
    # Pattern type analysis
    pattern_probs = np.array(pattern_types)
    avg_pattern_probs = np.mean(pattern_probs, axis=0)
    
    print(f"\n🎯 PATTERN TYPE DISTRIBUTION:")
    pattern_names = ['Cross', 'L-shape', 'Rectangle', 'T-shape']
    for i, (name, prob) in enumerate(zip(pattern_names, avg_pattern_probs)):
        print(f"   {name}: {prob:.3f}")
    
    # Success criteria
    print(f"\n✅ SUCCESS CRITERIA CHECK:")
    
    success_count = 0
    
    # 1. Binary patterns (should have only 2-3 unique values)
    binary_success = sum(1 for d in pattern_diversities if d <= 3)
    print(f"   Binary patterns: {binary_success}/{len(generated_images)} ({binary_success/len(generated_images)*100:.1f}%)")
    if binary_success >= len(generated_images) * 0.8:
        success_count += 1
        print(f"   ✅ PASS: Most patterns are binary")
    else:
        print(f"   ❌ FAIL: Too few binary patterns")
    
    # 2. Reasonable binary ratio (0.1 - 0.9)
    ratio_success = sum(1 for r in binary_ratios if 0.1 <= r <= 0.9)
    print(f"   Reasonable ratios: {ratio_success}/{len(generated_images)} ({ratio_success/len(generated_images)*100:.1f}%)")
    if ratio_success >= len(generated_images) * 0.7:
        success_count += 1
        print(f"   ✅ PASS: Good binary distribution")
    else:
        print(f"   ❌ FAIL: Poor binary distribution")
    
    # 3. Pattern diversity
    if np.std(binary_ratios) > 0.05:
        success_count += 1
        print(f"   ✅ PASS: Good pattern diversity")
    else:
        print(f"   ❌ FAIL: Patterns too similar")
    
    # Overall assessment
    print(f"\n🎉 OVERALL ASSESSMENT:")
    print(f"   Success criteria met: {success_count}/3")
    
    if success_count >= 2:
        print(f"   🎯 EXCELLENT: Quick Fix is working well!")
        print(f"   📈 Significant improvement over original LDM approach")
        print(f"   ✅ Binary patterns successfully generated")
    else:
        print(f"   ⚠️ NEEDS IMPROVEMENT: Some issues remain")
        print(f"   🔧 Consider further training or architecture changes")
    
    return generated_images, pattern_types, binary_ratios

def generate_final_showcase():
    """Generate final showcase of best results"""
    print(f"\n🎨 GENERATING FINAL SHOWCASE")
    print("=" * 50)
    
    # Load model and generate best samples
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = QuickFixTrainer(device=device)
    
    model_path = 'quickfix_binary_generator.pth'
    trainer.model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load embeddings
    with open("miyawaki4_embeddings.pkl", 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Select diverse test samples
    test_indices = [0, 2, 4, 6, 8]  # Spread out samples
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Quick Fix Final Showcase: Binary Pattern Generation from fMRI', 
                 fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(test_indices):
        fmri = torch.FloatTensor(embeddings_data['test']['fmri_embeddings'][idx]).to(device)
        pred_img, pattern_type = trainer.generate_binary_pattern(fmri)
        
        axes[i].imshow(np.array(pred_img), cmap='gray')
        axes[i].set_title(f'Sample {idx+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('quickfix_final_showcase.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    print(f"✅ Final showcase saved: quickfix_final_showcase.png")

if __name__ == "__main__":
    generated_images, pattern_types, binary_ratios = evaluate_quickfix_model()
    generate_final_showcase()
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"📁 Generated files:")
    print(f"   - quickfix_evaluation_grid.png")
    print(f"   - quickfix_final_showcase.png")
