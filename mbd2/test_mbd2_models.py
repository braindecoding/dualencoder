#!/usr/bin/env python3
"""
Test MBD2 Models - Comprehensive Evaluation
Test both advanced and 400-epoch models on MindBigData EEG dataset
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import clip
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from enhanced_eeg_transformer import EnhancedEEGToEmbeddingModel

def load_mbd2_data():
    """Load MBD2 explicit data splits"""
    print("📂 Loading MBD2 explicit data splits...")
    
    with open('explicit_eeg_data_splits.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Extract test data
    eegTest = data['test']['eegTest']
    stimTest = data['test']['stimTest']
    labelsTest = data['test']['labelsTest']
    
    # Also get validation for comparison
    eegVal = data['validation']['eegVal']
    stimVal = data['validation']['stimVal']
    labelsVal = data['validation']['labelsVal']
    
    print(f"✅ Data loaded successfully!")
    print(f"   Test: {len(eegTest)} samples")
    print(f"   Validation: {len(eegVal)} samples")
    print(f"   EEG shape: {eegTest.shape}")
    print(f"   Labels: {set(labelsTest)} (digits 0-9)")
    
    return eegTest, stimTest, labelsTest, eegVal, stimVal, labelsVal

def load_clip_model(device):
    """Load CLIP model for evaluation"""
    print("📥 Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Freeze CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False
    
    print("✅ CLIP model loaded and frozen")
    return clip_model, clip_preprocess

def create_enhanced_model(device):
    """Create enhanced EEG model"""
    model = EnhancedEEGToEmbeddingModel(
        n_channels=14,
        seq_len=256,
        d_model=256,  # Enhanced dimension
        embedding_dim=512,
        nhead=8,
        num_layers=8,  # Enhanced layers
        patch_size=16,
        dropout=0.1
    ).to(device)
    return model

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"📥 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('best_val_accuracy', 'unknown')
        print(f"✅ Model loaded from epoch {epoch}, val_acc: {val_acc}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model state dict loaded")
    
    model.eval()
    return model

def evaluate_model(model, eeg_data, stim_data, labels, clip_model, clip_preprocess, device, phase_name="test", batch_size=32):
    """Evaluate model performance"""
    print(f"🔍 Evaluating model on {phase_name} set...")
    
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eeg_data), batch_size), desc=f"Evaluating {phase_name}"):
            batch_eeg = eeg_data[i:i+batch_size]
            batch_stim = stim_data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Convert EEG to tensor
            batch_eeg = torch.FloatTensor(batch_eeg).to(device)
            
            # Process images for CLIP
            processed_images = []
            for img in batch_stim:
                # Convert PIL image to tensor
                processed_img = clip_preprocess(img).unsqueeze(0)
                processed_images.append(processed_img)
            processed_images = torch.cat(processed_images, dim=0).to(device)
            
            # Generate embeddings
            eeg_embeddings = model(batch_eeg)
            clip_embeddings = clip_model.encode_image(processed_images).float()
            
            # Normalize embeddings
            eeg_embeddings = nn.functional.normalize(eeg_embeddings, dim=1)
            clip_embeddings = nn.functional.normalize(clip_embeddings, dim=1)
            
            # Compute similarities
            similarities = torch.cosine_similarity(eeg_embeddings, clip_embeddings, dim=1)
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(batch_labels)
    
    # Calculate metrics
    similarities = np.array(all_similarities)
    accuracy = (similarities > 0.0).mean()
    mean_similarity = similarities.mean()
    std_similarity = similarities.std()
    
    print(f"📊 {phase_name.title()} Results:")
    print(f"   Accuracy (>0): {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   Mean Similarity: {mean_similarity:.4f} ± {std_similarity:.4f}")
    print(f"   Min Similarity: {similarities.min():.4f}")
    print(f"   Max Similarity: {similarities.max():.4f}")
    
    return {
        'accuracy': accuracy,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'similarities': similarities,
        'labels': all_labels
    }

def compare_models():
    """Compare both MBD2 models"""
    print("🚀 MBD2 MODELS COMPARISON")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load data
    eegTest, stimTest, labelsTest, eegVal, stimVal, labelsVal = load_mbd2_data()
    
    # Load CLIP
    clip_model, clip_preprocess = load_clip_model(device)
    
    # Test both models
    models_to_test = [
        {
            'name': 'Enhanced Model (Advanced)',
            'path': 'advanced_eeg_model_best.pth',
            'description': '256d, 8 layers, 7.5M params'
        },
        {
            'name': '400-Epoch Model',
            'path': 'eeg_contrastive_400epochs_best.pth', 
            'description': '128d, 6 layers, 1.6M params'
        }
    ]
    
    results = {}
    
    for model_info in models_to_test:
        print(f"\n🤖 Testing {model_info['name']}")
        print(f"   Description: {model_info['description']}")
        print("-" * 50)
        
        # Create and load model
        model = create_enhanced_model(device)
        
        try:
            model = load_model_checkpoint(model, model_info['path'], device)
            
            # Evaluate on validation set
            val_results = evaluate_model(
                model, eegVal, stimVal, labelsVal, 
                clip_model, clip_preprocess, device, "validation"
            )
            
            # Evaluate on test set
            test_results = evaluate_model(
                model, eegTest, stimTest, labelsTest,
                clip_model, clip_preprocess, device, "test"
            )
            
            results[model_info['name']] = {
                'validation': val_results,
                'test': test_results,
                'description': model_info['description']
            }
            
        except Exception as e:
            print(f"❌ Error loading {model_info['name']}: {e}")
            continue
    
    # Create comparison visualization
    plot_comparison_results(results)
    
    return results

def plot_comparison_results(results):
    """Plot comparison results"""
    print("\n📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MBD2 Models Comparison - EEG-to-Image Reconstruction', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    model_names = list(results.keys())
    val_accuracies = [results[name]['validation']['accuracy'] for name in model_names]
    test_accuracies = [results[name]['test']['accuracy'] for name in model_names]
    val_similarities = [results[name]['validation']['mean_similarity'] for name in model_names]
    test_similarities = [results[name]['test']['mean_similarity'] for name in model_names]
    
    # Plot 1: Accuracy comparison
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x_pos - width/2, val_accuracies, width, label='Validation', alpha=0.8, color='skyblue')
    bars2 = axes[0, 0].bar(x_pos + width/2, test_accuracies, width, label='Test', alpha=0.8, color='lightcoral')
    
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Similarity comparison
    bars3 = axes[0, 1].bar(x_pos - width/2, val_similarities, width, label='Validation', alpha=0.8, color='lightgreen')
    bars4 = axes[0, 1].bar(x_pos + width/2, test_similarities, width, label='Test', alpha=0.8, color='orange')
    
    axes[0, 1].set_ylabel('Mean Cosine Similarity')
    axes[0, 1].set_title('Mean Similarity Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Similarity distributions
    for i, name in enumerate(model_names):
        test_sims = results[name]['test']['similarities']
        axes[1, 0].hist(test_sims, bins=30, alpha=0.6, label=name, density=True)
    
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Test Similarity Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Threshold')
    
    # Plot 4: Summary statistics
    summary_text = "MBD2 Models Summary:\n\n"
    
    for name in model_names:
        val_acc = results[name]['validation']['accuracy']
        test_acc = results[name]['test']['accuracy']
        desc = results[name]['description']
        
        summary_text += f"{name}:\n"
        summary_text += f"  {desc}\n"
        summary_text += f"  Val Acc: {val_acc:.3f} ({val_acc*100:.1f}%)\n"
        summary_text += f"  Test Acc: {test_acc:.3f} ({test_acc*100:.1f}%)\n\n"
    
    # Add baseline comparison
    random_baseline = 0.1  # 10% for 10 classes
    summary_text += f"Baselines:\n"
    summary_text += f"  Random: {random_baseline:.3f} (10.0%)\n"
    summary_text += f"  Positive Sim: 0.000 (0.0%)\n\n"
    
    # Add improvement analysis
    if len(model_names) >= 2:
        best_model = max(model_names, key=lambda x: results[x]['test']['accuracy'])
        best_acc = results[best_model]['test']['accuracy']
        improvement = (best_acc / random_baseline - 1) * 100
        
        summary_text += f"Best Model: {best_model}\n"
        summary_text += f"Improvement over Random: {improvement:.1f}%\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'mbd2_models_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison plot saved: {plot_filename}")
    
    return plot_filename

def main():
    """Main function"""
    print("🧠 MBD2 MODELS TESTING & COMPARISON")
    print("=" * 60)
    print("Testing both Enhanced and 400-Epoch models")
    print("Dataset: MindBigData EEG (digits 0-9)")
    print("Task: EEG-to-Image reconstruction via CLIP alignment")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run comparison
    results = compare_models()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🏆 TESTING COMPLETED!")
    print(f"   Duration: {duration/60:.1f} minutes")
    print(f"   Models tested: {len(results)}")
    
    # Print final summary
    if results:
        print(f"\n📊 FINAL SUMMARY:")
        for name, result in results.items():
            test_acc = result['test']['accuracy']
            print(f"   {name}: {test_acc:.3f} ({test_acc*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    main()
