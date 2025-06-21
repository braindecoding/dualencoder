#!/usr/bin/env python3
"""
Analyze Miyawaki Dataset Image Characteristics
Understand what makes Miyawaki images unique for better LDM training
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

def load_miyawaki_images():
    """Load Miyawaki images for analysis"""
    print("📥 Loading Miyawaki images...")
    
    embeddings_path = "miyawaki4_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found")
        return None
    
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Combine train and test images
    train_images = embeddings_data['train']['original_images']
    test_images = embeddings_data['test']['original_images']
    
    all_images = np.concatenate([train_images, test_images], axis=0)
    
    print(f"✅ Loaded {len(all_images)} Miyawaki images")
    print(f"   📊 Image shape: {all_images.shape}")
    
    return all_images, embeddings_data

def analyze_color_distribution(images):
    """Analyze color distribution in Miyawaki images"""
    print("\n🎨 Analyzing color distribution...")
    
    # Convert to RGB format for analysis
    rgb_images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    
    # Flatten all pixels
    all_pixels = rgb_images.reshape(-1, 3)
    
    # Color statistics
    mean_colors = np.mean(all_pixels, axis=0)
    std_colors = np.std(all_pixels, axis=0)
    
    print(f"📊 Color Statistics:")
    print(f"   Mean RGB: [{mean_colors[0]:.3f}, {mean_colors[1]:.3f}, {mean_colors[2]:.3f}]")
    print(f"   Std RGB:  [{std_colors[0]:.3f}, {std_colors[1]:.3f}, {std_colors[2]:.3f}]")
    
    # Color clustering
    print("🔍 Performing color clustering...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    
    # Sample pixels for clustering (too many otherwise)
    sample_pixels = all_pixels[::100]  # Every 100th pixel
    kmeans.fit(sample_pixels)
    
    dominant_colors = kmeans.cluster_centers_
    
    return {
        'mean_colors': mean_colors,
        'std_colors': std_colors,
        'dominant_colors': dominant_colors,
        'all_pixels': all_pixels
    }

def analyze_texture_features(images):
    """Analyze texture features in Miyawaki images"""
    print("\n🔍 Analyzing texture features...")
    
    texture_features = []
    
    for i, img in enumerate(images[:20]):  # Analyze first 20 images
        # Convert to grayscale
        gray = np.mean(img, axis=0)  # Average across RGB channels
        gray = (gray * 255).astype(np.uint8)
        
        # Compute texture features
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Contrast (standard deviation)
        contrast = np.std(gray)
        
        # 3. Brightness
        brightness = np.mean(gray)
        
        # 4. Local variance
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = np.mean((gray.astype(np.float32) - local_mean) ** 2)
        
        texture_features.append({
            'edge_density': edge_density,
            'contrast': contrast,
            'brightness': brightness,
            'local_variance': local_var
        })
    
    # Compute statistics
    edge_densities = [f['edge_density'] for f in texture_features]
    contrasts = [f['contrast'] for f in texture_features]
    brightnesses = [f['brightness'] for f in texture_features]
    local_variances = [f['local_variance'] for f in texture_features]
    
    print(f"📊 Texture Statistics:")
    print(f"   Edge density: {np.mean(edge_densities):.4f} ± {np.std(edge_densities):.4f}")
    print(f"   Contrast: {np.mean(contrasts):.2f} ± {np.std(contrasts):.2f}")
    print(f"   Brightness: {np.mean(brightnesses):.2f} ± {np.std(brightnesses):.2f}")
    print(f"   Local variance: {np.mean(local_variances):.2f} ± {np.std(local_variances):.2f}")
    
    return {
        'edge_densities': edge_densities,
        'contrasts': contrasts,
        'brightnesses': brightnesses,
        'local_variances': local_variances
    }

def analyze_content_categories(images):
    """Analyze content categories in Miyawaki images"""
    print("\n📋 Analyzing content categories...")
    
    # Simple heuristic-based categorization
    categories = {
        'natural_scenes': 0,
        'objects': 0,
        'patterns': 0,
        'high_contrast': 0,
        'low_contrast': 0
    }
    
    for img in images:
        # Convert to RGB
        rgb_img = img.transpose(1, 2, 0)  # CHW -> HWC
        
        # Compute image statistics
        brightness = np.mean(rgb_img)
        contrast = np.std(rgb_img)
        edge_density = np.sum(cv2.Canny((rgb_img * 255).astype(np.uint8), 50, 150) > 0) / rgb_img.size
        
        # Simple categorization heuristics
        if contrast > 0.15:
            categories['high_contrast'] += 1
        else:
            categories['low_contrast'] += 1
        
        if edge_density > 0.1:
            categories['natural_scenes'] += 1
        elif edge_density > 0.05:
            categories['objects'] += 1
        else:
            categories['patterns'] += 1
    
    print(f"📊 Content Categories:")
    for category, count in categories.items():
        percentage = count / len(images) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    return categories

def create_comprehensive_visualization(images, color_analysis, texture_analysis, categories):
    """Create comprehensive visualization of Miyawaki dataset characteristics"""
    print("\n📊 Creating comprehensive visualization...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Miyawaki Dataset Characteristics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sample images
    for i in range(8):
        row, col = i // 4, i % 4
        if i < len(images):
            img = images[i].transpose(1, 2, 0)  # CHW -> HWC
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
    
    # 2. Color distribution
    ax = axes[2, 0]
    colors = ['Red', 'Green', 'Blue']
    means = color_analysis['mean_colors']
    stds = color_analysis['std_colors']
    
    x = np.arange(len(colors))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['red', 'green', 'blue'])
    ax.set_xlabel('Color Channel')
    ax.set_ylabel('Mean Value')
    ax.set_title('Color Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(colors)
    ax.grid(True, alpha=0.3)
    
    # 3. Dominant colors
    ax = axes[2, 1]
    dominant_colors = color_analysis['dominant_colors']
    color_patches = []
    for i, color in enumerate(dominant_colors):
        color_normalized = np.clip(color, 0, 1)
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color_normalized)
        ax.add_patch(rect)
    
    ax.set_xlim(0, len(dominant_colors))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Dominant Color Index')
    ax.set_title('Dominant Colors')
    ax.set_xticks(range(len(dominant_colors)))
    
    # 4. Texture features
    ax = axes[2, 2]
    texture_metrics = ['Edge Density', 'Contrast', 'Brightness', 'Local Var']
    texture_values = [
        np.mean(texture_analysis['edge_densities']),
        np.mean(texture_analysis['contrasts']) / 100,  # Normalize
        np.mean(texture_analysis['brightnesses']) / 255,  # Normalize
        np.mean(texture_analysis['local_variances']) / 1000  # Normalize
    ]
    
    bars = ax.bar(texture_metrics, texture_values, alpha=0.7, color='skyblue')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Texture Features')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 5. Content categories
    ax = axes[2, 3]
    category_names = list(categories.keys())
    category_counts = list(categories.values())
    
    wedges, texts, autotexts = ax.pie(category_counts, labels=category_names, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 8})
    ax.set_title('Content Categories')
    
    plt.tight_layout()
    plt.savefig('miyawaki_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Analysis saved as 'miyawaki_dataset_analysis.png'")

def generate_training_recommendations(color_analysis, texture_analysis, categories):
    """Generate recommendations for LDM training"""
    print("\n💡 TRAINING RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    # Color analysis recommendations
    mean_brightness = np.mean(color_analysis['mean_colors'])
    if mean_brightness < 0.3:
        recommendations.append("🔧 Images are relatively dark - consider brightness augmentation")
    elif mean_brightness > 0.7:
        recommendations.append("🔧 Images are relatively bright - consider dimming augmentation")
    
    # Texture analysis recommendations
    avg_contrast = np.mean(texture_analysis['contrasts'])
    if avg_contrast < 30:
        recommendations.append("🔧 Low contrast images - consider contrast enhancement")
    elif avg_contrast > 80:
        recommendations.append("🔧 High contrast images - consider contrast normalization")
    
    # Content recommendations
    total_images = sum(categories.values())
    if categories['natural_scenes'] / total_images > 0.6:
        recommendations.append("🌿 Majority natural scenes - fine-tune on landscape/nature datasets")
    
    if categories['high_contrast'] / total_images > 0.7:
        recommendations.append("⚡ High contrast content - use edge-preserving loss functions")
    
    # General recommendations
    recommendations.extend([
        "🎯 Use perceptual loss (VGG) for better visual quality",
        "🔄 Apply data augmentation: rotation, flip, color jitter",
        "📊 Monitor CLIP similarity during training",
        "⏱️ Use progressive training: start with low resolution",
        "🎨 Consider style transfer loss for Miyawaki-specific features"
    ])
    
    print("📋 Specific Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return recommendations

def main():
    """Main analysis function"""
    print("🔍 MIYAWAKI DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    
    # Load images
    images, embeddings_data = load_miyawaki_images()
    if images is None:
        return
    
    # Perform analyses
    color_analysis = analyze_color_distribution(images)
    texture_analysis = analyze_texture_features(images)
    categories = analyze_content_categories(images)
    
    # Create visualization
    create_comprehensive_visualization(images, color_analysis, texture_analysis, categories)
    
    # Generate recommendations
    recommendations = generate_training_recommendations(color_analysis, texture_analysis, categories)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"📊 Dataset Size: {len(images)} images")
    print(f"🎨 Color Profile: Mean RGB {color_analysis['mean_colors']}")
    print(f"🔍 Texture Profile: Avg contrast {np.mean(texture_analysis['contrasts']):.1f}")
    print(f"📋 Content: {max(categories, key=categories.get)} dominant")
    
    print(f"\n🎯 Key Insights:")
    print(f"   - Miyawaki images have specific visual characteristics")
    print(f"   - Pre-trained Stable Diffusion may not capture these patterns")
    print(f"   - Fine-tuning on Miyawaki dataset is ESSENTIAL")
    print(f"   - Custom loss functions may improve results")
    
    print(f"\n📁 Generated Files:")
    print(f"   - miyawaki_dataset_analysis.png")

if __name__ == "__main__":
    main()
