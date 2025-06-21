#!/usr/bin/env python3
"""
Data Quality Analysis for Digit69 Dataset
Comprehensive analysis of fMRI embeddings and image data quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path

class DataQualityAnalyzer:
    """Comprehensive data quality analysis"""
    
    def __init__(self, embeddings_file="digit69_embeddings.pkl"):
        self.embeddings_file = embeddings_file
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load embeddings data"""
        print("🔍 LOADING DATA FOR QUALITY ANALYSIS")
        print("=" * 50)
        
        with open(self.embeddings_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"✅ Data loaded successfully")
        print(f"   Splits available: {list(self.data.keys())}")
        
        for split in ['train', 'test']:  # Skip metadata
            if split in self.data:
                fmri_shape = self.data[split]['fmri_embeddings'].shape
                img_shape = self.data[split]['original_images'].shape
                print(f"   {split}: fMRI {fmri_shape}, Images {img_shape}")
    
    def analyze_fmri_embeddings(self):
        """Analyze fMRI embedding quality"""
        print(f"\n🧠 FMRI EMBEDDINGS ANALYSIS")
        print("=" * 40)
        
        # Combine all embeddings
        all_fmri = []
        all_labels = []
        
        for split in ['train', 'test']:
            embeddings = self.data[split]['fmri_embeddings']
            all_fmri.append(embeddings)
            all_labels.extend([split] * len(embeddings))
        
        all_fmri = np.vstack(all_fmri)
        
        print(f"📊 fMRI Embeddings Statistics:")
        print(f"   Shape: {all_fmri.shape}")
        print(f"   Mean: {all_fmri.mean():.4f}")
        print(f"   Std: {all_fmri.std():.4f}")
        print(f"   Min: {all_fmri.min():.4f}")
        print(f"   Max: {all_fmri.max():.4f}")
        print(f"   NaN count: {np.isnan(all_fmri).sum()}")
        print(f"   Inf count: {np.isinf(all_fmri).sum()}")
        
        # Check for zero embeddings
        zero_embeddings = np.all(all_fmri == 0, axis=1).sum()
        print(f"   Zero embeddings: {zero_embeddings}")
        
        # Check embedding diversity
        unique_embeddings = len(np.unique(all_fmri, axis=0))
        print(f"   Unique embeddings: {unique_embeddings}/{len(all_fmri)}")
        
        # Cosine similarity analysis
        similarity_matrix = cosine_similarity(all_fmri)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        print(f"   Average cosine similarity: {avg_similarity:.4f}")
        
        return all_fmri, all_labels
    
    def analyze_images(self):
        """Analyze image quality"""
        print(f"\n🖼️ IMAGE ANALYSIS")
        print("=" * 30)
        
        # Combine all images
        all_images = []
        all_labels = []
        
        for split in ['train', 'test']:
            images = self.data[split]['original_images']
            all_images.append(images)
            all_labels.extend([split] * len(images))
        
        all_images = np.vstack(all_images)
        
        print(f"📊 Image Statistics:")
        print(f"   Shape: {all_images.shape}")
        print(f"   Mean: {all_images.mean():.4f}")
        print(f"   Std: {all_images.std():.4f}")
        print(f"   Min: {all_images.min():.4f}")
        print(f"   Max: {all_images.max():.4f}")
        
        # Check image diversity
        unique_images = len(np.unique(all_images.reshape(len(all_images), -1), axis=0))
        print(f"   Unique images: {unique_images}/{len(all_images)}")
        
        # Check for blank images
        blank_images = np.all(all_images.reshape(len(all_images), -1) == 0, axis=1).sum()
        print(f"   Blank images: {blank_images}")
        
        return all_images, all_labels
    
    def visualize_embeddings(self, embeddings, labels):
        """Visualize fMRI embeddings using dimensionality reduction"""
        print(f"\n📈 VISUALIZING EMBEDDINGS")
        print("=" * 35)
        
        # PCA Analysis
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        
        print(f"📊 PCA Analysis:")
        print(f"   Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
        print(f"   Cumulative variance (first 10): {np.cumsum(pca.explained_variance_ratio_[:10])}")
        
        # t-SNE for visualization
        print("🔄 Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_tsne = tsne.fit_transform(embeddings_pca[:, :50])  # Use first 50 PCA components
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('fMRI Embeddings Quality Analysis', fontsize=16)
        
        # PCA explained variance
        axes[0, 0].plot(range(1, 21), pca.explained_variance_ratio_[:20], 'bo-')
        axes[0, 0].set_title('PCA Explained Variance Ratio')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].grid(True)
        
        # Cumulative explained variance
        axes[0, 1].plot(range(1, 21), np.cumsum(pca.explained_variance_ratio_[:20]), 'ro-')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].grid(True)
        
        # t-SNE visualization
        colors = ['blue' if label == 'train' else 'red' for label in labels]
        axes[1, 0].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors, alpha=0.7)
        axes[1, 0].set_title('t-SNE Visualization (Blue=Train, Red=Test)')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        
        # Embedding distribution
        axes[1, 1].hist(embeddings.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('fMRI Embedding Value Distribution')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('fmri_embeddings_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return embeddings_pca, embeddings_tsne
    
    def visualize_images(self, images, labels):
        """Visualize image samples and statistics"""
        print(f"\n🖼️ VISUALIZING IMAGES")
        print("=" * 30)
        
        # Sample images from each split
        train_indices = [i for i, label in enumerate(labels) if label == 'train']
        test_indices = [i for i, label in enumerate(labels) if label == 'test']
        
        fig, axes = plt.subplots(3, 8, figsize=(16, 9))
        fig.suptitle('Image Quality Analysis', fontsize=16)
        
        # Show train samples
        for i in range(8):
            if i < len(train_indices):
                idx = train_indices[i]
                img = images[idx]
                if len(img.shape) == 3:
                    img = img[0]  # Take first channel
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(f'Train {i+1}')
                axes[0, i].axis('off')
        
        # Show test samples
        for i in range(8):
            if i < len(test_indices):
                idx = test_indices[i]
                img = images[idx]
                if len(img.shape) == 3:
                    img = img[0]  # Take first channel
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title(f'Test {i+1}')
                axes[1, i].axis('off')
        
        # Show image statistics
        for i in range(8):
            if i < len(images):
                img = images[i]
                if len(img.shape) == 3:
                    img = img[0]
                
                # Create histogram
                axes[2, i].hist(img.flatten(), bins=20, alpha=0.7)
                axes[2, i].set_title(f'Hist {i+1}')
                axes[2, i].set_xlabel('Pixel Value')
                axes[2, i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('image_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_fmri_image_correlation(self, embeddings, images):
        """Analyze correlation between fMRI embeddings and images"""
        print(f"\n🔗 FMRI-IMAGE CORRELATION ANALYSIS")
        print("=" * 45)

        # Since dimensions don't match, we'll analyze other relationships
        print(f"📊 Dimension Analysis:")
        print(f"   fMRI embeddings: {embeddings.shape}")
        print(f"   Images: {images.shape}")

        # Convert images to grayscale and resize for analysis
        if len(images.shape) == 4 and images.shape[1] == 3:
            # Convert RGB to grayscale
            images_gray = np.mean(images, axis=1)  # (N, 224, 224)
        else:
            images_gray = images

        # Resize to smaller dimension for correlation analysis
        from scipy.ndimage import zoom
        target_size = 32  # Resize to 32x32
        scale_factor = target_size / images_gray.shape[-1]

        images_resized = []
        for img in images_gray:
            img_resized = zoom(img, scale_factor)
            images_resized.append(img_resized.flatten())

        images_resized = np.array(images_resized)  # (N, 1024)

        print(f"   Resized images for correlation: {images_resized.shape}")

        # Now we can't directly correlate different dimensions, so let's analyze other metrics
        # 1. Embedding similarity vs image similarity
        from sklearn.metrics.pairwise import cosine_similarity

        fmri_sim = cosine_similarity(embeddings)
        img_sim = cosine_similarity(images_resized)

        # Get upper triangular indices (excluding diagonal)
        triu_indices = np.triu_indices_from(fmri_sim, k=1)
        fmri_sim_flat = fmri_sim[triu_indices]
        img_sim_flat = img_sim[triu_indices]

        # Correlation between similarity matrices
        sim_correlation = np.corrcoef(fmri_sim_flat, img_sim_flat)[0, 1]

        correlations = np.array([sim_correlation])  # Single correlation value
        
        print(f"📊 fMRI-Image Similarity Correlation:")
        print(f"   Similarity matrix correlation: {sim_correlation:.4f}")
        print(f"   This measures how well fMRI similarity matches image similarity")

        # Additional statistics
        print(f"📊 Additional Statistics:")
        print(f"   fMRI similarity range: [{fmri_sim_flat.min():.4f}, {fmri_sim_flat.max():.4f}]")
        print(f"   Image similarity range: [{img_sim_flat.min():.4f}, {img_sim_flat.max():.4f}]")
        
        # Visualize similarity correlation
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(fmri_sim_flat, img_sim_flat, alpha=0.6)
        plt.xlabel('fMRI Similarity')
        plt.ylabel('Image Similarity')
        plt.title(f'fMRI vs Image Similarity\nCorrelation: {sim_correlation:.4f}')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist([fmri_sim_flat, img_sim_flat], bins=20, alpha=0.7,
                label=['fMRI Similarity', 'Image Similarity'])
        plt.xlabel('Similarity Value')
        plt.ylabel('Frequency')
        plt.title('Similarity Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('fmri_image_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'similarity_correlation': sim_correlation,
            'fmri_sim_range': [fmri_sim_flat.min(), fmri_sim_flat.max()],
            'img_sim_range': [img_sim_flat.min(), img_sim_flat.max()],
            'correlations': correlations
        }
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print(f"\n📋 GENERATING QUALITY REPORT")
        print("=" * 40)
        
        # Analyze all components
        fmri_embeddings, fmri_labels = self.analyze_fmri_embeddings()
        images, img_labels = self.analyze_images()
        
        # Visualizations
        self.visualize_embeddings(fmri_embeddings, fmri_labels)
        self.visualize_images(images, img_labels)
        
        # Correlation analysis
        correlation_results = self.analyze_fmri_image_correlation(fmri_embeddings, images)
        
        # Generate summary report
        report = {
            'fmri_stats': {
                'shape': fmri_embeddings.shape,
                'mean': float(fmri_embeddings.mean()),
                'std': float(fmri_embeddings.std()),
                'min': float(fmri_embeddings.min()),
                'max': float(fmri_embeddings.max()),
                'nan_count': int(np.isnan(fmri_embeddings).sum()),
                'zero_embeddings': int(np.all(fmri_embeddings == 0, axis=1).sum()),
                'unique_embeddings': int(len(np.unique(fmri_embeddings, axis=0)))
            },
            'image_stats': {
                'shape': images.shape,
                'mean': float(images.mean()),
                'std': float(images.std()),
                'min': float(images.min()),
                'max': float(images.max()),
                'unique_images': int(len(np.unique(images.reshape(len(images), -1), axis=0))),
                'blank_images': int(np.all(images.reshape(len(images), -1) == 0, axis=1).sum())
            },
            'correlation_stats': {
                'similarity_correlation': float(correlation_results['similarity_correlation']),
                'fmri_sim_range': [float(x) for x in correlation_results['fmri_sim_range']],
                'img_sim_range': [float(x) for x in correlation_results['img_sim_range']],
                'valid_count': len(correlation_results['correlations'])
            }
        }
        
        # Save report
        with open('data_quality_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        print(f"✅ Quality report saved: data_quality_report.pkl")
        return report

def main():
    """Main analysis function"""
    print("🔍 DIGIT69 DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    analyzer = DataQualityAnalyzer("digit69_embeddings.pkl")
    report = analyzer.generate_quality_report()
    
    print(f"\n🎯 QUALITY ASSESSMENT SUMMARY:")
    print(f"   fMRI Embeddings: {report['fmri_stats']['unique_embeddings']} unique out of {report['fmri_stats']['shape'][0]}")
    print(f"   Images: {report['image_stats']['unique_images']} unique out of {report['image_stats']['shape'][0]}")
    print(f"   fMRI-Image Similarity Correlation: {report['correlation_stats']['similarity_correlation']:.4f}")
    
    # Quality assessment
    if report['correlation_stats']['similarity_correlation'] > 0.1:
        print(f"✅ Good fMRI-image similarity correlation detected")
    else:
        print(f"⚠️ Weak fMRI-image similarity correlation - potential data quality issue")
    
    if report['fmri_stats']['unique_embeddings'] == report['fmri_stats']['shape'][0]:
        print(f"✅ All fMRI embeddings are unique")
    else:
        print(f"⚠️ Duplicate fMRI embeddings detected")
    
    print(f"\n📁 Generated files:")
    print(f"   - fmri_embeddings_analysis.png")
    print(f"   - image_quality_analysis.png") 
    print(f"   - fmri_image_correlations.png")
    print(f"   - data_quality_report.pkl")

if __name__ == "__main__":
    main()
