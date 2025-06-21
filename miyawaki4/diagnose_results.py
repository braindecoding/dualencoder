#!/usr/bin/env python3
"""
Diagnose Miyawaki Results
Analyze why generated images don't match original patterns
"""

import torch
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import warnings
warnings.filterwarnings("ignore")

class MiyawakiResultsDiagnostic:
    """Diagnostic tool for Miyawaki results analysis"""
    
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load embeddings and original images"""
        print("📊 Loading diagnostic data...")
        
        # Load embeddings
        with open("miyawaki4_embeddings.pkl", 'rb') as f:
            self.embeddings_data = pickle.load(f)
        
        print(f"✅ Loaded embeddings data")
        print(f"   Train samples: {len(self.embeddings_data['train']['fmri_embeddings'])}")
        print(f"   Test samples: {len(self.embeddings_data['test']['fmri_embeddings'])}")
    
    def analyze_original_patterns(self):
        """Analyze characteristics of original Miyawaki patterns"""
        print("\n🔍 ANALYZING ORIGINAL PATTERNS")
        print("=" * 50)

        # Check available keys
        print(f"Available keys in train data: {list(self.embeddings_data['train'].keys())}")
        print(f"Available keys in test data: {list(self.embeddings_data['test'].keys())}")

        # Get sample images from original_images (these are already numpy arrays)
        train_images = self.embeddings_data['train']['original_images'][:5]
        test_images = self.embeddings_data['test']['original_images'][:5]

        all_images = train_images + test_images

        # Convert to proper format (images are already numpy arrays)
        image_arrays = []
        for img_array in all_images:
            print(f"Original image shape: {img_array.shape}, dtype: {img_array.dtype}")

            # Handle different image formats
            if len(img_array.shape) == 1:  # Flattened image
                # Assume square image
                size = int(np.sqrt(img_array.shape[0]))
                img_array = img_array.reshape(size, size)

            # Ensure proper shape and type
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:  # Normalized to [0,1]
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
            elif len(img_array.shape) == 3 and img_array.shape[0] == 3:  # CHW format
                img_array = np.transpose(img_array, (1, 2, 0))  # Convert to HWC

            print(f"Processed image shape: {img_array.shape}, dtype: {img_array.dtype}")
            image_arrays.append(img_array)
        
        # Analyze characteristics
        characteristics = {
            'colors': [],
            'contrast': [],
            'binary_ratio': [],
            'edge_density': [],
            'pattern_type': []
        }
        
        for i, img_array in enumerate(image_arrays):
            # Color analysis
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            characteristics['colors'].append(unique_colors)
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Contrast analysis
            contrast = np.std(gray)
            characteristics['contrast'].append(contrast)
            
            # Binary ratio (how binary is the image)
            binary_threshold = 128
            binary_img = (gray > binary_threshold).astype(np.uint8)
            binary_ratio = np.sum(binary_img) / binary_img.size
            characteristics['binary_ratio'].append(binary_ratio)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            characteristics['edge_density'].append(edge_density)
            
            # Pattern type detection
            pattern_type = self.detect_pattern_type(binary_img)
            characteristics['pattern_type'].append(pattern_type)
        
        # Print analysis results
        print(f"📈 ORIGINAL PATTERN CHARACTERISTICS:")
        print(f"   Average unique colors: {np.mean(characteristics['colors']):.1f}")
        print(f"   Average contrast: {np.mean(characteristics['contrast']):.1f}")
        print(f"   Average binary ratio: {np.mean(characteristics['binary_ratio']):.3f}")
        print(f"   Average edge density: {np.mean(characteristics['edge_density']):.3f}")
        print(f"   Pattern types: {set(characteristics['pattern_type'])}")
        
        # Visualize original patterns
        self.visualize_original_patterns(image_arrays[:6])
        
        return characteristics
    
    def detect_pattern_type(self, binary_img):
        """Detect the type of geometric pattern"""
        h, w = binary_img.shape
        
        # Check for cross pattern
        center_row = binary_img[h//2, :]
        center_col = binary_img[:, w//2]
        
        if np.sum(center_row) > w * 0.3 and np.sum(center_col) > h * 0.3:
            return 'cross'
        
        # Check for L-shape
        corners = [
            binary_img[:h//2, :w//2],  # Top-left
            binary_img[:h//2, w//2:],  # Top-right
            binary_img[h//2:, :w//2],  # Bottom-left
            binary_img[h//2:, w//2:]   # Bottom-right
        ]
        
        corner_densities = [np.sum(corner) / corner.size for corner in corners]
        max_density = max(corner_densities)
        
        if max_density > 0.5:
            return 'L-shape'
        
        # Check for rectangle
        if np.sum(binary_img) / binary_img.size > 0.1:
            return 'rectangle'
        
        return 'unknown'
    
    def analyze_generated_results(self):
        """Analyze generated results and compare with originals"""
        print("\n🔍 ANALYZING GENERATED RESULTS")
        print("=" * 50)

        # Find generated images
        generated_files = list(Path('.').glob('miyawaki_sample_*.png'))

        if not generated_files:
            print("❌ No generated images found!")
            return None
        
        print(f"📁 Found {len(generated_files)} generated images")
        
        # Load and analyze generated images
        generated_characteristics = {
            'colors': [],
            'contrast': [],
            'binary_ratio': [],
            'edge_density': [],
            'pattern_type': []
        }
        
        generated_images = []
        for img_path in generated_files[:5]:  # Analyze first 5
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            generated_images.append(img_array)
            
            # Same analysis as original
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            generated_characteristics['colors'].append(unique_colors)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            generated_characteristics['contrast'].append(contrast)
            
            binary_img = (gray > 128).astype(np.uint8)
            binary_ratio = np.sum(binary_img) / binary_img.size
            generated_characteristics['binary_ratio'].append(binary_ratio)
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            generated_characteristics['edge_density'].append(edge_density)
            
            pattern_type = self.detect_pattern_type(binary_img)
            generated_characteristics['pattern_type'].append(pattern_type)
        
        print(f"📈 GENERATED IMAGE CHARACTERISTICS:")
        print(f"   Average unique colors: {np.mean(generated_characteristics['colors']):.1f}")
        print(f"   Average contrast: {np.mean(generated_characteristics['contrast']):.1f}")
        print(f"   Average binary ratio: {np.mean(generated_characteristics['binary_ratio']):.3f}")
        print(f"   Average edge density: {np.mean(generated_characteristics['edge_density']):.3f}")
        print(f"   Pattern types: {set(generated_characteristics['pattern_type'])}")
        
        # Visualize generated images
        self.visualize_generated_results(generated_images)
        
        return generated_characteristics
    
    def compare_characteristics(self, original_chars, generated_chars):
        """Compare original vs generated characteristics"""
        print("\n⚖️ COMPARISON ANALYSIS")
        print("=" * 50)
        
        metrics = ['colors', 'contrast', 'binary_ratio', 'edge_density']
        
        for metric in metrics:
            orig_avg = np.mean(original_chars[metric])
            gen_avg = np.mean(generated_chars[metric])
            difference = abs(orig_avg - gen_avg)
            
            print(f"📊 {metric.upper()}:")
            print(f"   Original: {orig_avg:.3f}")
            print(f"   Generated: {gen_avg:.3f}")
            print(f"   Difference: {difference:.3f}")
            print()
        
        # Pattern type comparison
        orig_patterns = set(original_chars['pattern_type'])
        gen_patterns = set(generated_chars['pattern_type'])
        
        print(f"🎯 PATTERN TYPES:")
        print(f"   Original: {orig_patterns}")
        print(f"   Generated: {gen_patterns}")
        print(f"   Match: {orig_patterns.intersection(gen_patterns)}")
        print(f"   Missing: {orig_patterns - gen_patterns}")
    
    def visualize_original_patterns(self, image_arrays):
        """Visualize original patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Original Miyawaki Patterns Analysis', fontsize=16, fontweight='bold')
        
        for i, img_array in enumerate(image_arrays):
            if i >= 6:
                break
                
            row = i // 3
            col = i % 3
            
            # Original image
            axes[row, col].imshow(img_array)
            axes[row, col].set_title(f'Original Pattern {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('original_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_generated_results(self, generated_images):
        """Visualize generated results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Generated Images Analysis', fontsize=16, fontweight='bold')
        
        for i, img_array in enumerate(generated_images):
            if i >= 6:
                break
                
            row = i // 3
            col = i % 3
            
            axes[row, col].imshow(img_array)
            axes[row, col].set_title(f'Generated Image {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('generated_images_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n📋 GENERATING DIAGNOSTIC REPORT")
        print("=" * 60)

        # Analyze original patterns
        original_chars = self.analyze_original_patterns()

        # Analyze generated results
        generated_chars = self.analyze_generated_results()

        # Compare characteristics only if we have generated results
        if generated_chars is not None:
            self.compare_characteristics(original_chars, generated_chars)
            self.generate_recommendations(original_chars, generated_chars)
        else:
            print("\n❌ NO GENERATED IMAGES TO COMPARE")
            print("=" * 50)
            print("🔧 NEXT STEPS:")
            print("1. Run the fine-tuning script first: python finetune_ldm_miyawaki.py")
            print("2. Or run intensive training: python intensive_miyawaki_training.py")
            print("3. Or try direct pattern generation: python direct_pattern_generator.py")

            # Still generate recommendations based on original analysis
            self.analyze_original_patterns_detailed(original_chars)

    def analyze_original_patterns_detailed(self, original_chars):
        """Detailed analysis of original patterns for recommendations"""
        print("\n🔍 DETAILED ORIGINAL PATTERN ANALYSIS")
        print("=" * 50)

        avg_colors = np.mean(original_chars['colors'])
        avg_contrast = np.mean(original_chars['contrast'])
        avg_binary = np.mean(original_chars['binary_ratio'])
        avg_edges = np.mean(original_chars['edge_density'])

        print(f"📊 ORIGINAL MIYAWAKI CHARACTERISTICS:")
        print(f"   Colors: {avg_colors:.1f} (should be ~2-3 for binary patterns)")
        print(f"   Contrast: {avg_contrast:.3f} (high contrast expected)")
        print(f"   Binary ratio: {avg_binary:.3f} (should be ~0.2-0.8)")
        print(f"   Edge density: {avg_edges:.3f} (geometric patterns have clear edges)")

        print(f"\n💡 TRAINING RECOMMENDATIONS:")
        print(f"1. 🎯 Focus on BINARY patterns (black/white)")
        print(f"2. 📐 Emphasize GEOMETRIC shapes (cross, L, rectangle)")
        print(f"3. 🔲 Use HIGH CONTRAST loss functions")
        print(f"4. ⚫ Add EDGE PRESERVATION constraints")
        print(f"5. 🎨 Minimize color complexity (binary classification)")

        if avg_binary < 0.1:
            print(f"\n⚠️ WARNING: Original patterns show very low binary ratio!")
            print(f"   This suggests the images might not be properly loaded.")
            print(f"   Check image preprocessing and normalization.")
    
    def generate_recommendations(self, original_chars, generated_chars):
        """Generate recommendations for improvement"""
        print("\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("=" * 60)
        
        recommendations = []
        
        # Color analysis
        orig_colors = np.mean(original_chars['colors'])
        gen_colors = np.mean(generated_chars['colors'])
        
        if gen_colors > orig_colors * 2:
            recommendations.append("🎨 REDUCE COLOR COMPLEXITY: Generated images have too many colors. Original Miyawaki patterns are mostly binary (black/white).")
        
        # Binary ratio analysis
        orig_binary = np.mean(original_chars['binary_ratio'])
        gen_binary = np.mean(generated_chars['binary_ratio'])
        
        if abs(orig_binary - gen_binary) > 0.2:
            recommendations.append("⚫ IMPROVE BINARY PATTERNS: Generated images don't match the binary nature of original patterns.")
        
        # Pattern type analysis
        orig_patterns = set(original_chars['pattern_type'])
        gen_patterns = set(generated_chars['pattern_type'])
        
        if not orig_patterns.intersection(gen_patterns):
            recommendations.append("🎯 FOCUS ON GEOMETRIC PATTERNS: Generated images don't contain the geometric patterns (cross, L-shape) found in originals.")
        
        # Contrast analysis
        orig_contrast = np.mean(original_chars['contrast'])
        gen_contrast = np.mean(generated_chars['contrast'])
        
        if gen_contrast < orig_contrast * 0.5:
            recommendations.append("📈 INCREASE CONTRAST: Generated images have lower contrast than original high-contrast patterns.")
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        if not recommendations:
            print("✅ Generated images match original characteristics well!")
        
        print(f"\n🔧 TECHNICAL SOLUTIONS:")
        print(f"1. Use binary loss functions (BCE) instead of MSE")
        print(f"2. Add geometric pattern constraints")
        print(f"3. Increase training epochs with pattern-specific loss")
        print(f"4. Use direct pattern generation instead of diffusion")
        print(f"5. Add edge preservation loss")

def run_diagnosis():
    """Run complete diagnostic analysis"""
    print("🔍 MIYAWAKI RESULTS DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    diagnostic = MiyawakiResultsDiagnostic()
    diagnostic.generate_diagnostic_report()
    
    print(f"\n📁 Generated diagnostic files:")
    print(f"   - original_patterns_analysis.png")
    print(f"   - generated_images_analysis.png")

if __name__ == "__main__":
    run_diagnosis()
