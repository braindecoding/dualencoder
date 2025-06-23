#!/usr/bin/env python3
"""
Check for Real Labels in Digit69 Dataset
"""

import pickle
import numpy as np

def check_for_real_labels():
    """Check if there are real labels in the dataset"""
    print("🔍 Searching for real labels in Digit69 dataset...")
    
    try:
        with open('digit69_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Data loaded successfully!")
        
        # Check metadata for clues
        if 'metadata' in data:
            print(f"\n📋 Metadata:")
            for key, value in data['metadata'].items():
                print(f"   {key}: {value}")
        
        # Check if there are any label-like fields
        def search_for_labels(obj, path=""):
            """Recursively search for label-like fields"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if 'label' in key.lower() or 'class' in key.lower() or 'digit' in key.lower():
                        print(f"🎯 Found potential labels at: {current_path}")
                        print(f"   Type: {type(value)}")
                        if isinstance(value, np.ndarray):
                            print(f"   Shape: {value.shape}")
                            print(f"   Unique values: {np.unique(value)}")
                        elif isinstance(value, list):
                            print(f"   Length: {len(value)}")
                            if len(value) > 0:
                                print(f"   Sample values: {value[:10]}")
                    
                    search_for_labels(value, current_path)
        
        search_for_labels(data)
        
        # Check if images contain digit information
        print(f"\n🖼️ Analyzing images for digit patterns...")
        
        train_images = data['train']['original_images']
        test_images = data['test']['original_images']
        
        print(f"Train images shape: {train_images.shape}")
        print(f"Test images shape: {test_images.shape}")
        
        # Try to extract digit information from image analysis
        # This is a heuristic approach
        def analyze_image_for_digit(image):
            """Simple heuristic to guess digit from image"""
            # Convert from (3, 224, 224) to (224, 224) grayscale
            if len(image.shape) == 3 and image.shape[0] == 3:
                gray = np.mean(image, axis=0)
            else:
                gray = image
            
            # Simple features
            center_intensity = gray[112:112+10, 112:112+10].mean()
            edge_intensity = np.concatenate([
                gray[0:10, :].flatten(),
                gray[-10:, :].flatten(),
                gray[:, 0:10].flatten(),
                gray[:, -10:].flatten()
            ]).mean()
            
            # Very rough heuristics (not reliable!)
            if center_intensity > 0.7:
                return 0  # Might be 0 or 8
            elif edge_intensity < 0.3:
                return 1  # Might be 1
            else:
                return -1  # Unknown
        
        # Analyze first few images
        print(f"\n🔍 Heuristic analysis of first 10 training images:")
        for i in range(min(10, len(train_images))):
            guessed_digit = analyze_image_for_digit(train_images[i])
            print(f"   Image {i}: Guessed digit = {guessed_digit}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    data = check_for_real_labels()
