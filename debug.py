#!/usr/bin/env python3
"""
Debug script to check your data
Run: python debug_data.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def debug_data():
    print("🔍 Debugging Data Loading...")
    print("=" * 50)
    
    # Check data directory
    data_dir = Path('data/synth')  # You used 'data/synth' not 'data/chars'
    
    if not data_dir.exists():
        print(f"❌ Directory {data_dir} doesn't exist!")
        return
    
    print(f"✅ Found directory: {data_dir}")
    
    # Expected folders
    expected_folders = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'alif', 'baa', 'haa', 'ttaa', 'waw', 'dal'
    ]
    
    total_images = 0
    class_counts = {}
    
    for folder in expected_folders:
        folder_path = data_dir / folder
        
        if not folder_path.exists():
            print(f"❌ Missing folder: {folder}")
            continue
        
        # Count images
        images = list(folder_path.glob('*.png')) + \
                list(folder_path.glob('*.jpg')) + \
                list(folder_path.glob('*.jpeg'))
        
        count = len(images)
        class_counts[folder] = count
        total_images += count
        
        print(f"📁 {folder}: {count} images")
        
        # Test loading first image
        if count > 0:
            test_img_path = images[0]
            img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"   ✅ Sample image loaded: {img.shape}, dtype: {img.dtype}")
                print(f"   📊 Pixel range: {img.min()}-{img.max()}")
            else:
                print(f"   ❌ Failed to load: {test_img_path}")
    
    print(f"\n📊 Total images: {total_images}")
    print(f"📊 Classes found: {len(class_counts)}")
    
    # Check for class imbalance
    if class_counts:
        counts = list(class_counts.values())
        print(f"📊 Min images per class: {min(counts)}")
        print(f"📊 Max images per class: {max(counts)}")
        
        if max(counts) - min(counts) > 100:
            print("⚠️  Warning: Significant class imbalance detected!")
    
    # Visual check - load and display sample images
    if total_images > 0:
        print("\n🖼️  Displaying sample images...")
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        axes = axes.flatten()
        
        for i, folder in enumerate(expected_folders):
            if i >= 16:
                break
                
            folder_path = data_dir / folder
            if folder_path.exists():
                images = list(folder_path.glob('*.png')) + \
                        list(folder_path.glob('*.jpg'))
                
                if len(images) > 0:
                    img = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(folder)
                        axes[i].axis('off')
                    else:
                        axes[i].text(0.5, 0.5, f'{folder}\nError', 
                                   ha='center', va='center')
                        axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f'{folder}\nEmpty', 
                               ha='center', va='center')
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'{folder}\nMissing', 
                           ha='center', va='center', color='red')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("💾 Sample images saved as 'data_samples.png'")

def test_model_basics():
    """Test if the model architecture is reasonable"""
    print("\n🧠 Testing Model Architecture...")
    print("=" * 50)
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Create a simple test model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("✅ Model compiled successfully")
    
    # Test with dummy data
    dummy_X = np.random.random((100, 64, 64, 1))
    dummy_y = keras.utils.to_categorical(np.random.randint(0, 16, 100), 16)
    
    print("🧪 Testing with dummy data...")
    history = model.fit(dummy_X, dummy_y, epochs=2, verbose=1, batch_size=16)
    
    if history.history['accuracy'][-1] > 0.1:
        print("✅ Model can learn (accuracy > 10% on dummy data)")
    else:
        print("❌ Model architecture might have issues")

if __name__ == "__main__":
    debug_data()
    test_model_basics()
