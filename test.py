#!/usr/bin/env python3
"""
Test your trained model on individual characters
Run: python test_inference.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

class MoroccanLPInference:
    def __init__(self, model_path='models/classifier/moroccan_lp_classifier.keras'):
        self.model = tf.keras.models.load_model(model_path)
        
        # Class names matching your training
        self.class_names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'ا', 'ب', 'ح', 'ت', 'و', 'د'
        ]
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {len(self.class_names)}")
    
    def predict_character(self, image_path):
        """Predict a single character image"""
        # Load and preprocess image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize to match training size
        img = cv2.resize(img, (64, 64))
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Reshape for model
        img = img.reshape(1, 64, 64, 1)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'character': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: predictions[0][i] 
                for i in range(len(self.class_names))
            }
        }
    
    def test_random_samples(self, data_dir='data/synth', samples_per_class=3):
        """Test on random samples from your dataset"""
        data_path = Path(data_dir)
        
        # Folder mapping
        folders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  '10', '11', '12', '13', '15', '16']
        expected_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'ا', 'ب', 'ح', 'ت', 'و', 'د']
        
        fig, axes = plt.subplots(len(folders), samples_per_class, 
                                figsize=(samples_per_class * 2, len(folders) * 1.5))
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (folder, expected_char) in enumerate(zip(folders, expected_chars)):
            folder_path = data_path / folder
            
            if not folder_path.exists():
                continue
            
            # Get random sample images
            image_files = list(folder_path.glob('*.png')) + \
                         list(folder_path.glob('*.jpg'))
            
            if len(image_files) == 0:
                continue
            
            sample_files = random.sample(image_files, 
                                       min(samples_per_class, len(image_files)))
            
            for j, img_file in enumerate(sample_files):
                # Load image for display
                img_display = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                # Predict
                result = self.predict_character(img_file)
                
                if result:
                    predicted_char = result['character']
                    confidence = result['confidence']
                    
                    # Check if correct
                    is_correct = predicted_char == expected_char
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Display
                    axes[i, j].imshow(img_display, cmap='gray')
                    axes[i, j].set_title(
                        f"True: {expected_char}\n"
                        f"Pred: {predicted_char}\n"
                        f"Conf: {confidence:.2f}\n"
                        f"{'✅' if is_correct else '❌'}",
                        fontsize=8
                    )
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[i, j].axis('off')
            
            # Fill remaining slots
            for j in range(len(sample_files), samples_per_class):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Show accuracy
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"\n🎯 Test Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        return accuracy

def test_single_image():
    """Test on a single image - modify path as needed"""
    classifier = MoroccanLPInference()
    
    # Test on a random image from your dataset
    data_dir = Path('data/synth')
    
    # Find first available image
    for folder in ['0', '1', '2', '10', '11']:
        folder_path = data_dir / folder
        if folder_path.exists():
            images = list(folder_path.glob('*.png'))
            if len(images) > 0:
                test_image = images[0]
                print(f"\n🧪 Testing image: {test_image}")
                
                result = classifier.predict_character(test_image)
                if result:
                    print(f"🎯 Predicted: {result['character']}")
                    print(f"📊 Confidence: {result['confidence']:.3f}")
                    
                    # Show top 3 predictions
                    sorted_probs = sorted(result['all_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    print(f"\nTop 3 predictions:")
                    for char, prob in sorted_probs[:3]:
                        print(f"  {char}: {prob:.3f}")
                break

if __name__ == "__main__":
    print("🧪 Testing Moroccan LP Character Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = MoroccanLPInference()
    
    # Test on random samples
    print("\n📊 Testing on random samples from dataset...")
    accuracy = classifier.test_random_samples(samples_per_class=2)
    
    # Test single image
    test_single_image()
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Predictions saved as 'test_predictions.png'")
