import tensorflow as tf
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Character mapping
CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'أ': 10, 'ب': 11, 'د': 12, 'ھ': 13, 'ط': 15, 'و': 16
}

IDX_TO_CHAR = {v: k for k, v in CHARACTERS.items()}

class CharacterClassifier:
    def __init__(self, char_size=(32, 32)):
        self.char_size = char_size
        self.model = None
        
    def load_dataset(self, dataset_dir):
        """Load and preprocess character images"""
        print("Loading character dataset...")
        images = []
        labels = []
        
        for char_idx in range(len(CHARACTERS)):
            char_dir = os.path.join(dataset_dir, str(char_idx))
            if not os.path.exists(char_dir):
                print(f"Warning: Directory {char_dir} not found")
                continue
                
            char_images = 0
            for img_file in os.listdir(char_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(char_dir, img_file)
                    
                    # Load image as grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Apply preprocessing
                    img = self._preprocess_image(img)
                    
                    images.append(img)
                    labels.append(char_idx)
                    char_images += 1
            
            print(f"Loaded {char_images} images for character '{IDX_TO_CHAR[char_idx]}'")
        
        return np.array(images), np.array(labels)
    
    def _preprocess_image(self, img):
        """Preprocess individual image"""
        # Resize
        resized = cv2.resize(img, self.char_size, interpolation=cv2.INTER_AREA)
        
        # Random preprocessing augmentation during training
        if random.random() < 0.3:
            # Apply adaptive thresholding
            resized = cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def build_model(self):
        """Build CNN model for character classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.char_size[0], self.char_size[1], 1)),
            
            # First conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Classifier
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(CHARACTERS), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, dataset_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Train the character classification model"""
        # Load data
        X, y = self.load_dataset(dataset_dir)
        
        if len(X) == 0:
            raise ValueError("No data found! Please generate synthetic data first.")
        
        print(f"Dataset loaded: {len(X)} images")
        
        # Reshape for CNN
        X = X.reshape(X.shape[0], self.char_size[0], self.char_size[1], 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Build model
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True, 
                monitor='val_accuracy', mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/classifier/best_char_classifier.h5',
                save_best_only=True, monitor='val_accuracy', mode='max'
            )
        ]
        
        # Create model directory
        os.makedirs('models/classifier', exist_ok=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFinal test accuracy: {test_accuracy:.4f}")
        
        # Generate detailed evaluation
        self._evaluate_model(X_test, y_test)
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def _evaluate_model(self, X_test, y_test):
        """Detailed model evaluation"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        char_names = [IDX_TO_CHAR[i] for i in range(len(CHARACTERS))]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=char_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=char_names, yticklabels=char_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def _plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='models/classifier/char_classifier.h5'):
        """Save the trained model"""
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/classifier/char_classifier.h5'):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    classifier = CharacterClassifier()
    
    # Train the model
    print("Training character classifier...")
    classifier.train('data/synth', epochs=30, batch_size=32)
    
    # Save the model
    classifier.save_model()
