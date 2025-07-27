#!/usr/bin/env python3
"""
Fixed training script with correct folder mapping
Run: python fixed_trainer.py
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2


def load_data():
    """Load data with correct folder mapping"""
    print("🔄 Loading data with corrected folder mapping...")

    # Updated folder mapping to match your actual folders
    folder_to_label = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        '10': 10,  # alif (ا)
        '11': 11,  # baa (ب)
        '12': 12,  # haa (ح)
        '13': 13,  # ttaa (ت)
        '15': 14,  # waw (و) - Note: folder 15 maps to label 14
        '16': 15  # dal (د) - Note: folder 16 maps to label 15
    }

    class_names = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'ا', 'ب', 'ح', 'ت', 'و', 'د'
    ]

    images = []
    labels = []

    data_dir = Path('data/synth')

    print("Checking folders...")
    for folder_name, label in folder_to_label.items():
        folder_path = data_dir / folder_name

        if not folder_path.exists():
            print(f"❌ Missing folder: {folder_name}")
            continue

        image_files = list(folder_path.glob('*.png')) + \
                      list(folder_path.glob('*.jpg')) + \
                      list(folder_path.glob('*.jpeg'))

        print(f"✅ {folder_name} -> {class_names[label]}: {len(image_files)} images")

        for img_file in image_files:
            try:
                # Load image
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Resize to 64x64
                img = cv2.resize(img, (64, 64))

                # Normalize to [0, 1]
                img = img.astype('float32') / 255.0

                # Check if image is not all zeros or ones
                if np.std(img) > 0.01:  # Has some variation
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"⚠️  Skipping flat image: {img_file}")

            except Exception as e:
                print(f"Error loading {img_file}: {e}")

    return np.array(images), np.array(labels), class_names


def create_better_model():
    """Create a simpler, more effective model"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(64, 64, 1)),

        # First block - simpler
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Third block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='softmax')  # 16 classes
    ])

    return model


def main():
    print("🚀 Fixed Moroccan LP Character Training")
    print("=" * 60)

    # Load data with correct mapping
    X, y, class_names = load_data()

    if len(X) == 0:
        print("❌ No images loaded! Check your data directory structure.")
        return

    print(f"\n📊 Dataset Summary:")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Classes: {len(set(y))}")
    print(f"Pixel range: {X.min():.3f} - {X.max():.3f}")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"  {class_names[class_idx]}: {count} images")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_val = X_val.reshape(-1, 64, 64, 1)

    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train, 16)
    y_val_cat = keras.utils.to_categorical(y_val, 16)

    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Create model
    model = create_better_model()

    # Compile with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nModel Summary:")
    model.summary()

    # Create output directory
    Path('models/classifier').mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/classifier/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print("\n🔥 Starting training...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=16,  # Smaller batch size
        epochs=50,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('models/classifier/moroccan_lp_classifier.keras')
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: models/classifier/moroccan_lp_classifier.keras")

    # Show results
    best_acc = max(history.history['val_accuracy'])
    print(f"🎯 Best validation accuracy: {best_acc:.3f}")

    if best_acc > 0.8:
        print("🎉 Great! Model is learning well!")
    elif best_acc > 0.5:
        print("🔄 Model is learning, but could improve")
    else:
        print("⚠️  Model struggling - check data quality")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model, history


if __name__ == "__main__":
    model, history = main()