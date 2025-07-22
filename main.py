import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
import random
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Character mapping for Moroccan license plates (16 classes)
CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'أ': 10,  # alif
    'ب': 11,  # baa
    'د': 12,  # dal
    'ھ': 13,  # haa
    'ط': 15,  # taa 
    'و': 16  # waw
}

# Reverse mapping
IDX_TO_CHAR = {v: k for k, v in CHARACTERS.items()}


class CharacterOCRSystem:
    def __init__(self):
        self.char_detector = None  # YOLOv8 model for character detection
        self.char_classifier = None  # CNN model for character classification
        self.char_size = (32, 32)  # Size for character images
        # Plate-like background colors
        self.background_colors = [
            (255, 255, 255),  # White
            (220, 220, 220),  # Light gray
            (255, 200, 150),  # Light orange
            (200, 255, 200)  # Light green
        ]

    def generate_synthetic_characters(self, output_dir, num_per_char=500):
        """Generate enhanced synthetic character images for training"""
        print("Generating synthetic character dataset...")

        os.makedirs(output_dir, exist_ok=True)

        # Try to load different fonts (system dependent)
        font_paths = [
            "fonts/ARIAL.TTF",
            "fonts/KFGQPC Uthman Taha Naskh Regular.ttf",
            "fonts/tahoma.ttf"
        ]

        fonts = []
        for font_path in font_paths:
            try:
                fonts.append(ImageFont.truetype(font_path, size=24))
                print(f"Loaded font: {font_path}")
            except Exception as e:
                print(f"Couldn't load font {font_path}: {str(e)}")
                continue

        # Fallback to default font if no system fonts found
        if not fonts:
            try:
                # Try to use a basic Arabic-capable font
                fonts.append(ImageFont.truetype("arial.ttf", size=24))
            except:
                print("Warning: Using default font - Arabic may not render properly")
                fonts = [ImageFont.load_default()]

        # Plate-like background and text colors
        plate_backgrounds = [
            (255, 255, 255),  # White
            (220, 220, 220),  # Light gray
            (255, 200, 150),  # Light orange
            (200, 255, 200)  # Light green
        ]
        text_colors = [(0, 0, 0), (30, 30, 30), (50, 50, 50)]

        for char, char_idx in CHARACTERS.items():
            char_dir = os.path.join(output_dir, str(char_idx))
            os.makedirs(char_dir, exist_ok=True)

            for i in range(num_per_char):
                # Create image with plate-like background
                img = Image.new('RGB', (48, 48), random.choice(plate_backgrounds))
                draw = ImageDraw.Draw(img)

                # Select random font and color
                font = random.choice(fonts)
                color = random.choice(text_colors)

                # Get text size and center it
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                x = (48 - text_width) // 2
                y = (48 - text_height) // 2

                # Draw character
                draw.text((x, y), char, font=font, fill=color)

                # Add plate-like border (70% of images)
                if random.random() < 0.7:
                    border_size = random.randint(1, 3)
                    draw.rectangle([(0, 0), (47, 47)],
                                   outline=(100, 100, 100),
                                   width=border_size)

                # Add plate-like artifacts (30% of images)
                if random.random() < 0.3:
                    for _ in range(random.randint(1, 5)):
                        x_dot, y_dot = random.randint(0, 47), random.randint(0, 47)
                        draw.point((x_dot, y_dot), fill=(random.randint(0, 255),) * 3)

                # Convert to numpy array for augmentation
                img_array = np.array(img)

                # Apply random transformations
                img_array = self.augment_character(img_array)

                # Save image
                img_final = Image.fromarray(img_array)
                img_final.save(os.path.join(char_dir, f"{char}_{i:03d}.png"))

            print(f"Generated {num_per_char} images for character '{char}'")

        print(f"Synthetic dataset saved to {output_dir}")

    def augment_character(self, img_array):
        """Apply realistic augmentations to simulate license plate conditions"""
        # Convert to PIL Image for some transformations
        img_pil = Image.fromarray(img_array)

        # Random rotation (-5° to 5°)
        angle = random.uniform(-5, 5)
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=self.get_random_background())

        # Random skew (perspective transform)
        if random.random() < 0.6:
            img_array = self.apply_skew(np.array(img_pil))
            img_pil = Image.fromarray(img_array)

        # Random scale (90-110%)
        scale = random.uniform(0.9, 1.1)
        new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
        img_pil = img_pil.resize(new_size, Image.BILINEAR)

        # Random position shift
        if random.random() < 0.7:
            x_offset = random.randint(-3, 3)
            y_offset = random.randint(-3, 3)
            img_pil = ImageChops.offset(img_pil, x_offset, y_offset)

        # Convert back to numpy for OpenCV operations
        img_array = np.array(img_pil)

        # Random blur (varying intensities)
        if random.random() < 0.7:
            blur_type = random.choice(['gaussian', 'median', 'motion'])
            if blur_type == 'gaussian':
                kernel = random.choice([1, 3, 5])
                img_array = cv2.GaussianBlur(img_array, (kernel, kernel), 0)
            elif blur_type == 'median':
                kernel = random.choice([3, 5])
                img_array = cv2.medianBlur(img_array, kernel)
            else:  # motion blur
                size = random.randint(3, 7)
                kernel = np.zeros((size, size))
                kernel[int((size - 1) / 2), :] = np.ones(size)
                kernel = kernel / size
                img_array = cv2.filter2D(img_array, -1, kernel)

        # Random noise (different types)
        if random.random() < 0.7:
            noise_type = random.choice(['gaussian', 'speckle', 'salt_pepper'])
            if noise_type == 'gaussian':
                mean = 0
                var = random.uniform(0.001, 0.01)
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, img_array.shape)
                img_array = np.clip(img_array + gauss * 255, 0, 255).astype(np.uint8)
            elif noise_type == 'speckle':
                noise = np.random.randn(*img_array.shape)
                img_array = np.clip(img_array + img_array * noise * 0.1, 0, 255).astype(np.uint8)
            else:  # salt & pepper
                s_vs_p = 0.5
                amount = random.uniform(0.001, 0.01)
                out = np.copy(img_array)

                # Salt mode
                num_salt = np.ceil(amount * img_array.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
                out[tuple(coords)] = 255

                # Pepper mode
                num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
                out[tuple(coords)] = 0
                img_array = out

        # Contrast adjustment
        contrast = random.uniform(0.7, 1.5)
        img_array = np.clip(127.5 + contrast * (img_array - 127.5), 0, 255).astype(np.uint8)

        # Brightness adjustment
        brightness = random.uniform(-30, 30)
        img_array = np.clip(img_array + brightness, 0, 255).astype(np.uint8)

        # Random shadows (30% chance)
        if random.random() < 0.3:
            img_array = self.add_shadow_effect(img_array)

        # Random dirt/dust spots (20% chance)
        if random.random() < 0.2:
            img_array = self.add_dirt_effect(img_array)

        # Random exposure changes (20% chance)
        if random.random() < 0.2:
            gamma = random.uniform(0.7, 1.5)
            img_array = self.adjust_gamma(img_array, gamma)

        return img_array

    def get_random_background(self):
        """Get random background color matching plate colors"""
        return random.choice(self.background_colors)

    def apply_skew(self, img_array):
        """Apply perspective transform for skew effect"""
        h, w = img_array.shape[:2]

        # Define original points
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        # Define skewed points
        max_skew = 0.2
        skew_x = random.uniform(-max_skew, max_skew) * w
        skew_y = random.uniform(-max_skew, max_skew) * h

        pts2 = np.float32([
            [0 + skew_x, 0 + skew_y],
            [w - skew_x, 0 + skew_y],
            [0 - skew_x, h - skew_y],
            [w + skew_x, h - skew_y]
        ])

        # Get transformation matrix and apply it
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def add_shadow_effect(self, img_array):
        """Add realistic shadow effects"""
        h, w = img_array.shape[:2]

        # Create gradient mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Random shadow direction
        direction = random.choice(['top', 'bottom', 'left', 'right', 'corner'])

        if direction == 'top':
            mask = np.tile(np.linspace(1, 0, h), (w, 1)).T
        elif direction == 'bottom':
            mask = np.tile(np.linspace(0, 1, h), (w, 1)).T
        elif direction == 'left':
            mask = np.tile(np.linspace(1, 0, w), (h, 1))
        elif direction == 'right':
            mask = np.tile(np.linspace(0, 1, w), (h, 1))
        else:  # corner
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xv, yv = np.meshgrid(x, y)
            mask = np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(2)

        # Adjust shadow strength
        strength = random.uniform(0.1, 0.4)

        # Handle RGB images
        if len(img_array.shape) == 3:
            mask = np.stack([mask] * 3, axis=2)

        shadow = (img_array * (1 - strength * mask)).astype(np.uint8)
        return shadow

    def add_dirt_effect(self, img_array):
        """Add realistic dirt/dust spots"""
        h, w = img_array.shape[:2]
        result = img_array.copy()

        # Add random spots
        num_spots = random.randint(1, 5)
        for _ in range(num_spots):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            radius = random.randint(1, 3)
            darkness = random.randint(50, 150)

            if len(img_array.shape) == 3:
                cv2.circle(result, (x, y), radius, (darkness, darkness, darkness), -1)
            else:
                cv2.circle(result, (x, y), radius, darkness, -1)

        return result

    def adjust_gamma(self, image, gamma=1.0):
        """Adjust image gamma"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def load_character_dataset(self, dataset_dir):
        """Enhanced image loading with better preprocessing"""
        images = []
        labels = []

        for char_idx in range(len(CHARACTERS)):
            char_dir = os.path.join(dataset_dir, str(char_idx))
            if not os.path.exists(char_dir):
                continue

            for img_file in os.listdir(char_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(char_dir, img_file)

                    # Load image as grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # Randomly apply adaptive thresholding (50% chance)
                    if random.random() < 0.5:
                        img = cv2.adaptiveThreshold(
                            img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 11, 2
                        )
                    else:
                        # Normalize
                        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

                    # Resize with appropriate interpolation
                    resized = cv2.resize(img, self.char_size,
                                         interpolation=cv2.INTER_AREA)

                    # Random morphological operations (30% chance)
                    kernel = np.ones((2, 2), np.uint8)
                    if random.random() < 0.15:
                        resized = cv2.dilate(resized, kernel, iterations=1)
                    elif random.random() < 0.15:
                        resized = cv2.erode(resized, kernel, iterations=1)

                    # Normalize and reshape
                    normalized = resized.astype(np.float32) / 255.0
                    images.append(normalized)
                    labels.append(char_idx)

        return np.array(images), np.array(labels)

    def build_character_classifier(self):
        """Build optimized lightweight CNN model for character classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.char_size[0], self.char_size[1], 1)),

            # First conv block
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            # Second conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            # Third conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            # Classifier
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(len(CHARACTERS), activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.char_classifier = model
        return model

    def train_character_classifier(self, dataset_dir, epochs=50, batch_size=32):
        """Train the character classification model with enhanced callbacks"""
        print("Loading character dataset...")
        X, y = self.load_character_dataset(dataset_dir)

        if len(X) == 0:
            print("No data found! Please generate synthetic data first.")
            return

        print(f"Loaded {len(X)} character images")

        # Reshape for CNN (add channel dimension)
        X = X.reshape(X.shape[0], self.char_size[0], self.char_size[1], 1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")

        # Build model if not already built
        if self.char_classifier is None:
            self.build_character_classifier()

        # Enhanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]

        # Train model
        history = self.char_classifier.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.char_classifier.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_accuracy:.4f}")

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def train_character_detector(self, dataset_path, epochs=100):
        """Train YOLOv8 model for character detection with enhanced settings"""
        print("Training character detection model...")

        # Initialize YOLO model
        model = YOLO('yolov8n.pt')  # Start with nano model for speed

        # Enhanced training settings
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='moroccan_plate_detector',
            patience=20,  # Early stopping patience
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # Distribution Focal Loss gain
            fl_gamma=2.0  # Focal loss gamma
        )

        self.char_detector = model
        return results

    def detect_plate_region(self, img):
        """Helper method to detect plate region (optional)"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to preserve edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edged = cv2.Canny(gray, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and get top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Loop over contours to find potential plates
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # If the approximated contour has 4 points, assume it's a plate
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Check aspect ratio (typical plates are rectangular)
                aspect_ratio = w / float(h)
                if 2.0 < aspect_ratio < 5.0:
                    return (x, y, x + w, y + h)

        return None

    def predict_plate_text(self, image_path, confidence_threshold=0.5):
        """Enhanced prediction with post-processing"""
        if self.char_detector is None or self.char_classifier is None:
            print("Please train both models first!")
            return ""

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return ""

        # Optional: Detect plate region first
        plate_region = self.detect_plate_region(img)
        if plate_region is not None:
            x1, y1, x2, y2 = plate_region
            img = img[y1:y2, x1:x2]

        # Step 1: Detect characters
        detection_results = self.char_detector.predict(img, conf=confidence_threshold)

        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            print("No characters detected")
            return ""

        # Step 2: Extract character images and positions
        boxes = detection_results[0].boxes
        characters = []

        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract character image
            char_img = img[y1:y2, x1:x2]

            # Enhanced preprocessing
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Resize with appropriate interpolation
            resized = cv2.resize(gray, self.char_size, interpolation=cv2.INTER_AREA)

            # Normalize
            normalized = resized.astype(np.float32) / 255.0

            # Predict character
            char_input = normalized.reshape(1, self.char_size[0], self.char_size[1], 1)
            prediction = self.char_classifier.predict(char_input, verbose=0)
            char_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            characters.append({
                'char': IDX_TO_CHAR[char_idx],
                'confidence': confidence,
                'position': (x1, y1, x2, y2),
                'x_center': (x1 + x2) / 2
            })

        # Step 3: Sort characters by x-position (left to right)
        characters.sort(key=lambda x: x['x_center'])

        # Step 4: Apply Moroccan plate rules
        plate_text = self.apply_plate_rules(characters)
        avg_confidence = np.mean([char['confidence'] for char in characters])

        print(f"Detected: '{plate_text}' (confidence: {avg_confidence:.3f})")

        # Optional: Visualize results
        self.visualize_detection(img, characters)

        return plate_text

    def apply_plate_rules(self, characters):
        """Apply Moroccan plate format rules to improve accuracy"""
        chars = [c['char'] for c in characters]
        confidences = [c['confidence'] for c in characters]

        # Basic Moroccan plate patterns:
        # Common formats: 123456 ا | 123 ا 45 | 12 ا 345 | etc.

        # If we have exactly 7 characters (common Moroccan plate length)
        if len(chars) == 7:
            # Check if last character is Arabic (common pattern)
            if chars[-1] in ['أ', 'ب', 'د', 'ھ', 'ط', 'و']:
                # Validate first 6 are digits
                if all(c.isdigit() for c in chars[:-1]):
                    return ''.join(chars)

            # Check for Arabic in middle (like 123 ا 45)
            if len(chars) >= 5 and chars[3] in ['أ', 'ب', 'د', 'ھ', 'ط', 'و']:
                if all(c.isdigit() for c in chars[:3]) and all(c.isdigit() for c in chars[4:]):
                    return ''.join(chars)

        # Fallback to simple concatenation
        return ''.join(chars)

    def visualize_detection(self, img, characters):
        """Visualize character detection and recognition results"""
        display_img = img.copy()

        for char in characters:
            x1, y1, x2, y2 = char['position']

            # Draw bounding box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put character and confidence
            label = f"{char['char']} {char['confidence']:.2f}"
            cv2.putText(display_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def save_models(self, classifier_path='char_classifier.h5', detector_path='char_detector.pt'):
        """Save trained models with default paths"""
        if self.char_classifier is not None:
            self.char_classifier.save(classifier_path)
            print(f"Character classifier saved to {classifier_path}")

        if self.char_detector is not None:
            self.char_detector.save(detector_path)
            print(f"Character detector saved to {detector_path}")

    def load_models(self, classifier_path='char_classifier.h5', detector_path='char_detector.pt'):
        """Load trained models with default paths"""
        try:
            self.char_classifier = tf.keras.models.load_model(classifier_path)
            print(f"Character classifier loaded from {classifier_path}")
        except Exception as e:
            print(f"Could not load classifier from {classifier_path}: {str(e)}")

        try:
            self.char_detector = YOLO(detector_path)
            print(f"Character detector loaded from {detector_path}")
        except Exception as e:
            print(f"Could not load detector from {detector_path}: {str(e)}")


# Example usage and training pipeline
if __name__ == "__main__":
    # Initialize system
    ocr_system = CharacterOCRSystem()

    print("=== Moroccan License Plate OCR System ===")
    print("\n1. First generate synthetic training data:")
    ocr_system.generate_synthetic_characters('synthetic_chars', num_per_char=200)

    print("\n2. Then train the character classifier:")
    #ocr_system.train_character_classifier('synthetic_chars', epochs=20)

    print("\n3. Prepare YOLO dataset for character detection and train:")
    print("ocr_system.train_character_detector('character_dataset.yaml', epochs=100)")

    print("\n4. Save your trained models:")
    print("ocr_system.save_models()")

    print("\n5. Load models and test on plate images:")
    print("ocr_system.load_models()")
    print("result = ocr_system.predict_plate_text('test_plate.jpg')")
    print("print(f\"Predicted: {result}\")")
