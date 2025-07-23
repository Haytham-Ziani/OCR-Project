import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.crop_and_sort import PlatePreprocessor

# Character mapping
CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'أ': 10, 'ب': 11, 'د': 12, 'ھ': 13, 'ط': 15, 'و': 16
}

IDX_TO_CHAR = {v: k for k, v in CHARACTERS.items()}


class MoroccanPlateOCR:
    def __init__(self, classifier_path='models/classifier/char_classifier.h5',
                 detector_path='models/detector/char_detector.pt'):
        self.classifier_path = classifier_path
        self.detector_path = detector_path
        self.classifier = None
        self.detector = None
        self.preprocessor = PlatePreprocessor()
        self.char_size = (32, 32)

        # Load models
        self.load_models()

    def load_models(self):
        """Load both classifier and detector models"""
        # Load character classifier
        try:
            self.classifier = tf.keras.models.load_model(self.classifier_path)
            print(f"✓ Character classifier loaded from {self.classifier_path}")
        except Exception as e:
            print(f"✗ Could not load classifier: {e}")
            self.classifier = None

        # Load character detector (YOLO)
        try:
            if os.path.exists(self.detector_path):
                self.detector = YOLO(self.detector_path)
                print(f"✓ Character detector loaded from {self.detector_path}")
            else:
                print(f"✗ Detector model not found at {self.detector_path}")
                self.detector = None
        except Exception as e:
            print(f"✗ Could not load detector: {e}")
            self.detector = None

    def preprocess_character(self, char_img):
        """Preprocess character image for classification"""
        # Convert to grayscale if needed
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img.copy()

        # Resize to model input size
        resized = cv2.resize(gray, self.char_size, interpolation=cv2.INTER_AREA)

        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Normalize
        normalized = thresh.astype(np.float32) / 255.0

        return normalized

    def classify_character(self, char_img):
        """Classify a single character image"""
        if self.classifier is None:
            return None, 0.0

        # Preprocess
        processed = self.preprocess_character(char_img)

        # Reshape for model input
        input_img = processed.reshape(1, self.char_size[0], self.char_size[1], 1)

        # Predict
        prediction = self.classifier.predict(input_img, verbose=0)
        char_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        return IDX_TO_CHAR[char_idx], confidence

    def read_plate_with_detector(self, image_path, confidence_threshold=0.5):
        """Read plate using YOLO character detector + CNN classifier"""
        if self.detector is None or self.classifier is None:
            print("Both detector and classifier models are required!")
            return ""

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return ""

        # Optional: Crop plate region first
        plate_region = self.preprocessor.detect_plate_region(img)
        if plate_region is not None:
            x1, y1, x2, y2 = plate_region
            img = img[y1:y2, x1:x2]

        # Detect characters using YOLO
        detection_results = self.detector.predict(img, conf=confidence_threshold)

        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            print("No characters detected")
            return ""

        # Extract character images and positions
        boxes = detection_results[0].boxes
        characters = []

        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract character image
            char_img = img[y1:y2, x1:x2]

            # Classify character
            char, confidence = self.classify_character(char_img)

            if char is not None:
                characters.append({
                    'char': char,
                    'confidence': confidence,
                    'position': (x1, y1, x2, y2),
                    'x_center': (x1 + x2) / 2
                })

        # Sort characters by x-position (left to right)
        characters.sort(key=lambda x: x['x_center'])

        # Apply Moroccan plate rules
        plate_text = self.apply_moroccan_plate_rules(characters)

        # Calculate average confidence
        if characters:
            avg_confidence = np.mean([char['confidence'] for char in characters])
            print(f"Detected: '{plate_text}' (confidence: {avg_confidence:.3f})")

        return plate_text

    def read_plate_with_segmentation(self, image_path):
        """Read plate using traditional segmentation + CNN classifier"""
        if self.classifier is None:
            print("Character classifier model is required!")
            return ""

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return ""

        # Crop plate region
        cropped_plate = self.preprocessor.crop_plate(img)

        # Segment characters
        char_images = self.preprocessor.segment_characters(cropped_plate)

        if not char_images:
            print("No characters found in segmentation")
            return ""

        # Classify each character
        characters = []
        for i, char_data in enumerate(char_images):
            char, confidence = self.classify_character(char_data['image'])

            if char is not None:
                characters.append({
                    'char': char,
                    'confidence': confidence,
                    'position': char_data['bbox']
                })

        # Apply Moroccan plate rules
        plate_text = self.apply_moroccan_plate_rules(characters)

        # Calculate average confidence
        if characters:
            avg_confidence = np.mean([char['confidence'] for char in characters])
            print(f"Detected: '{plate_text}' (confidence: {avg_confidence:.3f})")

        return plate_text

    def apply_moroccan_plate_rules(self, characters):
        """Apply Moroccan license plate format rules"""
        if not characters:
            return ""

        chars = [c['char'] for c in characters]
        confidences = [c['confidence'] for c in characters]

        # Filter out low-confidence characters
        filtered_chars = []
        for char, conf in zip(chars, confidences):
            if conf > 0.3:  # Confidence threshold
                filtered_chars.append(char)

        if not filtered_chars:
            return ''.join(chars)  # Fallback to all characters

        # Common Moroccan plate patterns validation
        plate_text = ''.join(filtered_chars)

        # Basic validation: check if we have reasonable mix of numbers and Arabic letters
        arabic_chars = ['أ', 'ب', 'د', 'ھ', 'ط', 'و']
        has_arabic = any(c in arabic_chars for c in filtered_chars)
        has_numbers = any(c.isdigit() for c in filtered_chars)

        if has_arabic and has_numbers:
            return plate_text
        elif has_numbers and len(filtered_chars) >= 4:
            return plate_text  # Might be old format with only numbers

        return plate_text  # Return whatever we found

    def visualize_detection(self, image_path, method='segmentation'):
        """Visualize the character detection/segmentation process"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return

        if method == 'detector' and self.detector is not None:
            # Use YOLO detection
            results = self.detector.predict(img, conf=0.5)
            annotated = results[0].plot()

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.title('YOLO Character Detection')
            plt.axis('off')
            plt.show()

        else:
            # Use traditional segmentation
            cropped_plate = self.preprocessor.crop_plate(img)
            char_images = self.preprocessor.segment_characters(cropped_plate)
            self.preprocessor.visualize_segmentation(cropped_plate, char_images)

    def batch_process(self, input_folder, output_file='results.txt', method='segmentation'):
        """Process multiple plate images"""
        results = []

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)

                print(f"Processing {filename}...")

                if method == 'detector':
                    plate_text = self.read_plate_with_detector(image_path)
                else:
                    plate_text = self.read_plate_with_segmentation(image_path)

                results.append(f"{filename}: {plate_text}")
                print(f"Result: {plate_text}\n")

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

        print(f"Results saved to {output_file}")
        return results


def main():
    """Main function for testing"""
    # Initialize OCR system
    ocr = MoroccanPlateOCR()

    # Test on a single image
    test_image = "data/plates/test_plate.jpg"
    if os.path.exists(test_image):
        print("=== Testing Single Image ===")

        # Try segmentation method
        print("Using segmentation method:")
        result1 = ocr.read_plate_with_segmentation(test_image)

        # Try detector method (if available)
        if ocr.detector is not None:
            print("\nUsing detector method:")
            result2 = ocr.read_plate_with_detector(test_image)

        # Visualize
        ocr.visualize_detection(test_image)

    # Batch process all plates
    if os.path.exists("data/plates"):
        print("\n=== Batch Processing ===")
        ocr.batch_process("data/plates", "plate_results.txt")


if __name__ == "__main__":
    main()