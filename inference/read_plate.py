import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

# Load external classifier logic
def load_character_classifier():
    """Load the trained classifier"""
    model_path = "models/classifier/moroccan_lp_classifier.keras"
    model = tf.keras.models.load_model(model_path)

    class_names = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'ا', 'ب', 'ح', 'ت', 'و', 'د'
    ]

    return model, class_names

def classify_character(image, model, class_names):
    """Classify a cropped character"""
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 64, 64, 1)

    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return class_names[predicted_class], confidence


class MoroccanPlateOCR:
    def __init__(self, detector_path='models/lp_detector/best.pt'):
        self.detector_path = detector_path
        self.classifier = None
        self.class_names = []
        self.detector = None
        self.preprocessor = PlatePreprocessor()

        # Load models
        self.load_models()

    def load_models(self):
        """Load classifier and detector models"""
        try:
            self.classifier, self.class_names = load_character_classifier()
            print("✓ Character classifier loaded.")
        except Exception as e:
            print(f"✗ Could not load classifier: {e}")
            self.classifier = None

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

    def classify_character(self, char_img):
        """Wrapper for classification"""
        if self.classifier is None:
            return None, 0.0
        return classify_character(char_img, self.classifier, self.class_names)

    def read_plate_with_detector(self, image_path, confidence_threshold=0.5):
        if self.detector is None or self.classifier is None:
            print("Both detector and classifier models are required!")
            return ""

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return ""

        plate_region = self.preprocessor.detect_plate_region(img)
        if plate_region is not None:
            x1, y1, x2, y2 = plate_region
            img = img[y1:y2, x1:x2]

        detection_results = self.detector.predict(img, conf=confidence_threshold)

        if not detection_results or not detection_results[0].boxes or detection_results[0].boxes.xyxy is None:
            print("No characters detected")
            return ""

        boxes = detection_results[0].boxes
        characters = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            char_img = img[y1:y2, x1:x2]
            char, confidence = self.classify_character(char_img)

            if char is not None:
                characters.append({
                    'char': char,
                    'confidence': confidence,
                    'position': (x1, y1, x2, y2),
                    'x_center': (x1 + x2) / 2
                })

        characters.sort(key=lambda x: x['x_center'])
        plate_text = self.apply_moroccan_plate_rules(characters)

        if characters:
            avg_conf = np.mean([c['confidence'] for c in characters])
            print(f"Detected: '{plate_text}' (confidence: {avg_conf:.3f})")

        return plate_text

    def read_plate_with_segmentation(self, image_path):
        if self.classifier is None:
            print("Character classifier model is required!")
            return ""

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return ""

        cropped_plate = self.preprocessor.crop_plate(img)
        char_images = self.preprocessor.segment_characters(cropped_plate)

        if not char_images:
            print("No characters found in segmentation")
            return ""

        characters = []
        for char_data in char_images:
            char, confidence = self.classify_character(char_data['image'])

            if char is not None:
                characters.append({
                    'char': char,
                    'confidence': confidence,
                    'position': char_data['bbox']
                })

        plate_text = self.apply_moroccan_plate_rules(characters)

        if characters:
            avg_conf = np.mean([c['confidence'] for c in characters])
            print(f"Detected: '{plate_text}' (confidence: {avg_conf:.3f})")

        return plate_text

    def apply_moroccan_plate_rules(self, characters):
        if not characters:
            return ""

        chars = [c['char'] for c in characters]
        confidences = [c['confidence'] for c in characters]

        filtered_chars = [
            char for char, conf in zip(chars, confidences) if conf > 0.3
        ]

        if not filtered_chars:
            return ''.join(chars)

        arabic_chars = ['ا', 'ب', 'ت', 'ح', 'و', 'د']
        has_arabic = any(c in arabic_chars for c in filtered_chars)
        has_numbers = any(c.isdigit() for c in filtered_chars)

        if has_arabic and has_numbers:
            return ''.join(filtered_chars)
        elif has_numbers and len(filtered_chars) >= 4:
            return ''.join(filtered_chars)

        return ''.join(filtered_chars)

    def visualize_detection(self, image_path, method='segmentation'):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return

        if method == 'detector' and self.detector is not None:
            results = self.detector.predict(img, conf=0.5)
            annotated = results[0].plot()

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.title('YOLO Character Detection')
            plt.axis('off')
            plt.show()
        else:
            cropped_plate = self.preprocessor.crop_plate(img)
            char_images = self.preprocessor.segment_characters(cropped_plate)
            self.preprocessor.visualize_segmentation(cropped_plate, char_images)

    def batch_process(self, input_folder, output_file='results.txt', method='segmentation'):
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

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

        print(f"Results saved to {output_file}")
        return results


def main():
    ocr = MoroccanPlateOCR()

    test_image = "data/plates/test_plate.jpg"
    if os.path.exists(test_image):
        print("=== Testing Single Image ===")
        print("Using segmentation method:")
        ocr.read_plate_with_segmentation(test_image)

        if ocr.detector is not None:
            print("\nUsing detector method:")
            ocr.read_plate_with_detector(test_image)

        ocr.visualize_detection(test_image)

    if os.path.exists("data/plates"):
        print("\n=== Batch Processing ===")
        ocr.batch_process("data/plates", "plate_results.txt")


if __name__ == "__main__":
    main()
