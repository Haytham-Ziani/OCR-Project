import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import random
import shutil
from pathlib import Path


class YOLOCharacterDetector:
    def __init__(self, model_size='n'):  # n, s, m, l, x
        self.model_size = model_size
        self.model = None
        self.dataset_path = None

    def prepare_yolo_dataset(self, plate_images_dir, output_dir='data/yolo_dataset'):
        """
        Prepare YOLO dataset from plate images
        This assumes you have plate images and want to create character detection annotations
        """
        print("Preparing YOLO dataset...")

        # Create directory structure
        train_dir = os.path.join(output_dir, 'images', 'train')
        val_dir = os.path.join(output_dir, 'images', 'val')
        train_labels_dir = os.path.join(output_dir, 'labels', 'train')
        val_labels_dir = os.path.join(output_dir, 'labels', 'val')

        for dir_path in [train_dir, val_dir, train_labels_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Get all plate images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(plate_images_dir).glob(ext))

        # Split into train/val (80/20)
        random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Found {len(image_files)} images")
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")

        # Process training images
        self._process_images(train_files, train_dir, train_labels_dir)

        # Process validation images
        self._process_images(val_files, val_dir, val_labels_dir)

        # Create dataset.yaml file
        self._create_dataset_yaml(output_dir)

        self.dataset_path = os.path.join(output_dir, 'dataset.yaml')
        print(f"YOLO dataset prepared at {output_dir}")
        return self.dataset_path

    def _process_images(self, image_files, img_output_dir, label_output_dir):
        """Process images and create YOLO format annotations"""
        from utils.crop_and_sort import PlatePreprocessor
        preprocessor = PlatePreprocessor()

        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Copy image to output directory
            output_img_path = os.path.join(img_output_dir, img_file.name)
            shutil.copy2(str(img_file), output_img_path)

            # Create annotation file
            annotation_path = os.path.join(label_output_dir,
                                           img_file.stem + '.txt')

            # Detect plate region and segment characters
            try:
                plate_region = preprocessor.detect_plate_region(img)
                if plate_region:
                    x1, y1, x2, y2 = plate_region
                    plate_img = img[y1:y2, x1:x2]
                    char_images = preprocessor.segment_characters(plate_img)

                    # Create YOLO annotations
                    annotations = []
                    img_h, img_w = img.shape[:2]

                    for char_data in char_images:
                        char_x1, char_y1, char_x2, char_y2 = char_data['bbox']

                        # Convert to absolute coordinates
                        abs_x1 = x1 + char_x1
                        abs_y1 = y1 + char_y1
                        abs_x2 = x1 + char_x2
                        abs_y2 = y1 + char_y2

                        # Convert to YOLO format (normalized center x, y, width, height)
                        center_x = (abs_x1 + abs_x2) / 2 / img_w
                        center_y = (abs_y1 + abs_y2) / 2 / img_h
                        width = (abs_x2 - abs_x1) / img_w
                        height = (abs_y2 - abs_y1) / img_h

                        # Class 0 for character (single class detection)
                        annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                    # Save annotations
                    with open(annotation_path, 'w') as f:
                        f.write('\n'.join(annotations))

                else:
                    # Create empty annotation file if no plate detected
                    with open(annotation_path, 'w') as f:
                        f.write('')

            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                # Create empty annotation file
                with open(annotation_path, 'w') as f:
                    f.write('')

    def _create_dataset_yaml(self, output_dir):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': os.path.abspath(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'character'}
        }

        yaml_path = os.path.join(output_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f)

    def train_detector(self, dataset_yaml_path=None, epochs=100, batch_size=16):
        """Train YOLO character detector"""
        if dataset_yaml_path is None:
            dataset_yaml_path = self.dataset_path

        if dataset_yaml_path is None:
            raise ValueError("Dataset path not provided. Please prepare dataset first.")

        print("Training YOLO character detector...")

        # Initialize YOLO model
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)

        # Create models directory
        os.makedirs('models/detector', exist_ok=True)

        # Training configuration
        results = self.model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            name='moroccan_char_detector',
            project='models/detector',
            patience=20,
            save_period=10,
            # Optimization parameters
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            # Loss parameters
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0
        )

        print("Training completed!")

        # Save the best model
        best_model_path = f'models/detector/moroccan_char_detector/weights/best.pt'
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, 'models/detector/char_detector.pt')
            print("Best model saved as models/detector/char_detector.pt")

        return results

    def evaluate_detector(self, dataset_yaml_path=None):
        """Evaluate the trained detector"""
        if self.model is None:
            print("No model loaded for evaluation")
            return

        if dataset_yaml_path is None:
            dataset_yaml_path = self.dataset_path

        print("Evaluating detector...")
        results = self.model.val(data=dataset_yaml_path)
        return results

    def test_detection(self, image_path, conf_threshold=0.5):
        """Test detection on a single image"""
        if self.model is None:
            print("No model loaded")
            return

        results = self.model.predict(image_path, conf=conf_threshold)

        # Show results
        for result in results:
            result.show()

        return results

    def export_model(self, format='onnx'):
        """Export model to different formats"""
        if self.model is None:
            print("No model loaded")
            return

        self.model.export(format=format)
        print(f"Model exported to {format} format")


def create_sample_dataset():
    """Create a sample dataset structure for demonstration"""
    print("Creating sample dataset structure...")

    # Create directory structure
    dirs = [
        'data/plates',
        'data/yolo_dataset/images/train',
        'data/yolo_dataset/images/val',
        'data/yolo_dataset/labels/train',
        'data/yolo_dataset/labels/val'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("Sample dataset structure created.")
    print("Please add your plate images to 'data/plates/' directory")


if __name__ == "__main__":
    # Initialize detector trainer
    detector_trainer = YOLOCharacterDetector(model_size='n')  # Use nano model for speed

    # Check if plate images exist
    if not os.path.exists('data/plates') or not os.listdir('data/plates'):
        print("No plate images found. Creating sample dataset structure...")
        create_sample_dataset()
        print("\nPlease add your plate images to 'data/plates/' and run again.")
    else:
        print("=== Training YOLO Character Detector ===")

        # Prepare dataset
        dataset_path = detector_trainer.prepare_yolo_dataset('data/plates')

        # Train detector
        results = detector_trainer.train_detector(dataset_path, epochs=50, batch_size=8)

        # Evaluate
        detector_trainer.evaluate_detector(dataset_path)

        print("Training completed! Model saved to models/detector/char_detector.pt")