import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from collections import defaultdict
import random


class DataPreparation:
    def __init__(self):
        self.expected_classes = {
            # Digits
            '0': 500, '1': 500, '2': 500, '3': 500, '4': 500,
            '5': 500, '6': 500, '7': 500, '8': 500, '9': 500,
            # Arabic letters (as numbers)
            '10': 500, '11': 500, '12': 500, '13': 500, '15': 500, '16': 500
        }

    def validate_data_structure(self, data_dir):
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"Error: Directory {data_dir} does not exist")
            return False

        print("Validating data structure...")
        print("=" * 50)

        total_images = 0
        missing_classes = []
        class_counts = {}

        for class_name, expected_count in self.expected_classes.items():
            class_dir = data_path / class_name

            if not class_dir.exists():
                missing_classes.append(class_name)
                print(f"❌ Missing directory: {class_name}")
                continue

            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))

            actual_count = len(image_files)
            class_counts[class_name] = actual_count
            total_images += actual_count

            if actual_count == expected_count:
                print(f"✅ {class_name}: {actual_count} images")
            elif actual_count < expected_count:
                print(f"⚠️  {class_name}: {actual_count} images (expected {expected_count})")
            else:
                print(f"ℹ️  {class_name}: {actual_count} images (more than expected {expected_count})")

        print("=" * 50)
        print(f"Total classes found: {len(class_counts)}/16")
        print(f"Total images: {total_images}")
        print(f"Expected total: {sum(self.expected_classes.values())}")

        if missing_classes:
            print(f"\nMissing classes: {missing_classes}")

        return len(missing_classes) == 0

    def visualize_samples(self, data_dir, samples_per_class=5):
        """Visualize sample images from each class using matplotlib only"""
        data_path = Path(data_dir)

        fig, axes = plt.subplots(16, samples_per_class, figsize=(20, 25))
        fig.suptitle('Sample Images from Each Class', fontsize=16)

        class_names = list(self.expected_classes.keys())

        for i, class_name in enumerate(class_names):
            class_dir = data_path / class_name

            if not class_dir.exists():
                for j in range(samples_per_class):
                    axes[i, j].text(0.5, 0.5, f'{class_name}\nNot Found',
                                    ha='center', va='center', fontsize=12, color='red')
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                continue

            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))

            if len(image_files) == 0:
                for j in range(samples_per_class):
                    axes[i, j].text(0.5, 0.5, f'{class_name}\nNo Images',
                                    ha='center', va='center', fontsize=12, color='red')
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                continue

            sample_files = random.sample(image_files, min(samples_per_class, len(image_files)))

            for j, img_file in enumerate(sample_files):
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        axes[i, j].imshow(img, cmap='gray')
                        axes[i, j].set_title(f'{class_name}')
                    else:
                        axes[i, j].text(0.5, 0.5, 'Error\nLoading',
                                        ha='center', va='center', fontsize=10, color='red')
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error:\n{str(e)[:20]}',
                                    ha='center', va='center', fontsize=8, color='red')

                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

            for j in range(len(sample_files), samples_per_class):
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    def check_image_properties(self, data_dir):
        data_path = Path(data_dir)

        print("Checking image properties...")
        print("=" * 50)

        dimensions = defaultdict(int)
        formats = defaultdict(int)
        issues = []

        for class_name in self.expected_classes.keys():
            class_dir = data_path / class_name

            if not class_dir.exists():
                continue

            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))

            for img_file in image_files[:50]:
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        issues.append(f"Could not load: {img_file}")
                        continue

                    dimensions[f"{img.shape[0]}x{img.shape[1]}"] += 1
                    formats[img_file.suffix.lower()] += 1

                    mean_brightness = np.mean(img)
                    if mean_brightness < 10:
                        issues.append(f"Very dark image: {img_file}")
                    elif mean_brightness > 245:
                        issues.append(f"Very bright image: {img_file}")

                except Exception as e:
                    issues.append(f"Error processing {img_file}: {e}")

        print("Image dimensions distribution:")
        for dim, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dim}: {count} images")

        print("\nImage formats:")
        for fmt, count in formats.items():
            print(f"  {fmt}: {count} images")

        if issues:
            print(f"\nIssues found ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        else:
            print("\n✅ No major issues found!")

    def create_train_test_split(self, data_dir, output_dir, test_ratio=0.2):
        data_path = Path(data_dir)
        output_path = Path(output_dir)

        train_dir = output_path / 'train'
        test_dir = output_path / 'test'

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating train/test split ({int((1 - test_ratio) * 100)}%/{int(test_ratio * 100)}%)...")
        print("=" * 50)

        for class_name in self.expected_classes.keys():
            class_dir = data_path / class_name

            if not class_dir.exists():
                print(f"Skipping {class_name} - directory not found")
                continue

            (train_dir / class_name).mkdir(exist_ok=True)
            (test_dir / class_name).mkdir(exist_ok=True)

            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))

            random.shuffle(image_files)
            split_idx = int(len(image_files) * (1 - test_ratio))

            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]

            for img_file in train_files:
                shutil.copy2(img_file, train_dir / class_name / img_file.name)

            for img_file in test_files:
                shutil.copy2(img_file, test_dir / class_name / img_file.name)

            print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")

        print(f"\nTrain/test split completed!")
        print(f"Train data: {train_dir}")
        print(f"Test data: {test_dir}")

    def preprocess_images(self, data_dir, output_dir, target_size=(64, 64)):
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Preprocessing images to {target_size[0]}x{target_size[1]}...")
        print("=" * 50)

        total_processed = 0

        for class_name in self.expected_classes.keys():
            class_dir = data_path / class_name
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(exist_ok=True)

            if not class_dir.exists():
                print(f"Skipping {class_name} - directory not found")
                continue

            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))

            processed_count = 0

            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    img_resized = cv2.resize(img, target_size)
                    output_file = output_class_dir / f"{img_file.stem}.png"
                    cv2.imwrite(str(output_file), img_resized)

                    processed_count += 1
                    total_processed += 1

                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

            print(f"{class_name}: {processed_count} images processed")

        print(f"\nTotal images processed: {total_processed}")
        print(f"Output directory: {output_path}")


if __name__ == "__main__":
    data_prep = DataPreparation()
    data_directory = "/home/haytham/mlp/OCR-Project/data/synth"

    print("Moroccan License Plate Data Preparation")
    print("=" * 50)

    is_valid = data_prep.validate_data_structure(data_directory)

    if is_valid:
        print("\n✅ Data structure is valid!")
        data_prep.check_image_properties(data_directory)

        print("\nGenerating sample visualization...")
        data_prep.visualize_samples(data_directory, samples_per_class=3)

        print("\nCreating train/test split...")
        data_prep.create_train_test_split(
            data_directory,
            "moroccan_lp_data_split",
            test_ratio=0.2
        )

        print("\nPreprocessing images...")
        data_prep.preprocess_images(
            data_directory,
            "moroccan_lp_preprocessed",
            target_size=(64, 64)
        )
    else:
        print("\n❌ Please fix data structure issues before proceeding")
        print("\nExpected structure:")
        for class_name, count in data_prep.expected_classes.items():
            print(f"  {class_name}/  ({count} images)")
