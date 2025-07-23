# Moroccan License Plate OCR System

A comprehensive OCR system designed specifically for Moroccan license plates using deep learning techniques. The system combines character detection (YOLOv8) and character classification (CNN) for accurate text recognition.

## Features

- **Two-stage OCR Pipeline**: Character detection + classification
- **Moroccan Character Support**: Handles Arabic characters (أ, ب, د, ھ, ط, و) and digits (0-9)
- **Synthetic Data Generation**: Creates training data with realistic augmentations
- **Multiple Detection Methods**: Traditional segmentation and YOLO-based detection
- **Batch Processing**: Process multiple images at once
- **Visualization Tools**: View detection and segmentation results

## Project Structure

```
OCR-Project/
├── data/
│   ├── chars/          # Individual character images
│   ├── plates/         # Original plate images
│   ├── synth/          # Synthetic training data
│   └── yolo_dataset/   # YOLO training dataset
├── fonts/              # Arabic/Latin fonts for synthesis
│   ├── ARIAL.TTF
│   ├── KFGQPC Uthman Taha Naskh Regular.ttf
│   └── tahoma.ttf
├── models/
│   ├── classifier/     # CNN character classifier models
│   └── detector/       # YOLO detection models
├── training/
│   ├── classify.py     # Train character classifier
│   └── detect.py       # Train character detector
├── utils/
│   ├── generate_synthetic.py  # Synthetic data generation
│   └── crop_and_sort.py      # Image preprocessing
├── inference/
│   └── read_plate.py   # Main OCR inference
├── main.py            # Main training/testing pipeline
└── requirements.txt
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd OCR-Project
```

2. **Create virtual environment:**
```bash
python -m venv mlp
source mlp/bin/activate  # On Windows: mlp\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup directory structure:**
```bash
python main.py setup
```

## Quick Start

### 1. Prepare Your Data

Add your fonts to the `fonts/` directory and plate images to `data/plates/`:

```bash
# Add fonts (required for Arabic character synthesis)
cp your_arabic_font.ttf fonts/

# Add plate images for detection training
cp your_plate_images/* data/plates/
```

### 2. Train the Complete System

```bash
# Generate synthetic data and train both models
python main.py train-all --epochs 50 --num-per-char 500
```

Or train components separately:

```bash
# Generate synthetic character data
python main.py generate --num-per-char 500

# Train character classifier
python main.py train-classifier --epochs 50

# Train character detector (requires plate images)
python main.py train-detector --epochs 100
```

### 3. Test OCR on Images

```bash
# Test on single image
python main.py test --image data/plates/test_plate.jpg --method segmentation

# Test on all images in data/plates/
python main.py test --method detector
```

## Usage Examples

### Basic OCR Inference

```python
from inference.read_plate import MoroccanPlateOCR

# Initialize OCR system
ocr = MoroccanPlateOCR()

# Read plate using segmentation method
result = ocr.read_plate_with_segmentation('plate_image.jpg')
print(f"Detected text: {result}")

# Read plate using YOLO detector (if trained)
result = ocr.read_plate_with_detector('plate_image.jpg')
print(f"Detected text: {result}")

# Visualize detection process
ocr.visualize_detection('plate_image.jpg')
```

### Generate Synthetic Training Data

```python
from utils.generate_synthetic import SyntheticDataGenerator

generator = SyntheticDataGenerator(font_dir='fonts')
generator.generate_character_images('data/synth', num_per_char=1000)
```

### Train Character Classifier

```python
from training.classify import CharacterClassifier

classifier = CharacterClassifier()
history = classifier.train('data/synth', epochs=50)
classifier.save_model('models/classifier/my_model.h5')
```

### Preprocess Plate Images

```python
from utils.crop_and_sort import PlatePreprocessor

preprocessor = PlatePreprocessor()
preprocessor.process_plate_folder('data/plates', 'data/chars')
```

## Character Set

The system supports 16 character classes for Moroccan license plates:

- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Arabic Letters**: أ (alif), ب (baa), د (dal), ھ (haa), ط (taa), و (waw)

## Model Architecture

### Character Classifier (CNN)
- Input: 32x32 grayscale images  
- Architecture: 3 conv blocks + 2 dense layers
- Features: Batch normalization, dropout, data augmentation
- Output: 16 character classes

### Character Detector (YOLOv8)
- Model: YOLOv8 Nano for speed
- Task: Object detection (character bounding boxes)
- Input: Variable size plate images
- Output: Character locations + confidence scores

## Training Tips

1. **Font Quality**: Use high-quality Arabic fonts for better synthetic data
2. **Data Balance**: Ensure balanced representation of all characters
3. **Augmentation**: The system applies realistic augmentations (blur, noise, rotation)
4. **Plate Quality**: Use clear, well-lit plate images for detector training
5. **Validation**: Monitor both accuracy and loss during training

## Performance Optimization

- **GPU Usage**: Enable GPU acceleration for faster training
- **Batch Size**: Adjust based on your GPU memory
- **Model Size**: Use YOLOv8n for speed, YOLOv8s/m for accuracy
- **Input Resolution**: Balance between accuracy and speed

## Troubleshooting

### Common Issues

1. **Arabic text not rendering**: Install proper Arabic fonts
2. **Low detection accuracy**: Increase training data or epochs
3. **GPU memory issues**: Reduce batch size
4. **Poor OCR results**: Check image quality and model training

### Performance Tuning

```python
# Adjust confidence thresholds
ocr.read_plate_with_detector('image.jpg', confidence_threshold=0.3)

# Use different preprocessing
preprocessor = PlatePreprocessor()
enhanced = preprocessor.enhance_plate_image(plate_img)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- TensorFlow/Keras for deep learning
- OpenCV for image processing


## Citation

If you use this system in your research, please cite:

```bibtex
@software{moroccan_plate_ocr,
  title={Moroccan License Plate OCR System},
  author={Haytham Ziani},
  year={2025},
  url={https://github.com/Haytham-Ziani/OCR-Project}
}
```
