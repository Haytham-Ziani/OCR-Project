# utils/crop_and_sort.py

import cv2
from ultralytics import YOLO
import os

# === CONFIG ===
image_path = "data/plates/test_plate.jpg"  # Change this to your image
model_path = "models/lp_detector/best.pt"  # Your trained YOLOv8 model
output_dir = "data/cropped_plates"

# === CREATE OUTPUT DIR ===
os.makedirs(output_dir, exist_ok=True)

# === LOAD MODEL ===
model = YOLO(model_path)

# === LOAD IMAGE ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# === DETECT LPs ===
results = model(image_path)[0]  # Run inference
boxes = results.boxes.xyxy.cpu().numpy()  # xyxy bounding boxes

# === CROP AND SAVE EACH LP ===
for i, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"⚠️ Skipping empty crop for box {i}: {x1},{y1},{x2},{y2}")
        continue

    save_path = os.path.join(output_dir, f"plate_{i}.jpg")
    cv2.imwrite(save_path, crop)
    print(f"✅ Saved: {save_path}")
