import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class PlatePreprocessor:
    def __init__(self):
        self.min_contour_area = 100
        self.max_contour_area = 50000
    
    def detect_plate_region(self, img):
        """Detect license plate region in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while smoothing
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        # Look for rectangular contours that could be plates
        plate_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Check if it looks like a license plate
            if (len(approx) >= 4 and 
                2.0 < aspect_ratio < 6.0 and 
                w > 60 and h > 15):
                
                plate_candidates.append({
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour
                })
        
        # Return the best candidate (largest area with good aspect ratio)
        if plate_candidates:
            best_plate = max(plate_candidates, key=lambda x: x['area'])
            return best_plate['bbox']
        
        return None
    
    def crop_plate(self, img, bbox=None):
        """Crop plate region from image"""
        if bbox is None:
            bbox = self.detect_plate_region(img)
            
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Add some padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)
            
            return img[y1:y2, x1:x2]
        
        return img
    
    def enhance_plate_image(self, plate_img):
        """Enhance plate image for better OCR"""
        # Convert to grayscale if needed
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def segment_characters(self, plate_img):
        """Segment individual characters from plate image"""
        # Enhance the image first
        enhanced = self.enhance_plate_image(plate_img)
        
        # Find contours of potential characters
        contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and aspect ratio
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            # Filter based on typical character properties
            if (0.2 < aspect_ratio < 2.0 and 
                50 < area < 2000 and 
                h > 10 and w > 5):
                char_contours.append((x, y, w, h))
        
        # Sort characters left to right
        char_contours.sort(key=lambda x: x[0])
        
        # Extract character images
        char_images = []
        for x, y, w, h in char_contours:
            char_img = enhanced[y:y+h, x:x+w]
            
            # Resize to standard size while maintaining aspect ratio
            char_img = self._resize_character(char_img, target_size=(32, 32))
            char_images.append({
                'image': char_img,
                'bbox': (x, y, x+w, y+h)
            })
        
        return char_images
    
    def _resize_character(self, char_img, target_size=(32, 32)):
        """Resize character image while maintaining aspect ratio"""
        h, w = char_img.shape
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Resize
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and paste resized image in center
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def visualize_segmentation(self, original_img, char_images):
        """Visualize character segmentation results"""
        fig, axes = plt.subplots(2, len(char_images) + 1, figsize=(15, 6))
        
        # Show original image
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Draw bounding boxes on original
        annotated = original_img.copy()
        for i, char_data in enumerate(char_images):
            x1, y1, x2, y2 = char_data['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, str(i), (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[1, 0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Segmented')
        axes[1, 0].axis('off')
        
        # Show individual characters
        for i, char_data in enumerate(char_images):
            if i + 1 < len(axes[0]):
                axes[0, i+1].imshow(char_data['image'], cmap='gray')
                axes[0, i+1].set_title(f'Char {i}')
                axes[0, i+1].axis('off')
                
                axes[1, i+1].axis('off')  # Hide bottom row for extra chars
        
        plt.tight_layout()
        plt.show()
    
    def process_plate_folder(self, input_folder, output_folder):
        """Process all plate images in a folder"""
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Crop plate region
                    cropped = self.crop_plate(img)
                    
                    # Segment characters
                    char_images = self.segment_characters(cropped)
                    
                    # Save results
                    base_name = os.path.splitext(filename)[0]
                    
                    # Save cropped plate
                    plate_path = os.path.join(output_folder, f"{base_name}_plate.jpg")
                    cv2.imwrite(plate_path, cropped)
                    
                    # Save individual characters
                    for i, char_data in enumerate(char_images):
                        char_path = os.path.join(output_folder, f"{base_name}_char_{i}.jpg")
                        cv2.imwrite(char_path, char_data['image'])
                    
                    print(f"Processed {filename}: {len(char_images)} characters found")

if __name__ == "__main__":
    preprocessor = PlatePreprocessor()
    
    # Process all plates in the plates folder
    preprocessor.process_plate_folder('data/plates', 'data/chars')
