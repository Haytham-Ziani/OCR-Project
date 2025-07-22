import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2

# Character mapping for Moroccan license plates
CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'أ': 10,  # alif
    'ب': 11,  # baa
    'د': 12,  # dal
    'ھ': 13,  # haa
    'ط': 15,  # taa 
    'و': 16   # waw
}

IDX_TO_CHAR = {v: k for k, v in CHARACTERS.items()}

class SyntheticDataGenerator:
    def __init__(self, font_dir='fonts'):
        self.font_dir = font_dir
        self.char_size = (32, 32)
        self.background_colors = [
            (255, 255, 255),  # White
            (220, 220, 220),  # Light gray
            (255, 200, 150),  # Light orange
            (200, 255, 200)   # Light green
        ]
        self.fonts = self._load_fonts()

    def _load_fonts(self):
        """Load available fonts from fonts directory"""
        fonts = []
        font_files = ['ARIAL.TTF', 'KFGQPC Uthman Taha Naskh Regular.ttf', 'tahoma.ttf']
        
        for font_file in font_files:
            font_path = os.path.join(self.font_dir, font_file)
            try:
                fonts.append(ImageFont.truetype(font_path, size=24))
                print(f"Loaded font: {font_path}")
            except Exception as e:
                print(f"Couldn't load font {font_path}: {str(e)}")
        
        if not fonts:
            print("Warning: Using default font - Arabic may not render properly")
            fonts = [ImageFont.load_default()]
        
        return fonts

    def generate_character_images(self, output_dir, num_per_char=500):
        """Generate synthetic character images for training"""
        print("Generating synthetic character dataset...")
        os.makedirs(output_dir, exist_ok=True)
        
        text_colors = [(0, 0, 0), (30, 30, 30), (50, 50, 50)]
        
        for char, char_idx in CHARACTERS.items():
            char_dir = os.path.join(output_dir, str(char_idx))
            os.makedirs(char_dir, exist_ok=True)
            
            for i in range(num_per_char):
                # Create base image
                img = Image.new('RGB', (48, 48), random.choice(self.background_colors))
                draw = ImageDraw.Draw(img)
                
                # Select random font and color
                font = random.choice(self.fonts)
                color = random.choice(text_colors)
                
                # Center the text
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (48 - text_width) // 2
                y = (48 - text_height) // 2
                
                # Draw character
                draw.text((x, y), char, font=font, fill=color)
                
                # Add plate-like border (70% chance)
                if random.random() < 0.7:
                    border_size = random.randint(1, 3)
                    draw.rectangle([(0, 0), (47, 47)], 
                                 outline=(100, 100, 100), width=border_size)
                
                # Convert to numpy and apply augmentations
                img_array = np.array(img)
                img_array = self._augment_character(img_array)
                
                # Save image
                img_final = Image.fromarray(img_array)
                img_final.save(os.path.join(char_dir, f"{char}_{i:03d}.png"))
            
            print(f"Generated {num_per_char} images for character '{char}'")
        
        print(f"Synthetic dataset saved to {output_dir}")

    def _augment_character(self, img_array):
        """Apply realistic augmentations"""
        img_pil = Image.fromarray(img_array)
        
        # Random rotation (-5° to 5°)
        angle = random.uniform(-5, 5)
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, 
                                fillcolor=random.choice(self.background_colors))
        
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
        
        # Random blur
        if random.random() < 0.7:
            blur_type = random.choice(['gaussian', 'median'])
            if blur_type == 'gaussian':
                kernel = random.choice([1, 3, 5])
                img_array = cv2.GaussianBlur(img_array, (kernel, kernel), 0)
            else:
                kernel = random.choice([3, 5])
                img_array = cv2.medianBlur(img_array, kernel)
        
        # Random noise
        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Contrast and brightness adjustment
        contrast = random.uniform(0.7, 1.5)
        brightness = random.uniform(-30, 30)
        img_array = np.clip(127.5 + contrast * (img_array - 127.5) + brightness, 0, 255).astype(np.uint8)
        
        return img_array

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_character_images('data/synth', num_per_char=300)
