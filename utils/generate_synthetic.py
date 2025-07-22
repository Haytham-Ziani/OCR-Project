import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2

# Character mapping for Moroccan license plates
# Note: The 'ط' character has a value of 15, implying 14 is skipped.
# This is fine if intentional, otherwise consider re-indexing or adding missing characters.
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
    def __init__(self, char_output_size=(48, 48), font_size=24, font_dir=None):
        """
        Initializes the synthetic data generator.

        Args:
            char_output_size (tuple): The desired (width, height) of the output character images.
            font_size (int): The base font size for rendering characters.
            font_dir (str, optional): The directory where fonts are located.
                                      If None, it tries to deduce the path relative to the script.
        """
        self.char_output_size = char_output_size # (width, height)
        self.font_size = font_size

        if font_dir is None:
            # Deduce font_dir relative to the script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.font_dir = os.path.join(script_dir, '..', 'fonts')
        else:
            self.font_dir = font_dir

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
                # Use self.font_size for consistent font loading
                fonts.append(ImageFont.truetype(font_path, size=self.font_size))
                print(f"Loaded font: {font_path}")
            except Exception as e:
                print(f"Couldn't load font {font_path}: {str(e)}")

        if not fonts:
            print("Warning: No custom fonts loaded. Using default font - Arabic may not render properly.")
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
                # Create base image with the target output size
                img = Image.new('RGB', self.char_output_size, random.choice(self.background_colors))
                draw = ImageDraw.Draw(img)

                # Select random font and color
                font = random.choice(self.fonts)
                color = random.choice(text_colors)

                # Center the text
                # textbbox uses (left, top, right, bottom)
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Calculate position to center the text in the output image
                x = (self.char_output_size[0] - text_width) // 2 - bbox[0] # Adjust for font's internal bbox origin
                y = (self.char_output_size[1] - text_height) // 2 - bbox[1]

                # Draw character
                draw.text((x, y), char, font=font, fill=color)

                # Add plate-like border (70% chance)
                if random.random() < 0.7:
                    border_size = random.randint(1, 3)
                    # Draw border on the image, respecting its dimensions
                    draw.rectangle([(0, 0), (self.char_output_size[0]-1, self.char_output_size[1]-1)],
                                 outline=(100, 100, 100), width=border_size)

                # Convert to numpy and apply augmentations
                img_array = np.array(img)
                img_array = self._augment_character(img_array)

                # Save image
                img_final = Image.fromarray(img_array)
                # Ensure the final image is saved at the specified output size
                # This is already handled if img was created with char_output_size and augmentations preserve it.
                # If augmentations change size (e.g., severe scaling), you might need a final resize here.
                # For current augmentations, `img_pil.resize` is used within augmentation, so this is fine.
                img_final.save(os.path.join(char_dir, f"{char}_{i:03d}.png"))

            print(f"Generated {num_per_char} images for character '{char}'")

        print(f"Synthetic dataset saved to {output_dir}")

    def _augment_character(self, img_array):
        """Apply realistic augmentations"""
        img_pil = Image.fromarray(img_array)
        original_size = img_pil.size # Store original size to crop back if needed

        # Random rotation (-5° to 5°)
        angle = random.uniform(-5, 5)
        # expand=False keeps the original size, cropping if rotation goes out of bounds
        # expand=True would enlarge the canvas to fit the rotation, then we'd need to crop/resize
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR,
                                fillcolor=random.choice(self.background_colors), expand=False)


        # Random scale (90-110%)
        # Note: Scaling might change dimensions. We'll resize back at the end of augmentation.
        scale_factor = random.uniform(0.9, 1.1)
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
        img_pil = img_pil.resize((new_width, new_height), Image.BILINEAR)

        # Random position shift (using paste to handle shifting out of bounds by filling with background)
        if random.random() < 0.7:
            x_offset = random.randint(-3, 3)
            y_offset = random.randint(-3, 3)
            temp_img = Image.new('RGB', img_pil.size, random.choice(self.background_colors))
            temp_img.paste(img_pil, (x_offset, y_offset))
            img_pil = temp_img

        # Convert back to numpy for OpenCV operations
        img_array = np.array(img_pil)

        # Random blur
        if random.random() < 0.7:
            blur_type = random.choice(['gaussian', 'median'])
            if blur_type == 'gaussian':
                kernel = random.choice([1, 3, 5]) # Kernel must be odd
                img_array = cv2.GaussianBlur(img_array, (kernel, kernel), 0)
            else: # median
                kernel = random.choice([3, 5]) # Kernel must be odd and > 1
                img_array = cv2.medianBlur(img_array, kernel)

        # Random noise (Gaussian noise)
        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(5, 15), img_array.shape).astype(np.int32)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Contrast and brightness adjustment
        contrast = random.uniform(0.7, 1.5)
        brightness = random.uniform(-30, 30)
        img_array = np.clip(127.5 + contrast * (img_array - 127.5) + brightness, 0, 255).astype(np.uint8)

        # Ensure the final image is resized back to the original char_output_size if it changed
        # This is crucial if scaling or other ops changed the size
        img_final_aug = Image.fromarray(img_array)
        if img_final_aug.size != self.char_output_size:
            img_final_aug = img_final_aug.resize(self.char_output_size, Image.BILINEAR)

        return np.array(img_final_aug)

if __name__ == "__main__":
    # Initialize the generator with the desired output size for character images
    # 48x48 seems to be the intended size from the original script
    generator = SyntheticDataGenerator(char_output_size=(48, 48), font_size=24)
    generator.generate_character_images('data/synth', num_per_char=300)