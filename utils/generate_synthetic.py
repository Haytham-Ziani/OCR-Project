import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter, ImageEnhance
import cv2

# Character mapping for Moroccan license plates (fixed continuous indexing)
CHARACTERS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'أ': 10,  # alif
    'ب': 11,  # baa
    'د': 12,  # dal
    'ھ': 13,  # haa
    'ط': 15,  # taa (fixed index)
    'و': 16  # waw (fixed index)
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
        self.char_output_size = char_output_size  # (width, height)
        self.font_size = font_size

        if font_dir is None:
            # Deduce font_dir relative to the script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.font_dir = os.path.join(script_dir, '..', 'fonts')
        else:
            self.font_dir = font_dir

        # Expanded background colors for more variety
        self.background_colors = [
            (255, 255, 255),  # White
            (250, 250, 250),  # Off-white
            (240, 240, 240),  # Light gray
            (220, 220, 220),  # Gray
            (255, 255, 240),  # Cream
            (250, 245, 230),  # Beige
            (255, 200, 150),  # Light orange
            (200, 255, 200),  # Light green
            (245, 245, 220)  # Light yellow
        ]
        self.fonts = self._load_fonts()

    def _load_fonts(self):
        """Load available fonts from fonts directory with size variations"""
        fonts = []
        font_files = ['ARIAL.TTF', 'KFGQPC Uthman Taha Naskh Regular.ttf', 'tahoma.ttf']

        for font_file in font_files:
            font_path = os.path.join(self.font_dir, font_file)
            try:
                # Load multiple sizes for each font
                base_sizes = [self.font_size - 2, self.font_size, self.font_size + 2, self.font_size + 4]
                for size in base_sizes:
                    fonts.append(ImageFont.truetype(font_path, size=size))
                print(f"Loaded font: {font_path} with {len(base_sizes)} size variations")
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

        # Expanded text colors including weathered/faded variations
        text_colors = [
            (0, 0, 0),  # Black
            (20, 20, 20),  # Near black
            (40, 40, 40),  # Dark gray
            (60, 60, 60),  # Medium dark gray
            (80, 80, 80),  # Medium gray
            (15, 25, 35),  # Dark blue-gray (weathered)
            (45, 35, 25),  # Brown-gray (dirt/rust)
            (35, 45, 35),  # Green-gray (oxidation)
        ]

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
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Calculate position to center the text in the output image
                x = (self.char_output_size[0] - text_width) // 2 - bbox[0]
                y = (self.char_output_size[1] - text_height) // 2 - bbox[1]

                # Draw character
                draw.text((x, y), char, font=font, fill=color)

                # Add plate-like border (70% chance)
                if random.random() < 0.7:
                    border_size = random.randint(1, 3)
                    border_color = (random.randint(80, 120), random.randint(80, 120), random.randint(80, 120))
                    draw.rectangle([(0, 0), (self.char_output_size[0] - 1, self.char_output_size[1] - 1)],
                                   outline=border_color, width=border_size)

                # Convert to numpy and apply augmentations
                img_array = np.array(img)
                img_array = self._augment_character(img_array)

                # Save image
                img_final = Image.fromarray(img_array)
                img_final.save(os.path.join(char_dir, f"{char}_{i:03d}.png"))

            print(f"Generated {num_per_char} images for character '{char}'")

        print(f"Synthetic dataset saved to {output_dir}")

    def _apply_perspective_distortion(self, img_array):
        """Apply random perspective distortion to simulate angled license plates"""
        h, w = img_array.shape[:2]

        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Add random distortion to destination points
        distortion = random.uniform(2, 8)  # Distortion strength
        dst_points = np.float32([
            [random.uniform(-distortion, distortion), random.uniform(-distortion, distortion)],
            [w + random.uniform(-distortion, distortion), random.uniform(-distortion, distortion)],
            [w + random.uniform(-distortion, distortion), h + random.uniform(-distortion, distortion)],
            [random.uniform(-distortion, distortion), h + random.uniform(-distortion, distortion)]
        ])

        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        distorted = cv2.warpPerspective(img_array, matrix, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=random.choice(self.background_colors))

        return distorted

    def _apply_lighting_effects(self, img_array):
        """Apply realistic lighting variations"""
        img_pil = Image.fromarray(img_array)

        # Random brightness variation across the image (uneven illumination)
        if random.random() < 0.6:
            # Create gradient overlay
            overlay = Image.new('L', img_pil.size, 128)
            draw = ImageDraw.Draw(overlay)

            # Random gradient direction and intensity
            gradient_type = random.choice(['horizontal', 'vertical', 'radial'])
            intensity = random.uniform(0.3, 0.8)

            if gradient_type == 'horizontal':
                for x in range(img_pil.size[0]):
                    gray_val = int(128 + intensity * 127 * (x / img_pil.size[0] - 0.5))
                    draw.line([(x, 0), (x, img_pil.size[1])], fill=gray_val)
            elif gradient_type == 'vertical':
                for y in range(img_pil.size[1]):
                    gray_val = int(128 + intensity * 127 * (y / img_pil.size[1] - 0.5))
                    draw.line([(0, y), (img_pil.size[0], y)], fill=gray_val)
            else:  # radial
                center_x, center_y = img_pil.size[0] // 2, img_pil.size[1] // 2
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                for x in range(img_pil.size[0]):
                    for y in range(img_pil.size[1]):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        gray_val = int(128 + intensity * 127 * (dist / max_dist - 0.5))
                        overlay.putpixel((x, y), gray_val)

            # Apply overlay
            img_pil = ImageChops.multiply(img_pil, overlay.convert('RGB'))

        return np.array(img_pil)

    def _apply_weathering_effects(self, img_array):
        """Apply weathering effects like scratches, dirt, and fading"""
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)

        # Add scratches (30% chance)
        if random.random() < 0.3:
            num_scratches = random.randint(1, 3)
            for _ in range(num_scratches):
                x1, y1 = random.randint(0, img_pil.size[0]), random.randint(0, img_pil.size[1])
                length = random.randint(5, 15)
                angle = random.uniform(0, 2 * np.pi)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                scratch_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                draw.line([(x1, y1), (x2, y2)], fill=scratch_color, width=1)

        # Add dirt spots (40% chance)
        if random.random() < 0.4:
            num_spots = random.randint(1, 5)
            for _ in range(num_spots):
                x = random.randint(0, img_pil.size[0] - 3)
                y = random.randint(0, img_pil.size[1] - 3)
                size = random.randint(1, 3)
                dirt_color = (random.randint(80, 150), random.randint(70, 140), random.randint(60, 130))
                draw.ellipse([x, y, x + size, y + size], fill=dirt_color)

        # Apply fading (reduce contrast slightly)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(random.uniform(0.8, 0.95))

        return np.array(img_pil)

    def _augment_character(self, img_array):
        """Apply realistic augmentations in optimal order"""

        # 1. Apply perspective distortion first (40% chance)
        if random.random() < 0.4:
            img_array = self._apply_perspective_distortion(img_array)

        # 2. Apply weathering effects
        img_array = self._apply_weathering_effects(img_array)

        # 3. Basic geometric transformations
        img_pil = Image.fromarray(img_array)
        original_size = img_pil.size

        # Random rotation (-8° to 8°, increased range)
        angle = random.uniform(-8, 8)
        img_pil = img_pil.rotate(angle, resample=Image.BILINEAR,
                                 fillcolor=random.choice(self.background_colors), expand=False)

        # Random scale (85-115%, wider range)
        scale_factor = random.uniform(0.85, 1.15)
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
        img_pil = img_pil.resize((new_width, new_height), Image.BILINEAR)

        # Random position shift
        if random.random() < 0.7:
            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)
            temp_img = Image.new('RGB', img_pil.size, random.choice(self.background_colors))
            temp_img.paste(img_pil, (x_offset, y_offset))
            img_pil = temp_img

        # 4. Apply lighting effects
        img_array = self._apply_lighting_effects(np.array(img_pil))

        # 5. Blur effects (apply after lighting for realism)
        if random.random() < 0.6:
            blur_type = random.choice(['gaussian', 'median', 'motion'])
            if blur_type == 'gaussian':
                kernel = random.choice([1, 3, 5])
                img_array = cv2.GaussianBlur(img_array, (kernel, kernel), 0)
            elif blur_type == 'median':
                kernel = random.choice([3, 5])
                img_array = cv2.medianBlur(img_array, kernel)
            else:  # motion blur
                kernel_size = random.choice([3, 5, 7])
                angle = random.uniform(0, 180)
                M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
                motion_blur_kernel = np.zeros((kernel_size, kernel_size))
                motion_blur_kernel[kernel_size // 2, :] = 1
                motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
                motion_blur_kernel = motion_blur_kernel / kernel_size
                img_array = cv2.filter2D(img_array, -1, motion_blur_kernel)

        # 6. Add noise (apply after blur)
        if random.random() < 0.6:
            noise_type = random.choice(['gaussian', 'salt_pepper'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(5, 20), img_array.shape).astype(np.int32)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            else:  # salt and pepper noise
                s_vs_p = 0.5
                amount = random.uniform(0.001, 0.01)
                noisy = np.copy(img_array)
                # Salt noise
                num_salt = np.ceil(amount * img_array.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
                noisy[coords[0], coords[1], :] = 255
                # Pepper noise
                num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
                noisy[coords[0], coords[1], :] = 0
                img_array = noisy

        # 7. Final contrast and brightness adjustment
        contrast = random.uniform(0.6, 1.6)  # Wider range
        brightness = random.uniform(-40, 40)  # Wider range
        img_array = np.clip(127.5 + contrast * (img_array - 127.5) + brightness, 0, 255).astype(np.uint8)

        # 8. Ensure final image is correct size
        img_final_aug = Image.fromarray(img_array)
        if img_final_aug.size != self.char_output_size:
            img_final_aug = img_final_aug.resize(self.char_output_size, Image.BILINEAR)

        return np.array(img_final_aug)


if __name__ == "__main__":
    # Initialize the generator with the desired output size for character images
    generator = SyntheticDataGenerator(char_output_size=(48, 48), font_size=24)
    generator.generate_character_images('../data/synth', num_per_char=500)