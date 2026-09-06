import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance

class DocumentAugmenter:
    """
    Data Augmentation Pipeline designed to prevent overfitting on document forgery datasets.
    Simulates real-world noise, JPEG re-compression, blur, brightness shifts, and spatial warps.
    """
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Resize both image and mask
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # 1. Random Horizontal Flip (50% chance)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 2. Random Affine Transformation (Rotation +/- 5 deg, subtle scale)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(image, angle=angle, translate=[0, 0], scale=scale, shear=0)
            mask = TF.affine(mask, angle=angle, translate=[0, 0], scale=scale, shear=0)

        # 3. Color & Brightness Jitter (Image only)
        if random.random() > 0.4:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        # 4. Subtle Gaussian Blur (Image only, 30% chance)
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Convert to Tensors & Normalize
        img_tensor = TF.to_tensor(image)
        # Normalize with Standard ImageNet Mean & Std
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask_array = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor
