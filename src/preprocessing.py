# preprocessing.py

import sys
from pathlib import Path

# Dynamically add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import centralized config
from config import IMAGE_SIZE

from PIL import Image
import numpy as np

def preprocess_image(path, size=IMAGE_SIZE):
    """
    Loads an image, resizes it to IMAGE_SIZE, and normalizes pixel values to [0, 1].
    
    Args:
        path (str or Path): Path to the image file.
        size (tuple): Target size for resizing (width, height).
    
    Returns:
        np.ndarray: Normalized image array.
    """
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array
