"""
Image processing

"""
import cv2
from PIL import Image

import numpy as np


def resize(image: Image, size: tuple[int, int]) -> np.ndarray:
    """
    Resize an image for training

    """
    # Resizing uses some filter by default
    # Copy so that we own the data and can resize
    return np.asarray(image.resize(size))
