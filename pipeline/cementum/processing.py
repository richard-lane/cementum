"""
Image processing

"""
from PIL import Image

import numpy as np


def resize(image: Image, size: tuple[int, int]) -> np.ndarray:
    """
    Resize an image for training

    """
    # Resizing uses some filter by default
    # Copy so that we own the data and can resize
    return np.asarray(image.resize(size))


def resample_by_interpolation(
    signal: np.ndarray, input_fs: float, output_fs: float
) -> np.ndarray:
    """
    Resample a signal to a different sample rate using interpolation.

    :param signal: The input signal to be resampled.
    :param input_fs: The original sample rate of the signal.
    :param output_fs: The desired sample rate.

    :returns: The resampled signal.

    """
    # Calculate new length of sample
    scale = output_fs / input_fs
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint=False avoids an off-by-one error and gives less noise in the resampled version
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )


def rotate_and_flip(
    generator: np.random.Generator, images: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate and flip the images and masks.

    :param generator: The random number generator to use.
    :param images: The images to rotate and flip.
    :param masks: The masks to rotate and flip.

    """
    assert len(images) == len(masks), "images and masks must have the same length"

    # Generate random numbers of rotations and flips
    rotations = generator.integers(0, 4, size=len(images))
    flips = generator.choice([True, False], size=len(images), replace=True)

    # Initialize arrays to hold the rotated and flipped images and masks
    rotated_images = np.empty_like(images)
    rotated_masks = np.empty_like(masks)

    for i in range(len(images)):
        # Rotate
        rotated_images[i] = np.rot90(images[i], rotations[i], axes=(0, 1))
        rotated_masks[i] = np.rot90(masks[i], rotations[i], axes=(0, 1))

        # Flip
        if flips[i]:
            rotated_images[i] = np.flip(rotated_images[i], axis=1)
            rotated_masks[i] = np.flip(rotated_masks[i], axis=1)

    return rotated_images, rotated_masks
