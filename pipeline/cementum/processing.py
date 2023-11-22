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
