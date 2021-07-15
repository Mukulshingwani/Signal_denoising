"""
Defines and implements various routines related to:

- De-blurring
- Denoising
"""

import numpy as np
import utils


# ------------------------------- Denoising --------------------------------- #

def get_denoising_kernel(kernel_size: int) -> np.ndarray:
    # uniform kernel
    kernel = np.ones(kernel_size) / kernel_size
    """
    this forms our required uniform kernel
    Example:
    if size = 3, then kernel would look like
    [1/3, 1/3, 1/3]
    """
    return kernel


def _denoise(inp: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = get_denoising_kernel(kernel_size)

    # padding and all is taken care inside of conv1d function.
    out = utils.conv1d(inp, kernel)

    return out


def denoise(inp: np.ndarray) -> np.ndarray:
    pass


# ------------------------------ De-Blurring -------------------------------- #
