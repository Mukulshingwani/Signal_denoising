"""
Defines and implements various routines related to:

- De-blurring
- Denoising
"""

import numpy as np
import utils


# ------------------------------- Denoising --------------------------------- #

def get_denoising_kernel(kernel_size: int) -> np.ndarray:
    # forming our uniform kernel
    kernel = np.ones(kernel_size) / kernel_size
    """
    this forms our required uniform kernel
    Example:
    if size = 3, then kernel would look like
    [1/3, 1/3, 1/3]
    """
    return kernel


def denoise(inp: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    denoising_kernel = get_denoising_kernel(kernel_size)

    # padding and all is taken care inside of conv1d function.
    out = utils.conv1d(inp, denoising_kernel)

    return out


# ------------------------------ De-Blurring -------------------------------- #
