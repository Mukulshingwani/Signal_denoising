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

def deblur(blurred_signal: np.ndarray, blur_kernel: np.ndarray) -> np.ndarray:
    signal_size = blurred_signal.shape

    dft_inp = utils.DFT(blurred_signal)
    dft_blur_kernel = utils.DFT(blur_kernel, indices_collide=False)

    recovered_signal = utils.IFT(dft_inp / dft_blur_kernel)

    # now this recovered signal consists of the original (unblurred) signal
    # But it is repeated after every `signal_size` indices
    # So, if we just consider the values from index 0 to `signal_size`-1,
    # they must contain the original (unblurred) signal.
    return recovered_signal[:signal_size]
