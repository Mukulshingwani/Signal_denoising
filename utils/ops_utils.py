"""

Implements (base) functions and utilities for the following operations:

- 1D Convolution
- Fourier Transform
- Inverse Fourier Transform

"""

from typing import Iterable, List
import numpy as np


def convert_to_np(*args: List[Iterable], dtype='float64') -> np.ndarray:
    converted = []
    for i, item in enumerate(args):
        if isinstance(item, np.ndarray):
            converted.append(item.astype(dtype))
        else:
            converted.append(np.array(item, dtype=dtype))

    return converted


def zero_pad(inp: np.ndarray or Iterable, pad_len: int) -> np.ndarray:
    """
    Pads the given input by `pad_len` on each side.
    Padding is done on each side due to the fact that the middle element
    of the signal indicates its value at `n=0`

    Example:

    >>> i = [1, 2, 3]
    >>> pad_len = 2
    >>> zero_pad(i, pad_len)
    [0, 0, 1, 2, 3, 0, 0]
    """
    inp = list(inp)

    out = [0]*pad_len + inp + [0]*pad_len

    return np.array(out)


def conv1d(inp: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs 1D Convolution operation.
    ------------------------------------

    A Note about Padding:
        The `inp` signal is padded with zeros in such a way that the output
        signal has the same shape as that of the original `inp` signal.

    """

    # flip the kernel
    kernel = kernel[::-1]
    # after flipping the kernel, the convolution operation is equivalent to a
    # a weighted sum of inputs in the window frame of kernel with it sliding
    # all over the input.

    kernel_size = len(kernel)
    len_to_pad = (kernel_size - 1) // 2

    inp = zero_pad(inp, pad_len=len_to_pad)

    out_signal = np.array([])

    # index to start the convolution from.
    curr_idx = 0

    # index in the padded input after which all the entries are padded zeros
    end_idx = len(inp) - len_to_pad

    while curr_idx < end_idx:
        inp_slice = inp[curr_idx:curr_idx + kernel_size]
        entry = np.dot(inp_slice, kernel)
        # or entry = np.sum(inp_slice * kernel)  :)
        out_signal = np.append(out_signal, entry)
        curr_idx += 1

    return out_signal


def fourier_transform(inp: np.ndarray) -> np.ndarray:
    pass


def inverse_fourier_transform(inp: np.ndarray) -> np.ndarray:
    pass


def FT(inp: np.ndarray) -> np.ndarray:
    """
    Alias for `fourier_transform`
    """
    return fourier_transform(inp)


def IFT(inp: np.ndarray) -> np.ndarray:
    """
    Alias for `inverse_fourier_transform`
    """
    return inverse_fourier_transform(inp)
