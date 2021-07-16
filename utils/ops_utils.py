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
            # checking whether `item` is of the specified type i.e. ndarray
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

    # forming our desired output list with proper padding done
    out = [0]*pad_len + inp + [0]*pad_len

    # returning it by converting it into a numpy array
    return np.array(out)


def conv1d(inp: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs 1D Convolution operation.
    ------------------------------------

    A Note about Padding:
        The `inp` signal is padded with zeros in such a way that the output
        signal has the same shape as that of the original `inp` signal.

    """

    # flipping the kernel
    kernel = kernel[::-1]
    kernel_size = len(kernel)
    len_to_pad = (kernel_size - 1) // 2

    # calling out the function formed above to pad the input
    inp = zero_pad(inp, pad_len=len_to_pad)

    # array to store our final output signal
    out_signal = np.array([])

    # index to start the convolution from.
    curr_idx = 0

    # index in the padded input after which all the entries are padded zeros
    end_idx = len(inp) - kernel_size

    # after flipping the kernel, we are basically sliding our kernel over
    # our input and taking the sum of element wise multiplication obtained
    # in each traversal , which will finally give us the required Convolution
    while curr_idx <= end_idx:
        inp_slice = inp[curr_idx:curr_idx + kernel_size]
        print(inp_slice)
        entry = np.dot(inp_slice, kernel)
        # taking the dot product
        # OR
        # entry = np.sum(inp_slice * kernel)  :)
        out_signal = np.append(out_signal, entry)
        curr_idx += 1

    return out_signal


def fourier_transform(inp: np.ndarray) -> np.ndarray:
    pass


def inverse_fourier_transform(inp: np.ndarray) -> np.ndarray:
    """
    - This Function is used for calculating the inverse Fourier Transform
    - formula used is mentioned in our report
    """
    input_sig = np.asarray(inp, dtype=float)
    # array length
    N = input_sig.shape[0]
    # new array of length N [0, N-1], as per the formula
    n = np.arange(N)
    # since k varies from 0 to N-1
    k = n.reshape((N, 1))
    # Calculate the exponential of all elements in the input array
    # as per the formula of inverse DFT (mentioned in our report)
    expo_term = np.exp(2j * np.pi * n * k / N)

    return 1 / N * np.dot(expo_term, input_sig)


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
