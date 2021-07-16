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


def discrete_fourier_transform(inp: np.ndarray, num_samples: int = 1000,
                               indices_collide: bool = True) -> np.ndarray:
    """
    a note about `indices_collide`:
        - Should be set to `True` only when the value at zeroth index of `inp`
          array denotes the value of signal at n=0.
        - Should be set to `False` when it doesn't. This is the case with the
          blur kernel h[n] since the question states that the mid element(6/16)
          of h[n] corresponds to n=0.
    """
    signal_len = len(inp)

    if indices_collide:
        # range of summation in the formula
        range_of_summation = np.arange(signal_len).reshape(signal_len, 1)
    else:
        left_limit = -signal_len // 2
        right_limit = signal_len // 2

        # since arange(-n, n) produces [-n,...,n-1]
        range_of_summation = np.arange(left_limit, right_limit+1)\
            .reshape(signal_len, 1)

    # this array contains the indices of samples
    sample_indices = np.arange(num_samples).reshape(1, num_samples)

    # Fourier Transform matrix
    transformation_matrix = np.exp(-2j * np.pi * range_of_summation
                                   * sample_indices / num_samples)

    dft = inp.reshape(1, signal_len) @ transformation_matrix

    return dft.flatten()


def inverse_fourier_transform(inp: np.ndarray) -> np.ndarray:
    """
    - This Function is used for calculating the inverse Fourier Transform
    - formula used is mentioned in our report
    """
    # array length
    N = inp.shape[0]
    # new array of length N [0, N-1], as per the formula
    n = np.arange(N)
    # since k varies from 0 to N-1
    k = n.reshape((N, 1))
    # Calculate the exponential of all elements in the input array
    # as per the formula of inverse DFT (mentioned in our report)
    expo_term = np.exp(2j * np.pi * n * k / N)

    ift = 1 / N * np.dot(expo_term, inp)

    # since we are dealing with real parts only here
    # and imaginary part is quite insignificant:
    return ift.real


def DFT(inp: np.ndarray) -> np.ndarray:
    """
    Alias for `fourier_transform`
    """
    return discrete_fourier_transform(inp)


def IFT(inp: np.ndarray) -> np.ndarray:
    """
    Alias for `inverse_fourier_transform`
    """
    return inverse_fourier_transform(inp)
