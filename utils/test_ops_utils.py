"""
Contains test cases for the functions defined in the file `ops_utils.py`

We use numpy to validate the output of our functions.

Running this file directly is not recommended for we will not be able to see
the exact root of errors.
"""

from .ops_utils import (
    discrete_fourier_transform, inverse_fourier_transform, zero_pad, conv1d
    )
import numpy as np


def test_pad():
    x = [2, 4, 6, 4, 2]
    inp = np.array(x)

    pad_len = 3

    expected_out = [0, 0, 0, 2, 4, 6, 4, 2, 0, 0, 0]

    out = zero_pad(inp, pad_len)

    assert np.allclose(out, expected_out)


def test_conv1d():
    a = [1, 2, 3]
    b = [1, 2, 3, 4, 5, 6]
    a = np.array(a)
    b = np.array(b)
    assert np.allclose(conv1d(b, a), np.convolve(b, a, mode='same'))


def test_Discrete_Fourier_Transform():
    inp = [1, 2, 3, 4, 5, 6, 7]
    inp = np.array(inp)
    our_output = discrete_fourier_transform(inp, 1000, True)
    fft_output = np.fft.fft(inp, 1000)
    assert np.allclose(our_output, fft_output)


def test_Inverse_Fourier_Transform():
    inp = [1, 2, 3, 4, 5, 6, 7]
    inp = np.array(inp)
    our_output = inverse_fourier_transform(inp)
    ifft_output = np.fft.ifft(inp).real
    assert np.allclose(our_output, ifft_output)


def test_sync():
    """
    test for checking the syncing of the function by applying
    inverse fourier transform to the discrete fourier transform
    of the signal to obtain back the original input signal
    """
    inp = [1, 2, 3, 4, 5]
    inp = np.array(inp)
    fft_of_inp = discrete_fourier_transform(inp, 1000, True)
    recovered_inp = inverse_fourier_transform(fft_of_inp)
    assert np.allclose(inp, recovered_inp[:len(inp)])


if __name__ == '__main__':
    # test all the functions
    test_pad()
    test_conv1d()

    # this will only be executed once all test pass the assertion test :)
    print('\n\nWohoooo! All tests have been passed!!\n\n')
