"""
Contains test cases for the functions defined in the file `ops_utils.py`

We use numpy to validate the output of our functions.

Running this file directly is not recommended for we will not be able to see
the exact root of errors.
"""

from .ops_utils import (
    zero_pad, conv1d
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


if __name__ == '__main__':
    # test all the functions
    test_pad()
    test_conv1d()

    # this will only be executed once all test pass the assertion test :)
    print('\n\nWohoooo! All tests have been passed!!\n\n')
