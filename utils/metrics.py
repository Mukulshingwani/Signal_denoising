"""
Contains utility functions to measure similarity and distances between
two given vectors.
"""


from typing import Iterable
import numpy as np
from .ops_utils import convert_to_np


def l2_dist(x: Iterable, y: Iterable) -> np.float64:
    """
    Returns Euclidean Distance.
    """
    x, y = convert_to_np(x, y)

    return np.sum((x-y)**2)


def l1_dist(x: Iterable, y: Iterable) -> np.float64:
    """
    Returns Manhattan Distance.
    """
    x, y = convert_to_np(x, y)

    return np.sum((np.abs(x-y)))


def correlation(x: Iterable, y: Iterable) -> np.float64:
    x, y = convert_to_np(x, y)

    return np.sum(np.correlate(x, y))


def energy_diff(x: Iterable, y: Iterable) -> np.float64:
    x, y = convert_to_np(x, y)

    diff = np.sum(y**2) - np.sum(x**2)

    return diff
