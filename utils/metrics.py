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


def cos_sim(x: Iterable, y: Iterable) -> np.float64:
    """
    Cosine similarity:
    This can be viewed as Similarity (in Time domain) between
    the signals `x` and `y`
    """
    x, y = convert_to_np(x, y)

    dot = np.dot(x, y)
    inv_scale_f = np.linalg.norm(x) * np.linalg.norm(y)

    return dot / inv_scale_f


def correlation(x: Iterable, y: Iterable) -> np.float64:
    x, y = convert_to_np(x, y)

    return np.sum(np.correlate(x, y))


def energy_dif(x: Iterable, y: Iterable) -> np.float64:
    x, y = convert_to_np(x, y)

    diff = np.sum(np.abs(x**2 - y**2))

    return diff
