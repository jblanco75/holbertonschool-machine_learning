#!/usr/bin/env python3
"""
Function that converts a numeric label vector into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        shape = (classes, Y.shape[0])
        one_hot = np.eye(classes)[Y]
        return one_hot.T
    except Exception:
        return None
