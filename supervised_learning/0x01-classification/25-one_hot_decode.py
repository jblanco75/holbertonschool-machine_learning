#!/usr/bin/env python3
"""
Function that converts a one-hot matrix into a vector of labels
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        one_hot = np.argmax(one_hot, axis=0)
        return one_hot
    except Exception:
        return None
