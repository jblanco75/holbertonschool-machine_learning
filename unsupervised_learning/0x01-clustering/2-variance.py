#!/usr/bin/env python3
"""
Function that calculates the total intra-cluster variance for a data set
"""


import numpy as np


def variance(X, C):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
     var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    try:
        n, d = X.shape
        distance = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        min_distance = np.min(distance, axis=0)
        var = np.sum(min_distance ** 2)
        return var
    except Exception:
        return None
