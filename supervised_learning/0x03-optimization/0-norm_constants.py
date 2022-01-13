#!/usr/bin/env python3
"""
Function that calculates the normalization
(standardization) constants of a matrix
"""


import numpy as np


def normalization_constants(X):
    """
    Returns: the mean and standard
    deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    st_dev = np.std(X, axis=0)
    return mean, st_dev
