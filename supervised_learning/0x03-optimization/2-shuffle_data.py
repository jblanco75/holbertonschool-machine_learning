#!/usr/bin/env python3
"""
Function that shuffles the data points in two matrices the same way
"""


import numpy as np


def shuffle_data(X, Y):
    """
    Returns: the shuffled X and Y matrices
    """
    shuffled = np.random.permutation(X.shape[0])
    return X[shuffled, :], Y[shuffled, :]
