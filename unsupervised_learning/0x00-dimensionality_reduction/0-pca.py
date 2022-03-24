#!/usr/bin/env python3
"""
Function that performs PCA on a dataset
"""


import numpy as np


def pca(X, var=0.95):
    """
    X is a numpy.ndarray of shape (n, d) where:
      n is the number of data points
      d is the number of dimensions in each point
      all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation
     should maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s
     original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
     of the transformed X
    """
    u, s, vh = np.linalg.svd(X)
    total_variance = np.cumsum(s) / np.sum(s)
    r = (np.argwhere(total_variance >= var))[0, 0]
    W = vh[:r + 1].T
    return W’
