#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
      n is the number of data points
      d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations,
    your function should return
    Initialize the cluster centroids using a multivariate uniform
    distribution (based on0-initialize.py)
    If a cluster contains no data points during the update step,
    reinitialize its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
      C is a numpy.ndarray of shape (k, d) containing the centroid
      means for each cluster
      clss is a numpy.ndarray of shape (n,) containing the index of
      the cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    for i in range(iterations):
        C_copy = np.copy(C)
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        clss = np.argmin(distances, axis=0)
        for k in range(C.shape[0]):
            if (X[clss == k].size == 0):
                C[k, :] = np.random.uniform(low, high, size=(1, d))
            else:
                C[k, :] = (X[clss == k].mean(axis=0))
        if (C_copy == C).all():
            return (C, clss)
    return (C, clss)
