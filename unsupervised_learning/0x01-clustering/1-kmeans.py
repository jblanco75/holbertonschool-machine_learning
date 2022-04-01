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
    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                          size=(k, d))
    for i in range(iterations):
        copy = C.copy()
        D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(np.min(X, axis=0),
                                         np.max(X, axis=0),
                                         size=(1, d))
            else:
                C[j] = (X[clss == j]).mean(axis=0)
        D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        if np.all(copy == C):
            return C, clss

    return C, clss
