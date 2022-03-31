#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""


import sklearn.cluster


def kmeans(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss
      C is a numpy.ndarray of shape (k, d) containing the centroid means
      for each cluster
      clss is a numpy.ndarray of shape (n,) containing the index of the
      cluster in C that each data point belongs to
    """
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = km.cluster_centers_
    clss = km.labels_
    return C, clss
