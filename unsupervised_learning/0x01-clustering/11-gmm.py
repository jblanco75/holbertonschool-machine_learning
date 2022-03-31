#!/usr/bin/env python3
"""
Function that calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.mixture
    Returns: pi, m, S, clss, bic
      pi is a numpy.ndarray of shape (k,) containing the cluster priors
      m is a numpy.ndarray of shape (k, d) containing the centroid means
      S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices
      clss is a numpy.ndarray of shape (n,) containing the cluster indices
      for each data point
      bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
      value for each cluster size tested
    """
    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_
    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
