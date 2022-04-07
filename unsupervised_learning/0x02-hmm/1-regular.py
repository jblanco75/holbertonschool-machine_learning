#!/usr/bin/env python3
"""
Function that determines the steady state probabilities
of a regular markov chain:
"""


import numpy as np


def regular(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
      P[i, j] is the probability of transitioning from state i to state j
      n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if type(P) is not np.ndarray or P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    s = np.full(n, (1 / n))[np.newaxis, ...]
    P_pow = P.copy()
    while True:
        s_prev = s
        s = np.matmul(s, P)
        P_pow = P * P_pow
        if np.any(P_pow <= 0):
            return (None)
        if np.all(s_prev == s):
            return (s)
