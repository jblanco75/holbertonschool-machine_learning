#!/usr/bin/env python3
"""
Function that determines if a markov chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
      P[i, j] is the probability of transitioning from state i to state j
      n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    if np.all(np.diag(P) != 1):
        return False
    if np.all(np.diag(P) == 1):
        return True
    for i in range(n):
        if np.any(P[i, :] == 1):
            continue
        break

    II = P[:i, :i]
    Id = np.identity(n - i)
    R = P[i:, :i]
    Q = P[i:, i:]

    try:
        F = np.linalg.inv(Id - Q)
    except Exception:
        return False
    FR = np.matmul(F, R)
    P_b = np.zeros((n, n))
    P_b[:i, :i] = P[:i, :i]
    P_b[i:, :i] = FR
    Q_b = P_b[i:, i:]
    if np.all(Q_b == 0):
        return True
