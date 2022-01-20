#!/usr/bin/env python3
"""
Function that calculates the cost of a
neural network with L2 regularization
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Returns: the cost of the network
    accounting for L2 regularization
    """
    W_norm = 0
    for i in range(L):
        w = weights['W{}'.format(i + 1)]
        W_norm += np.linalg.norm(w)
        L2 = (lambtha / (2 * m)) * W_norm
    return cost + L2
