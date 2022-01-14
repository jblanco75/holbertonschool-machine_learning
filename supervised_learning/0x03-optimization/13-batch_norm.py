#!/usr/bin/env python3
"""
Function that normalizes an unactivated
output of a neural network using batch normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Returns: the normalized Z matrix
    """
    variance = np.var(Z, axis=0)
    norm = (Z - np.mean(Z, axis=0)) / np.sqrt(variance + epsilon)
    return norm * gamma + beta
