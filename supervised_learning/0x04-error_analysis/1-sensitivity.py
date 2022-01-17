#!/usr/bin/env python3
"""
Function that calculates the sensitivity
for each class in a confusion matrix
"""


import numpy as np


def sensitivity(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the sensitivity of each class
    """
    true_positive = np.diagonal(confusion)
    positives = np.sum(confusion, axis=1)
    sensitivity = true_positive / positives
    return sensitivity
