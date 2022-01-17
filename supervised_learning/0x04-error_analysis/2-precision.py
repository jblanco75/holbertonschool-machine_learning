#!/usr/bin/env python3
"""
Function that calculates the precision
for each class in a confusion matrix
"""


import numpy as np


def precision(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the precision of each class
    """
    true_positive = np.diagonal(confusion)
    positives_false = np.sum(confusion, axis=0)
    precision = true_positive / positives_false
    return precision
