#!/usr/bin/env python3
"""
Function that calculates the F1 score of a confusion matrix
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the F1 score of each class
    """
    Sensitivity = sensitivity(confusion)
    Precision = precision(confusion)
    f1_score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    return f1_score
