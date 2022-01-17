#!/usr/bin/env python3
"""
Function that calculates the specificity
for each class in a confusion matrix
"""


import numpy as np


def specificity(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """
    false_positives = np.sum(confusion, axis=0) - np.diagonal(confusion)
    false_negatives = np.sum(confusion, axis=1) - np.diagonal(confusion)
    true_positives = np.diagonal(confusion)
    true_negatives = np.sum(confusion) - (false_positives + false_negatives
                                          + true_positives)
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity
