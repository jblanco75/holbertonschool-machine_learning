#!/usr/bin/env python3
"""
Function that creates a confusion matrix
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Returns: a confusion numpy.ndarray of shape
    (classes, classes) with row indices representing
    the correct labels and column indices representing
    the predicted labels
    """
    return labels.T @ logits
