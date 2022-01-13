#!/usr/bin/env python3
"""
Function that normalizes (standardizes) a matrix
"""


def normalize(X, m, s):
    """
    Returns: The normalized X matrix
    """
    X = (X - m) / s
    return X
