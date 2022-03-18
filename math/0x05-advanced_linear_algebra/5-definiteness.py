#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""


import numpy as np


def definiteness(matrix):
    """
    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
    calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message
    matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite if the
    matrix is positive definite, positive semi-definite, negative
    semi-definite, negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix) >= 1 and np.array_equal(matrix, matrix.T):
        eigenvalues = np.linalg.eigvals(matrix)

        if np.all(eigenvalues > 0):
            return "Positive definite"
        elif np.all(eigenvalues >= 0):
            return "Positive semi-definite"
        elif np.all(eigenvalues < 0):
            return "Negative definite"
        elif np.all(eigenvalues <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    else:
        return None
