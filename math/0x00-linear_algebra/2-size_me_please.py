#!/usr/bin/env python3
"""Function that defines shape of a matrix"""


def matrix_shape(matrix):
    """Calculates shape of a matrix"""
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
