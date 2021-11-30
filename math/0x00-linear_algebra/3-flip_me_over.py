#!/usr/bin/env python3
"""Function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    for i in matrix:
        transpose = [[matrix[j][i] for j in range(len(matrix))]
                     for i in range(len(matrix[0]))]
    return transpose
