#!/usr/bin/env python3
"""Function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Returns the sum of 2 matrices"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i, j in enumerate(mat1):
        result.append([])
        for k in range(len(j)):
            result[i].append(mat1[i][k] + mat2[i][k])
    return result
