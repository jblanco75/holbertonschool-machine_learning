#!/usr/bin/env python3
"""Function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    else:
        result = [[sum(i * j for i, j in zip(row, col))
                   for col in zip(*mat2)] for row in mat1]
    return result
