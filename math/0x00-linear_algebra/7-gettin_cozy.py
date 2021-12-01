#!/usr/bin/env python3
"""Function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates depending axis value"""
    if axis == 0:
        new = [i.copy() for i in mat1]
        new += [i.copy() for i in mat2]
        return new
    elif axis == 1:
        new = [mat1[i] + mat2[i] for i in range(len(mat1))]
        return new
    else:
        return None
