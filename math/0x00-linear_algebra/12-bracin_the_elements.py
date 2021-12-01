#!/usr/bin/env python3
"""Function  that performs element-wise addition,
   subtraction, multiplication, and division."""


def np_elementwise(mat1, mat2):
    """
    Return a tuple containing the element-wise sum,
    difference, product, and quotient, respectively
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
