#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Returns integral of a polynomial"""
    integral = [C]
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    for coef in poly:
        if type(coef) is not int and type(coef) is not float:
            return None
    if type(C) is float and C is int:
        C = int(C)
    for exp, coef in enumerate(poly):
        integral.append((1/(exp+1))*coef)
    integral = [int(i) if i % 1 == 0 else i for i in integral]
    return integral
