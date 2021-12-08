#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Returns integral of a polynomial"""
    integral = [C]
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    for exp, coef in enumerate(poly):
        integral.append((1/(exp+1))*coef)
    return integral
