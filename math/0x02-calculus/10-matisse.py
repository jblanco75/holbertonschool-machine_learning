#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Returns a list with derivative of coef"""
    deriv = []
    if type(poly) is not list or len(poly) < 1:
        return None
    if len(poly) < 2:
        return [0]
    for coef in poly:
        if type(coef) is not int and type(coef) is not float:
            return None
    for exp, coef in enumerate(poly):
        deriv.append(exp*coef)
    deriv.pop(0)
    return deriv
