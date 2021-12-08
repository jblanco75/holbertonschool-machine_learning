#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Returns a list with derivative of coef"""
    deriv = []
    if poly is None:
        return None
    if len(poly) < 2:
        return [0]
    for exp, coef in enumerate(poly):
        deriv.append(exp*coef)
    deriv.pop(0)
    return deriv
