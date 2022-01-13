#!/usr/bin/env python3
"""
Function that updates a variable using the gradient
descent with momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Returns: the updated variable and the new moment, respectively
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated_var = var - alpha * momentum
    return updated_var, momentum
