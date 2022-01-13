#!/usr/bin/env python3
"""
Function that updates a variable using the RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Returns: the updated variable and the new moment, respectively
    """
    new_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated_var = var - alpha * grad / (new_moment ** (1 / 2) + epsilon)
    return updated_var, new_moment
