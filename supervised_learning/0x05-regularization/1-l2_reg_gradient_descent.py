#!/usr/bin/env python3
"""
Function that updates the weights and biases
of a neural network using gradient descent with
L2 regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Implements gradient descent with L2 regularization
    """
    m = Y.shape[1]
    w_copy = weights.copy()

    for layer in range(L, 0, -1):
        A = cache["A{}".format(layer)]
        A1 = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = A - Y
        else:
            dz = np.matmul(w_copy["W{}".format(layer + 1)].T, dz) * (1 - A ** 2)

            dw = np.matmul(dz, A1.T) / m
            dw_l2 = dw + (lambtha / m) * w_copy['W{}'.format(layer)]
            db = np.sum(dz, axis=1, keepdims=True) / m
            w = w_copy["W{}".format(layer)]
            b = w_copy["b{}".format(layer)]

            weights["W{}".format(layer)] = w - alpha * dw_l2
            weights["b{}".format(layer)] = b - alpha * db
