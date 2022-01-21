#!/usr/bin/env python3
"""
Function that updates the weights of a neural
network with Dropout regularization using gradient descent
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with
    Dropout regularization using gradient descent
    """
    m = Y.shape[1]
    w_copy = weights.copy()
    for layer in range(L, 0, -1):
        A = cache["A{}".format(layer)]
        A1 = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = A - Y
        else:
            dz = np.matmul(w_copy["W{}".format(layer + 1)].T,
                           dz) * (1 - A ** 2)
            dz *= cache["D{}".format(layer)] / keep_prob
        dw = np.matmul(dz, A1.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        w = w_copy["W{}".format(layer)]
        b = w_copy["b{}".format(layer)]
        weights["W{}".format(layer)] = w - alpha * dw
        weights["b{}".format(layer)] = b - alpha * db
