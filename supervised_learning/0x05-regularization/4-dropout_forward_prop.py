#!/usr/bin/env python3
"""
Function that conducts forward propagation using Dropout
"""


import numpy as np


def softmax(z):
    """softmax activation"""
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Returns: a dictionary containing the outputs of each
    layer and the dropout mask used on each layer
    """
    cache = {'A0': X}
    for layer in range(L):
        w = weights['W{}'.format(layer + 1)]
        b = weights['b{}'.format(layer + 1)]
        a = 'A{}'.format(layer + 1)
        k = 'D{}'.format(layer + 1)
        v = np.matmul(w, cache['A{}'.format(layer)]) + b
        dropout = np.random.binomial(1, keep_prob, size=v.shape)

        if layer == L - 1:
            A = softmax(v)
            cache[a] = A

        else:
            A = np.tanh(v)
            cache[k] = dropout
            cache[a] = (A * cache[k]) / keep_prob
    return cache
