#!/usr/bin/env python3
"""
Creation of the class GRUCell
"""


import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        x_concat0 = np.concatenate((h_prev, x_t), axis=1)
        r = np.matmul(x_concat0, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        z = np.matmul(x_concat0, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        x_concat1 = np.concatenate((r * h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat1, self.Wh) + self.bh
        h_next = np.tanh(h_next)
        h = (1 - z) * h_prev + z * h_next

        y = np.matmul(h, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h, y
